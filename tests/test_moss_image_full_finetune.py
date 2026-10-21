"""MOSS-Image full fine-tuning (moss_image_train.py on training/dense_trainer.py): the parser's defaults are a valid
dense recipe, LoRA-only and unsafe combinations are refused, and a checkpoint is written in the vendor / ComfyUI layout
(fused proj.qkv, rope.bands) that load_dit's split reads back weight for weight."""

import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner import moss_image_train as ft
from musubi_tuner.moss_image import moss_model
from musubi_tuner.moss_image.moss_model import MossImage, MossImageParams
from musubi_tuner.training.dense_trainer import DenseFinetuneModule
from musubi_tuner.training.parser_common import read_config_from_file

REQUIRED = ["--dit", "dit.safetensors", "--dataset_config", "d.toml", "--output_dir", "out", "--output_name", "o"]
PARAMS = dict(dim=64, ffn_dim=96, num_heads=2, num_layers=3, text_dim=48, freq_dim=16, latent_channels=8)
BASE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Training_Models_MOSS_Image", "moss_image_base_bf16.safetensors")


def parse(*extra):
    parser = ft.create_parser()
    return read_config_from_file(parser.parse_args([*REQUIRED, *extra]), parser)


def test_parser_defaults_form_a_valid_dense_recipe():
    args = parse()
    assert args.full_bf16 and args.fused_backward_pass and args.mem_eff_save and args.gradient_checkpointing
    assert args.network_module is None and args.optimizer_type == "Adafactor" and args.max_grad_norm == 0.0
    assert args.learning_rate == 1e-5
    ft.MossImageTrainer()._validate_full_finetune_args(args)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--fp8_base"], "BF16 weights"),
        (["--fp8_scaled"], "BF16 weights"),
        (["--no-full_bf16"], "--full_bf16"),
        (["--block_swap_h2d_only"], "h2d_only"),
        (["--network_weights", "x.safetensors"], "LoRA initialization"),
        (["--save_precision", "fp32"], "BF16"),
        (["--optimizer_type", "AdamW"], "Adafactor"),
        (["--max_grad_norm", "1.0"], "max_grad_norm 0"),
        (["--adafactor_triton", "--no-fused_backward_pass"], "requires --fused_backward_pass"),
        (["--block_swap_trainable_ring"], "requires --blocks_to_swap"),
        (["--no-fused_backward_pass", "--optimizer_type", "AdamW", "--blocks_to_swap", "4"], "requires --fused_backward_pass"),
    ],
)
def test_validation_rejects_unsupported_combinations(extra, message):
    with pytest.raises(ValueError, match=message):
        ft.MossImageTrainer()._validate_full_finetune_args(parse(*extra))


def test_non_fused_optimizer_is_allowed_without_block_swap():
    ft.MossImageTrainer()._validate_full_finetune_args(parse("--no-fused_backward_pass", "--optimizer_type", "AdamW8bit", "--max_grad_norm", "1.0"))


def test_the_checkpoint_has_the_vendor_layout_and_loads_back(tmp_path):
    torch.manual_seed(0)
    params = MossImageParams(**PARAMS)
    model = MossImage(params, "torch", False).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.02)
    path = tmp_path / "moss_ft.safetensors"
    module = DenseFinetuneModule(model, ft.MossImageTrainer().dense_state_dict_for_save, model_name="MOSS-Image")
    module.save_weights(str(path), torch.bfloat16, {"ss_training_type": "full_finetune"})

    saved = load_file(str(path))
    assert "layers.0.proj.qkv.weight" in saved and "layers.0.proj.q.weight" not in saved
    assert saved["layers.0.proj.qkv.weight"].shape == (3 * params.dim, params.dim)
    assert saved["rope.bands"].dtype == torch.float32 and saved["rope.bands"].shape == (params.dim // params.num_heads // 8,)
    with safe_open(str(path), "pt") as handle:
        assert handle.metadata()["ss_training_type"] == "full_finetune"

    reloaded = MossImage(params, "torch", False).to(torch.bfloat16)
    reloaded.load_state_dict(moss_model.split_fused_qkv(dict(saved)), strict=True)
    for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert torch.equal(a, b), name


@pytest.mark.skipif(not os.path.isfile(BASE_FILE), reason="needs the MOSS-Image Base checkpoint")
def test_the_saved_layout_matches_the_released_file():
    """The released Base file's keys are exactly what fuse_qkv writes for the full-size model (379 keys), rope.bands
    included bit for bit."""
    with safe_open(BASE_FILE, "pt") as handle:
        keys = set(handle.keys())
        bands = handle.get_tensor("rope.bands")
    with torch.device("meta"):
        model = MossImage(MossImageParams(), "torch", False)
    fused = moss_model.fuse_qkv(model.state_dict(), model.params)
    assert set(fused) == keys
    assert torch.equal(fused["rope.bands"], bands)


def test_full_fine_tuning_runs_the_loop_whose_hooks_the_dense_trainer_overrides():
    """The extended loop (training_loop.train, the LoRA trainer's) builds its network from --network_module inline, so
    the dense trainer's _build_network / _prepare_with_accelerator / _register_hooks_and_resume act only on
    trainer_base's loop: the first GPU run of moss_image_train.py on the extended loop stopped at
    importlib.import_module(None)."""
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer, MossImageTrainerMixin
    from musubi_tuner.training import trainer_base, training_loop

    assert ft.MossImageTrainer.train is trainer_base.NetworkTrainer.train
    assert MossImageNetworkTrainer.train is training_loop.train
    assert issubclass(ft.MossImageTrainer, MossImageTrainerMixin) and not issubclass(ft.MossImageTrainer, MossImageNetworkTrainer)
