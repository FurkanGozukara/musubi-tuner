"""Anima full fine-tuning (anima_train.py on training/dense_trainer.py): the parser's defaults are a valid dense recipe,
LoRA-only and unsafe combinations are refused, the LLM adapter stays frozen unless asked, and a checkpoint is written in
the released layout (net. prefix, the anima-base-v1.0 tensors) that load_dit reads back weight for weight."""

import os
import types

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner import anima_train as ft
from musubi_tuner.anima import anima_model, anima_utils
from musubi_tuner.anima.anima_model import Anima, AnimaParams
from musubi_tuner.training.dense_trainer import DenseFinetuneModule
from musubi_tuner.training.parser_common import read_config_from_file

REQUIRED = ["--dit", "dit.safetensors", "--dataset_config", "d.toml", "--output_dir", "out", "--output_name", "o"]
PARAMS = dict(
    model_channels=64, num_blocks=3, num_heads=2, crossattn_emb_channels=32, adaln_lora_dim=16, adapter_dim=32, adapter_layers=2,
    adapter_heads=2, adapter_vocab_size=50,
)
BASE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Training_Models_Anima", "anima-base-v1.0.safetensors")


def parse(*extra):
    parser = ft.create_parser()
    return read_config_from_file(parser.parse_args([*REQUIRED, *extra]), parser)


def _tiny() -> Anima:
    torch.manual_seed(0)
    model = Anima(AnimaParams(**PARAMS))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.05)
    return model.to(torch.bfloat16)


def test_parser_defaults_form_a_valid_dense_recipe():
    args = parse()
    assert args.full_bf16 and args.fused_backward_pass and args.mem_eff_save and args.gradient_checkpointing
    assert args.network_module is None and args.optimizer_type == "Adafactor" and args.max_grad_norm == 0.0
    assert args.learning_rate == 1e-5 and not args.train_llm_adapter
    ft.AnimaTrainer()._validate_full_finetune_args(args)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--fp8_base"], "BF16 weights"),
        (["--fp8_scaled"], "BF16 weights"),
        (["--convrot_int8"], "BF16 weights"),  # INT8 trains LoRA only (an INT8 file is refused by load_dit)
        (["--no-full_bf16"], "--full_bf16"),
        (["--block_swap_h2d_only"], "h2d_only"),
        (["--network_weights", "x.safetensors"], "LoRA initialization"),
        (["--save_precision", "fp32"], "BF16"),
        (["--max_grad_norm", "1.0"], "max_grad_norm 0"),
        (["--block_swap_trainable_ring"], "requires --blocks_to_swap"),
    ],
)
def test_validation_rejects_unsupported_combinations(extra, message):
    with pytest.raises(ValueError, match=message):
        ft.AnimaTrainer()._validate_full_finetune_args(parse(*extra))


@pytest.mark.parametrize("train_adapter", [False, True])
def test_the_llm_adapter_trains_only_when_asked(train_adapter):
    """The model card: "Don't train the LLM adapter"; --train_llm_adapter puts it back in."""
    args = parse(*(["--train_llm_adapter"] if train_adapter else []))
    args.gradient_checkpointing = False
    model = _tiny()
    network = ft.AnimaTrainer()._build_network(args, None, model, None, torch.bfloat16)
    groups, _ = network.prepare_optimizer_params(1e-5)
    trained = {id(p) for p in groups[0]["params"]}
    adapter = [p for p in model.llm_adapter.parameters()]
    assert all((id(p) in trained) == train_adapter for p in adapter)
    assert all(id(p) in trained for p in model.blocks.parameters()) and id(model.x_embedder.proj[1].weight) in trained


def test_the_checkpoint_has_the_released_layout_and_loads_back(tmp_path):
    model = _tiny()
    path = tmp_path / "anima_ft.safetensors"
    module = DenseFinetuneModule(model, ft.AnimaTrainer().dense_state_dict_for_save, model_name="Anima")
    module.save_weights(str(path), torch.bfloat16, {"ss_training_type": "full_finetune"})
    saved = load_file(str(path))
    assert all(key.startswith("net.") for key in saved) and all(value.dtype == torch.bfloat16 for value in saved.values())
    with safe_open(str(path), "pt") as handle:
        assert handle.metadata()["ss_training_type"] == "full_finetune"
    reloaded = anima_utils.load_dit("cpu", str(path), "torch", False, "cpu", torch.bfloat16, params=AnimaParams(**PARAMS))
    for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert torch.equal(a, b), name


@pytest.mark.skipif(not os.path.isfile(BASE_FILE), reason="needs the Anima Base checkpoint")
def test_the_saved_layout_matches_the_released_file():
    """The released Base file's 685 keys and shapes are exactly what the full-size model saves."""
    with safe_open(BASE_FILE, "pt") as handle:
        released = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
    with torch.device("meta"):
        model = Anima()
    saved = anima_model.comfy_state_dict(model.state_dict())
    assert {key: tuple(value.shape) for key, value in saved.items()} == released


def test_full_fine_tuning_runs_the_loop_whose_hooks_the_dense_trainer_overrides():
    from musubi_tuner.anima_train_network import AnimaNetworkTrainer, AnimaTrainerMixin
    from musubi_tuner.training import trainer_base, training_loop

    assert ft.AnimaTrainer.train is trainer_base.NetworkTrainer.train
    assert AnimaNetworkTrainer.train is training_loop.train
    assert issubclass(ft.AnimaTrainer, AnimaTrainerMixin) and not issubclass(ft.AnimaTrainer, AnimaNetworkTrainer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainable block-swap ring needs CUDA")
@pytest.mark.parametrize("activation_cpu_offload", [False, True], ids=["gpu_activations", "cpu_activations"])
def test_trainable_ring_backward_matches_the_resident_model(activation_cpu_offload):
    """Each checkpointed block's backward recompute waits for its ring slot (recompute_wait, as MOSS-Image)."""
    import copy

    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

    base = _tiny()
    cuda = torch.device("cuda")
    dense, ring = copy.deepcopy(base).to(cuda), copy.deepcopy(base)
    dense.enable_gradient_checkpointing()
    ring.enable_gradient_checkpointing(activation_cpu_offload)
    config = BlockSwapConfig(device=cuda, supports_backward=True, use_pinned_memory=True, trainable_ring=True, ring_size=2)
    ring.enable_block_swap(1, config)
    assert all(block.recompute_wait is not None for block in ring.blocks)
    ring.move_to_device_except_swap_blocks(cuda)
    ring.prepare_block_swap_before_forward()
    ring.switch_block_swap_for_training()
    dense.train()
    ring.train()
    torch.manual_seed(1)
    x = torch.randn(1, 16, 4, 4, device=cuda, dtype=torch.bfloat16)
    t = torch.tensor([0.4], device=cuda)
    states = torch.randn(1, 6, 32, device=cuda, dtype=torch.bfloat16)
    ids = torch.randint(2, 50, (1, 4), device=cuda)
    try:
        outputs = []
        for model in (dense, ring):
            out = model(x, t, states, t5_ids=ids)
            out.float().pow(2).mean().backward()
            outputs.append(out.detach().float().cpu())
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0.02, atol=0.02)
        for (name, expected), (_, actual) in zip(dense.named_parameters(), ring.named_parameters()):
            if expected.grad is None:
                continue
            assert actual.grad is not None, name
            torch.testing.assert_close(actual.grad.float().cpu(), expected.grad.float().cpu(), rtol=0.05, atol=0.01)
    finally:
        ring.offloader.offload_to_cpu(ring.blocks)
