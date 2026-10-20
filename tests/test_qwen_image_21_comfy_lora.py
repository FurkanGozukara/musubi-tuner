"""Qwen-Image 2.1 LoRAs for ComfyUI / SwarmUI (kohya-ss/musubi-tuner #1163).

Comfy-Org's Qwen-Image 2.1 checkpoints fuse each block's MLP gate and up layers (``img_mlp.gate_up``); ComfyUI maps LoRA
weights onto the two halves only for diffusers keys (``transformer.<module>.lora_A.weight``), so with the trainer's own
``lora_unet_`` keys ``img_mlp.proj`` and ``img_mlp.gate_layer`` were skipped (5 of 7 layers per block applied; ComfyUI
master's own loader applies all 7 to the converted file, the halves as slices with their DoRA scales). Each checkpoint
therefore gets a ``<name>.comfy.safetensors`` copy; old checkpoints are removed with it.
"""

import argparse
import types

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.networks import lora_qwen_image_21
from musubi_tuner.qwen_image_21.qwen_image_21_model import QwenImage21Transformer2DModel
from musubi_tuner.qwen_image_21_train_network import QwenImage21NetworkTrainer
from musubi_tuner.training.trainer_base import comfy_lora_path

TINY = dict(in_channels=4, out_channels=4, num_layers=2, attention_head_dim=8, num_attention_heads=2, context_in_dim=12,
            axes_dims_rope=[2, 2, 4])
SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".alpha", ".dora_scale")


def test_each_checkpoint_gets_a_comfyui_copy_with_diffusers_keys(tmp_path):
    model = QwenImage21Transformer2DModel(**TINY)
    network = lora_qwen_image_21.create_arch_network(1, 2, 2, None, [], model, use_dora="True")
    path = tmp_path / "run-000025.safetensors"
    metadata = {"ss_network_dim": "2", "ss_epoch": "25"}
    network.save_weights(str(path), torch.float32, metadata)

    accelerator = types.SimpleNamespace(is_main_process=True, print=lambda *args, **kwargs: None)
    args = argparse.Namespace(output_dir=str(tmp_path), huggingface_repo_id=None)
    QwenImage21NetworkTrainer().on_post_save(args, accelerator, network, None, path.name, torch.float32, metadata, False)

    comfy_file = comfy_lora_path(str(path))
    assert comfy_file.endswith("run-000025.comfy.safetensors")
    weights = load_file(comfy_file)
    modules = {name for name, _ in model.named_modules()}
    converted = {}
    for key in weights:
        assert key.startswith("transformer."), key
        body = key[len("transformer.") :]
        suffix = next(s for s in SUFFIXES if body.endswith(s))
        converted.setdefault(body[: -len(suffix)], set()).add(suffix)
    assert set(converted) <= modules  # real module paths, no "img.mlp.gate.layer"
    assert len(converted) == len(network.unet_loras) == 14  # 7 projections per block
    assert {"transformer_blocks.0.img_mlp.proj", "transformer_blocks.0.img_mlp.gate_layer"} <= set(converted)
    # diffusers keys carry no alpha: alpha / rank is folded into lora_A and lora_B (sqrt each); DoRA magnitudes travel along
    assert all(suffixes == {".lora_A.weight", ".lora_B.weight", ".dora_scale"} for suffixes in converted.values())
    original = load_file(str(path))
    name = "lora_unet_transformer_blocks_0_img_mlp_gate_layer"
    scale = original[f"{name}.alpha"] / original[f"{name}.lora_down.weight"].shape[0]
    expected = original[f"{name}.lora_up.weight"] @ original[f"{name}.lora_down.weight"] * scale
    module = "transformer.transformer_blocks.0.img_mlp.gate_layer"
    torch.testing.assert_close(weights[f"{module}.lora_B.weight"] @ weights[f"{module}.lora_A.weight"], expected)
    with safe_open(comfy_file, framework="pt") as handle:
        assert handle.metadata()["ss_epoch"] == "25"
