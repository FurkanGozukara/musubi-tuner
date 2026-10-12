"""Numerical ring parity on the actual multimodal K6 training path.

Small CUDA-only BF16/SDPA models; this checks correctness, not Pro-model memory
capacity or throughput. Cases cover resident-prefix/all-streamed layouts,
activation CPU offload, and frozen first/middle/last blocks through two
accumulated optimizer updates.
"""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner import kandinsky6_train as ft
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from test_kandinsky6_reference import make_batch, parse_train, r2av_trainer, randomize, tiny_dit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for trainable ring storage")
@pytest.mark.parametrize("swapped", [2, 3])
@pytest.mark.parametrize("frozen", [None, 0, 2, 4])
@pytest.mark.parametrize("activation_cpu_offload", [False, True], ids=["gpu_activations", "cpu_activations"])
def test_multimodal_dense_ring_fused_accumulation_matches_resident(swapped, frozen, activation_cpu_offload):
    base = randomize(tiny_dit(reference_slots=2, num_visual_blocks=5), seed=33).to(dtype=torch.bfloat16)
    if frozen is not None:
        base.visual_transformer_blocks[frozen].requires_grad_(False)
    dense = copy.deepcopy(base).cuda()
    ring = copy.deepcopy(base)
    dense.enable_gradient_checkpointing()
    ring.enable_gradient_checkpointing(activation_cpu_offload)
    ring.enable_block_swap(
        swapped,
        BlockSwapConfig(
            device=torch.device("cuda"), supports_backward=True, use_pinned_memory=True, trainable_ring=True, ring_size=2
        ),
    )
    ring.move_to_device_except_swap_blocks(torch.device("cuda"))
    ring.prepare_block_swap_before_forward()
    ring.switch_block_swap_for_training()
    trainer = r2av_trainer(ft.Kandinsky6Trainer, slots=2)
    args = parse_train("--task", "r2av")
    args.gradient_checkpointing = True
    args.video_only = False
    args.audio_loss_weight = 1.0
    args.independent_time = False
    accelerator = SimpleNamespace(sync_gradients=False, num_processes=1, device=torch.device("cuda"), autocast=nullcontext)
    for model in (dense, ring):
        network = ft.Kandinsky6FullFinetuneModule(model)
        params, _ = network.prepare_optimizer_params(unet_lr=1e-3)
        _, _, optimizer, _, _ = trainer.get_optimizer(args, params)
        trainer._install_fused_optimizer(args, accelerator, optimizer)
    try:
        for step in range(4):
            accelerator.sync_gradients = step % 2 == 1
            before = {name: parameter.detach().cpu().clone() for name, parameter in ring.named_parameters()}
            torch.manual_seed(80 + step)
            batch = make_batch([torch.randn(2, 16, 2, 2)])
            latents = torch.randn(1, 16, 2, 2, 2, device="cuda", dtype=torch.bfloat16)
            noise = torch.randn_like(latents)
            losses = []
            for model in (dense, ring):
                torch.manual_seed(800 + step)
                loss, _ = trainer.process_batch(
                    args,
                    accelerator,
                    model,
                    None,
                    batch,
                    latents,
                    noise,
                    None,
                    torch.bfloat16,
                    torch.bfloat16,
                    None,
                    step,
                )
                (loss / 2).backward()
                losses.append(loss.detach().cpu())
            torch.testing.assert_close(losses[0], losses[1], rtol=0.005, atol=0.005)
            if accelerator.sync_gradients:
                ring.offloader.commit()
                for (name, expected), (_, actual) in zip(dense.named_parameters(), ring.named_parameters()):
                    torch.testing.assert_close(
                        actual.cpu(), expected.cpu(), rtol=0.01, atol=0.003, msg=lambda message, name=name: name + ": " + message
                    )
            else:
                # Fused hooks must preserve weights and accumulated gradients until sync.
                torch.cuda.synchronize()
                for name, parameter in ring.named_parameters():
                    assert torch.equal(parameter.detach().cpu(), before[name]), name
                assert any(parameter.grad is not None for parameter in ring.parameters())
    finally:
        ring.offloader.offload_to_cpu(ring.visual_transformer_blocks)
