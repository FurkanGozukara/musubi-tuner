from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint
from transformers import Adafactor

from musubi_tuner import kandinsky6_train as ft
from musubi_tuner.kandinsky6.dit import DiffusionTransformer3D
from musubi_tuner.training.parser_common import read_config_from_file

REQUIRED = ["--dit", "dit.safetensors", "--dataset_config", "d.toml", "--output_dir", "out", "--output_name", "o"]


def tiny_dit(num_visual_blocks: int = 2) -> DiffusionTransformer3D:
    return DiffusionTransformer3D(
        in_visual_dim=16,
        out_visual_dim=16,
        in_text_dim=8,
        in_text_dim2=6,
        time_dim=8,
        patch_size=(1, 1, 1),
        model_dim=12,
        ff_dim=20,
        num_text_blocks=1,
        num_visual_blocks=num_visual_blocks,
        axes_dims=(4, 4, 4),
        visual_cond=True,
        is_multimodal=True,
        in_audio_dim=40,
        out_audio_dim=40,
        model_dim_a=12,
        time_dim_a=8,
        ff_dim_a=20,
        axes_dims_a=(4, 4, 4),
        attention_engine="sdpa",
        text_token_padding=True,
        ca_rope=True,
        cross_gates=True,
        fix_modulation=True,
        visual_token_type_num_embeddings=2,
    )


def parse(*extra):
    parser = ft.create_parser()
    args = read_config_from_file(parser.parse_args([*REQUIRED, *extra]), parser)
    args.dit_dtype = None
    return args


def test_parser_defaults_form_a_valid_dense_recipe():
    args = parse()
    assert args.full_bf16 and args.fused_backward_pass and args.mem_eff_save
    assert args.network_module is None
    assert args.optimizer_type == "Adafactor" and args.max_grad_norm == 0.0
    ft.Kandinsky6Trainer()._validate_full_finetune_args(args)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--convrot_int8"], "BF16 weights"),
        (["--fp8_base"], "BF16 weights"),
        (["--no-full_bf16"], "--full_bf16"),
        (["--block_swap_h2d_only"], "h2d_only"),
        (["--network_weights", "x.safetensors"], "LoRA initialization"),
        (["--save_precision", "fp32"], "BF16"),
        (["--optimizer_type", "AdamW"], "Adafactor"),
        (["--max_grad_norm", "1.0"], "max_grad_norm 0"),
        (["--adafactor_triton", "--no-fused_backward_pass"], "requires --fused_backward_pass"),
        (["--block_swap_trainable_ring"], "requires --blocks_to_swap"),
        (["--no-fused_backward_pass", "--optimizer_type", "AdamW", "--blocks_to_swap", "4"], "requires --fused_backward_pass"),
        (["--k6_freeze_blocks", "3-1"], "invalid block range"),
    ],
)
def test_validation_rejects_unsupported_combinations(extra, message):
    with pytest.raises(ValueError, match=message):
        ft.Kandinsky6Trainer()._validate_full_finetune_args(parse(*extra))


def test_non_fused_optimizer_is_allowed_without_block_swap():
    ft.Kandinsky6Trainer()._validate_full_finetune_args(parse("--no-fused_backward_pass", "--optimizer_type", "AdamW8bit"))


@pytest.mark.parametrize("ring", [False, True])
def test_fused_block_swap_is_accepted(ring):
    extra = ["--blocks_to_swap", "1"]
    if ring:
        extra += ["--block_swap_trainable_ring", "--use_pinned_memory_for_block_swap"]
    ft.Kandinsky6Trainer()._validate_full_finetune_args(parse(*extra))


def test_prequantized_checkpoint_detected_by_parent_is_rejected(monkeypatch):
    trainer = ft.Kandinsky6Trainer()

    def fake_parent(self, args):
        args.convrot_int8 = True  # what handle_model_specific_args does for a prequantized file
        return True

    monkeypatch.setattr(ft.Kandinsky6NetworkTrainer, "_validate_args_and_init", fake_parent)
    with pytest.raises(ValueError, match="pre-quantized"):
        trainer._validate_args_and_init(parse())


@pytest.mark.parametrize("slots, expected", [("0", None), ("4", 4)])
def test_r2av_reference_slots_zero_means_the_default(monkeypatch, slots, expected):
    """The GUI's R2AV Reference Slots writes 0 for "8, or the checkpoint's own count": that is the unset value."""
    from musubi_tuner import kandinsky6_train_network as network

    monkeypatch.setattr(network, "has_comfy_quant_tensors", lambda *args, **kwargs: False)
    args = parse("--task", "r2av", "--max_reference_slots", slots, "--mixed_precision", "bf16", "--timestep_sampling", "uniform", "--discrete_flow_shift", "1.0")
    ft.Kandinsky6Trainer().handle_model_specific_args(args)
    assert args.max_reference_slots == expected
    with pytest.raises(ValueError, match="must be positive"):
        ft.Kandinsky6Trainer().handle_model_specific_args(
            parse("--task", "r2av", "--max_reference_slots", "-1", "--mixed_precision", "bf16", "--timestep_sampling", "uniform", "--discrete_flow_shift", "1.0")
        )


def test_freeze_uses_kandinsky6_visual_block_names():
    model = tiny_dit(num_visual_blocks=3)
    model.requires_grad_(True)
    frozen, elements = ft.freeze_transformer_parameters(model, "0,2", ["audio_text_transformer_blocks"])
    assert frozen > 0 and elements > 0
    for name, parameter in model.named_parameters():
        expect_frozen = name.startswith(
            ("visual_transformer_blocks.0.", "visual_transformer_blocks.2.", "audio_text_transformer_blocks")
        )
        assert parameter.requires_grad is not expect_frozen, name


def test_freeze_rejects_out_of_range_blocks_and_dead_patterns():
    model = tiny_dit(num_visual_blocks=2)
    with pytest.raises(ValueError, match="has 2"):
        ft.freeze_transformer_parameters(model, "2", None)
    with pytest.raises(ValueError, match="match no"):
        ft.freeze_transformer_parameters(model, None, ["no_such_parameter"])


@pytest.mark.parametrize("mem_eff_save", [True, False])
def test_dense_checkpoint_round_trips_into_a_fresh_model(tmp_path, mem_eff_save):
    model = tiny_dit().to(torch.bfloat16)
    module = ft.Kandinsky6FullFinetuneModule(model, mem_eff_save=mem_eff_save)
    path = tmp_path / "dense.safetensors"
    module.save_weights(str(path), torch.bfloat16, {"ss_training_type": "full_finetune"})
    state = load_file(str(path))
    assert set(state) == set(model.state_dict())
    fresh = tiny_dit().to(torch.bfloat16)
    fresh.load_state_dict(state, strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value), key
    with pytest.raises(ValueError, match="BF16"):
        module.save_weights(str(path), torch.float32, None)


def test_ring_checkpoint_loader_rejects_broadcastable_wrong_shape(tmp_path):
    model = torch.nn.Linear(4, 3, bias=True)
    state = model.state_dict()
    checkpoint = tmp_path / "wrong_shape.safetensors"
    ft.mem_eff_save_file(
        {"weight": state["weight"][:1].clone(), "bias": state["bias"].clone()},
        str(checkpoint),
    )
    before = state["weight"].clone()
    with pytest.raises(RuntimeError, match=r"'weight'.*shape \(1, 4\).*expected \(3, 4\)"):
        ft.load_ring_checkpoint_state(model, checkpoint)
    assert torch.equal(model.weight, before)


def test_dense_resume_cursor_round_trip_and_signature_guard():
    from musubi_tuner.training.trainer_base import DenseResumeCursor

    cursor = DenseResumeCursor(
        global_step=3,
        epoch=1,
        step_in_epoch=4,
        optimizer_signature="abc",
        gradient_accumulation_steps=2,
        data_seed=123,
        num_batches_per_epoch=7,
    )
    restored = DenseResumeCursor(optimizer_signature="abc", gradient_accumulation_steps=2, data_seed=123)
    restored.load_state_dict(cursor.state_dict())
    assert (restored.global_step, restored.epoch, restored.step_in_epoch) == (3, 1, 4)
    with pytest.raises(RuntimeError, match="optimizer parameter mapping"):
        DenseResumeCursor(optimizer_signature="different", gradient_accumulation_steps=2, data_seed=123).load_state_dict(
            cursor.state_dict()
        )
    with pytest.raises(RuntimeError, match="gradient_accumulation_steps"):
        DenseResumeCursor(optimizer_signature="abc", gradient_accumulation_steps=1, data_seed=123).load_state_dict(
            cursor.state_dict()
        )
    with pytest.raises(RuntimeError, match="data seed"):
        DenseResumeCursor(optimizer_signature="abc", gradient_accumulation_steps=2, data_seed=456).load_state_dict(
            cursor.state_dict()
        )


@pytest.mark.parametrize("interrupt_after_batches", [2, 4])
def test_dense_resume_epoch_sampler_and_noise_are_continuous_at_ga2(interrupt_after_batches):
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from musubi_tuner.training.trainer_base import seed_dense_dataloader_for_epoch

    items = list(range(4))
    seed = 314
    accelerator = Accelerator()

    def make_iterator(epoch, skip, training_rng):
        loader = DataLoader(items, batch_size=1, shuffle=True)
        loader = accelerator.prepare(loader, device_placement=[False])
        if skip:
            loader = accelerator.skip_first_batches(loader, skip)
        seed_dense_dataloader_for_epoch(loader, seed + epoch * 1009)
        iterator = iter(loader)
        torch.set_rng_state(training_rng)
        return iterator

    torch.manual_seed(991)
    uninterrupted = []
    for epoch in range(2):
        iterator = make_iterator(epoch, 0, torch.get_rng_state())
        uninterrupted.extend((int(batch.item()), float(torch.randn(()))) for batch in iterator)

    torch.manual_seed(991)
    prefix = []
    cursor_epoch = 0
    cursor_step = 0
    for epoch in range(2):
        iterator = make_iterator(epoch, 0, torch.get_rng_state())
        for step, batch in enumerate(iterator, start=1):
            prefix.append((int(batch.item()), float(torch.randn(()))))
            cursor_epoch, cursor_step = epoch, step
            if len(prefix) == interrupt_after_batches:
                break
        if len(prefix) == interrupt_after_batches:
            break
    checkpoint_rng = torch.get_rng_state()
    if cursor_step == len(items):
        cursor_epoch += 1
        cursor_step = 0

    resumed = list(prefix)
    for epoch in range(cursor_epoch, 2):
        skip = cursor_step if epoch == cursor_epoch else 0
        iterator = make_iterator(epoch, skip, checkpoint_rng if epoch == cursor_epoch else torch.get_rng_state())
        resumed.extend((int(batch.item()), float(torch.randn(()))) for batch in iterator)
        cursor_step = 0
    assert resumed == uninterrupted


def test_fused_adafactor_updates_dense_weights_during_backward():
    trainer = ft.Kandinsky6Trainer()
    trainer._scheduler_scale = 1.0
    model = tiny_dit(num_visual_blocks=1)
    torch.manual_seed(0)
    with torch.no_grad():
        # Fresh K6 blocks are zero-gated (adaLN-zero style), which would leave most
        # tensors with exactly zero gradient; randomize so every path is exercised.
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
    args = parse()
    args.gradient_checkpointing = False
    network = trainer._build_network(args, None, model, None, None)
    params, _ = network.prepare_optimizer_params(unet_lr=1e-2)
    _, _, optimizer, _, _ = trainer.get_optimizer(args, params)
    accelerator = SimpleNamespace(sync_gradients=True, num_processes=1, device=torch.device("cpu"), autocast=nullcontext)
    trainer._install_fused_optimizer(args, accelerator, optimizer)

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    batch = {
        "latents_audio": torch.randn(1, 40, 6),
        "audio_present": torch.ones(1),
        "text_embeds": [torch.randn(2, 8)],
        "attention_mask": [torch.ones(2, dtype=torch.bool)],
        "pooled_embed": torch.randn(1, 6),
        "timesteps": torch.tensor([0.5]),
    }
    run_args = SimpleNamespace(video_only=False, audio_loss_weight=1.0, independent_time=False, gradient_checkpointing=False)
    trainer._task = "t2av"
    loss, _ = trainer.process_batch(
        run_args,
        accelerator,
        model,
        network,
        batch,
        torch.randn(1, 16, 2, 2, 2),
        torch.randn(1, 16, 2, 2, 2),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    loss.backward()

    # Every gradient was consumed by its hook and the weights moved.
    assert all(parameter.grad is None for parameter in model.parameters())
    changed = [name for name, parameter in model.named_parameters() if not torch.equal(parameter, before[name])]
    assert len(changed) > len(before) // 2
    # The wrapper's own step is a no-op afterwards (no gradients left).
    snapshot = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.step()
    assert all(torch.equal(parameter, snapshot[name]) for name, parameter in model.named_parameters())


@pytest.mark.parametrize("fused", [False, True])
def test_adafactor_resume_preserves_fp32_moments_exactly(fused):
    trainer = ft.Kandinsky6Trainer()
    args = parse() if fused else parse("--no-fused_backward_pass")
    parameter = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.bfloat16))
    _, _, optimizer, _, _ = trainer.get_optimizer(args, [{"params": [parameter], "lr": 1e-2}])
    accelerator = SimpleNamespace(sync_gradients=True, num_processes=1)
    trainer._install_fused_optimizer(args, accelerator, optimizer)

    saved = optimizer.state_dict()
    parameter_id = saved["param_groups"][0]["params"][0]
    row = torch.tensor([1.00001, 1.00002, 1.00003, 1.00004, 1.00005], dtype=torch.float32)
    col = torch.linspace(0.123456, 0.123462, 7, dtype=torch.float32)
    saved["state"][parameter_id] = {"step": 9, "RMS": 0, "exp_avg_sq_row": row, "exp_avg_sq_col": col}
    optimizer.load_state_dict(saved)

    state = optimizer.state[parameter]
    assert state["exp_avg_sq_row"].dtype == torch.float32
    assert state["exp_avg_sq_col"].dtype == torch.float32
    assert torch.equal(state["exp_avg_sq_row"], row)
    assert torch.equal(state["exp_avg_sq_col"], col)


def test_triton_adafactor_falls_back_where_its_kernels_cannot_run():
    """--adafactor_triton where Triton cannot build or launch its kernels (here the CPU) trains with the standard fused
    update instead of stopping at the first step's backward pass."""
    from musubi_tuner.modules.adafactor_fused import adafactor_step_param

    trainer = ft.Kandinsky6Trainer()
    args = parse()
    args.adafactor_triton = True
    parameter = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.bfloat16))
    _, _, optimizer, _, _ = trainer.get_optimizer(args, [{"params": [parameter], "lr": 1e-2}])
    accelerator = SimpleNamespace(sync_gradients=True, num_processes=1, device=torch.device("cpu"))
    trainer._install_fused_optimizer(args, accelerator, optimizer)
    assert optimizer.step_param.__func__ is adafactor_step_param

    before = parameter.detach().clone()
    (parameter.float() ** 2).sum().backward()
    assert parameter.grad is None and not torch.equal(parameter, before)


def test_trainable_ring_configuration_requires_training_prerequisites():
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

    base = dict(
        block_swap_h2d_only=False,
        block_swap_ring_size=2,
        block_swap_trainable_ring=True,
        use_pinned_memory_for_block_swap=True,
        gradient_checkpointing=True,
        fused_backward_pass=True,
    )
    config = BlockSwapConfig.from_args(SimpleNamespace(**base), torch.device("cuda"), supports_backward=True)
    assert config.trainable_ring
    for key, message in (
        ("use_pinned_memory_for_block_swap", "pinned"),
        ("gradient_checkpointing", "gradient_checkpointing"),
        ("fused_backward_pass", "fused_backward_pass"),
    ):
        invalid = dict(base)
        invalid[key] = False
        with pytest.raises(ValueError, match=message):
            BlockSwapConfig.from_args(SimpleNamespace(**invalid), torch.device("cuda"), supports_backward=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_trainable_ring_matches_checkpointed_dense_forward_and_gradients():
    from musubi_tuner.modules.custom_offloading_utils import TrainableBlockRingOffloader

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(16, 24), torch.nn.SiLU(), torch.nn.Linear(24, 16)) for _ in range(5)]
            )

        def forward(self, value, offloader=None):
            for index, block in enumerate(self.blocks):
                if offloader is not None:
                    offloader.wait_for_block(index)

                def run(hidden, block=block, index=index):
                    if offloader is not None and offloader.recompute_requires_wait:
                        offloader.wait_for_block(index)
                    return block(hidden)

                value = checkpoint(run, value, use_reentrant=False)
                if offloader is not None:
                    offloader.submit_move_blocks_forward(self.blocks, index)
            return value

    import copy

    torch.manual_seed(123)
    reference = Toy().to(device="cuda", dtype=torch.bfloat16)
    streamed = copy.deepcopy(reference).cpu()
    offloader = TrainableBlockRingOffloader(
        "toy", streamed.blocks, 5, 3, True, torch.device("cuda"), ring_size=2, use_pinned_memory=True
    )
    offloader.prepare_block_devices_before_forward(streamed.blocks)
    ref_input = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ring_input = ref_input.detach().clone().requires_grad_(True)
    ref_output = reference(ref_input)
    ring_output = streamed(ring_input, offloader)
    ref_output.float().square().sum().backward()
    ring_output.float().square().sum().backward()
    torch.testing.assert_close(ring_output, ref_output)
    torch.testing.assert_close(ring_input.grad, ref_input.grad)
    for ref_parameter, ring_parameter in zip(reference.parameters(), streamed.parameters()):
        torch.testing.assert_close(ring_parameter.grad, ref_parameter.grad)
    offloader.offload_to_cpu(streamed.blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ring_resume_parks_resident_slots_before_model_load(tmp_path):
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

    model = tiny_dit(num_visual_blocks=3).to(torch.bfloat16)
    model.enable_gradient_checkpointing()
    model.enable_block_swap(
        1,
        BlockSwapConfig(
            device=torch.device("cuda"),
            supports_backward=True,
            use_pinned_memory=True,
            trainable_ring=True,
        ),
    )
    model.move_to_device_except_swap_blocks(torch.device("cuda"))
    model.prepare_block_swap_before_forward()
    model.switch_block_swap_for_training()
    network = ft.Kandinsky6FullFinetuneModule(model)
    restored = {
        key: value.detach().cpu().clone().add_(0.25) if value.is_floating_point() else value.detach().cpu().clone()
        for key, value in network.state_dict().items()
    }

    class FakeAccelerator:
        device = torch.device("cuda")
        is_main_process = True

        def __init__(self):
            self.save_hooks = []
            self.load_hooks = []

        def unwrap_model(self, value):
            return value

        def register_save_state_pre_hook(self, hook):
            self.save_hooks.append(hook)

        def register_load_state_pre_hook(self, hook):
            self.load_hooks.append(hook)

        def register_for_checkpointing(self, obj):
            self.checkpoint_object = obj

        def get_state_dict(self, value, unwrap=False):
            assert not unwrap
            return value.state_dict()

        def load_state(self, path):
            models = [network]
            for hook in self.load_hooks:
                hook(models, path)

    accelerator = FakeAccelerator()
    ft.mem_eff_save_file(restored, str(tmp_path / "model.safetensors"))
    torch.save({"last_epoch": 1}, tmp_path / "scheduler.bin")
    args = SimpleNamespace(
        block_swap_trainable_ring=True,
        resume=str(tmp_path),
        resume_from_huggingface=False,
        gradient_accumulation_steps=1,
        seed=42,
    )
    ft.Kandinsky6Trainer()._register_hooks_and_resume(args, accelerator, network)
    assert accelerator.save_hooks  # ring drain plus the base filtering hook
    for index in range(len(model.visual_transformer_blocks)):
        model.offloader.wait_for_block(index)
        model.offloader.submit_move_blocks_forward(model.visual_transformer_blocks, index)
    # Accelerate gathers these tensor views before invoking save prehooks. The
    # tail blocks currently own ring slots, so commit will reuse those buffers;
    # the hook must replace the dictionaries after committing.
    stale_weights = [network.state_dict()]
    models = [network]
    for hook in accelerator.save_hooks:
        hook(models, stale_weights, str(tmp_path))
    assert stale_weights == []
    saved_model = load_file(str(tmp_path / "model.safetensors"))
    assert set(saved_model) == set(restored)
    for key, expected in restored.items():
        torch.testing.assert_close(saved_model[key], expected, msg=lambda message, key=key: f"save {key}: {message}")
    model.offloader.commit()
    actual = network.state_dict()
    for key, expected in restored.items():
        torch.testing.assert_close(actual[key].cpu(), expected, msg=lambda message, key=key: f"{key}: {message}")
    model.close_block_swap()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ring_fused_adafactor_accumulates_then_commits_updates():
    from musubi_tuner.modules.custom_offloading_utils import TrainableBlockRingOffloader

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(4)])

        def forward(self, value, offloader):
            for index, block in enumerate(self.blocks):
                offloader.wait_for_block(index)

                def run(hidden, block=block, index=index):
                    offloader.wait_for_block(index)
                    return torch.nn.functional.silu(block(hidden))

                value = checkpoint(run, value, use_reentrant=False)
                offloader.submit_move_blocks_forward(self.blocks, index)
            return value

    torch.manual_seed(456)
    model = Toy().to(dtype=torch.bfloat16)
    offloader = TrainableBlockRingOffloader(
        "toy", model.blocks, 4, 2, True, torch.device("cuda"), ring_size=1, use_pinned_memory=True
    )
    offloader.prepare_block_devices_before_forward(model.blocks)
    before = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    args = parse()
    optimizer = Adafactor(model.parameters(), lr=1e-2, relative_step=False, scale_parameter=False, warmup_init=False)
    accelerator = SimpleNamespace(sync_gradients=False, num_processes=1)
    ft.Kandinsky6Trainer()._install_fused_optimizer(args, accelerator, optimizer)

    model(torch.randn(2, 16, device="cuda", dtype=torch.bfloat16), offloader).float().square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    offloader.commit()
    assert all(torch.equal(parameter.detach().cpu(), old) for parameter, old in zip(model.parameters(), before))

    accelerator.sync_gradients = True
    model(torch.randn(2, 16, device="cuda", dtype=torch.bfloat16), offloader).float().square().mean().backward()
    offloader.commit()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert any(not torch.equal(parameter.detach().cpu(), old) for parameter, old in zip(model.parameters(), before))
    offloader.offload_to_cpu(model.blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_accelerate_save_state_keeps_every_packed_ring_key(tmp_path):
    from accelerate import Accelerator
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

    accelerator = Accelerator()
    model = tiny_dit(num_visual_blocks=3).to(torch.bfloat16)
    model.enable_gradient_checkpointing()
    model.enable_block_swap(
        1,
        BlockSwapConfig(
            device=accelerator.device,
            supports_backward=True,
            use_pinned_memory=True,
            trainable_ring=True,
        ),
    )
    network = ft.Kandinsky6FullFinetuneModule(model)
    network = accelerator.prepare(network, device_placement=[False])
    unwrapped = accelerator.unwrap_model(network)
    unwrapped.transformer.move_to_device_except_swap_blocks(accelerator.device)
    unwrapped.transformer.prepare_block_swap_before_forward()
    unwrapped.transformer.switch_block_swap_for_training()
    args = SimpleNamespace(
        block_swap_trainable_ring=True,
        resume=None,
        resume_from_huggingface=False,
        gradient_accumulation_steps=1,
        seed=42,
    )
    ft.Kandinsky6Trainer()._register_hooks_and_resume(args, accelerator, network)

    blocks = unwrapped.transformer.visual_transformer_blocks
    for index in range(len(blocks)):
        unwrapped.transformer.offloader.wait_for_block(index)
        unwrapped.transformer.offloader.submit_move_blocks_forward(blocks, index)
    expected = {key: value.detach().cpu().clone() for key, value in unwrapped.state_dict().items()}
    accelerator.save_state(str(tmp_path))

    path = tmp_path / "model.safetensors"
    assert path.exists()
    saved = load_file(str(path))
    assert set(saved) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(saved[key], value, msg=lambda message, key=key: f"{key}: {message}")
    assert (tmp_path / "random_states_0.pkl").exists()

    with torch.no_grad():
        for parameter in unwrapped.parameters():
            parameter.add_(1)
    accelerator.load_state(str(tmp_path))
    unwrapped.transformer.move_to_device_except_swap_blocks(accelerator.device)
    unwrapped.transformer.prepare_block_swap_before_forward()
    unwrapped.transformer.switch_block_swap_for_training()
    for index in range(len(blocks)):
        unwrapped.transformer.offloader.wait_for_block(index)
        unwrapped.transformer.offloader.submit_move_blocks_forward(blocks, index)
    unwrapped.transformer.offloader.commit()
    reloaded = unwrapped.state_dict()
    for key, value in expected.items():
        torch.testing.assert_close(reloaded[key].cpu(), value, msg=lambda message, key=key: f"reload {key}: {message}")
    unwrapped.transformer.close_block_swap()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_adafactor_updates_bf16_matrix_with_fp32_moments():
    pytest.importorskip("triton")
    from musubi_tuner.modules.adafactor_triton import patch_adafactor_triton

    torch.manual_seed(321)
    parameter = torch.nn.Parameter(torch.randn(64, 128, device="cuda", dtype=torch.bfloat16))
    before = parameter.detach().clone()
    parameter.grad = torch.randn_like(parameter)
    optimizer = Adafactor([parameter], lr=1e-2, relative_step=False, scale_parameter=False, warmup_init=False)
    patch_adafactor_triton(optimizer)
    optimizer.step_param(parameter, optimizer.param_groups[0])
    state = optimizer.state[parameter]
    assert state["exp_avg_sq_row"].dtype == torch.float32
    assert state["exp_avg_sq_col"].dtype == torch.float32
    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_matrix_step_matches_reference_adafactor_numerically():
    pytest.importorskip("triton")
    from musubi_tuner.modules.adafactor_triton import _triton_matrix_step

    torch.manual_seed(90210)
    initial = torch.randn(33, 129, device="cuda", dtype=torch.float32) * 0.4
    actual = initial.clone()
    expected = torch.nn.Parameter(initial.clone())
    optimizer = Adafactor(
        [expected],
        lr=3e-3,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        clip_threshold=0.35,
        weight_decay=0.07,
    )
    row_state = torch.zeros(33, device="cuda", dtype=torch.float32)
    col_state = torch.zeros(129, device="cuda", dtype=torch.float32)
    gradients = [
        torch.randn_like(actual) * 2.5,
        torch.linspace(-3.0, 4.0, actual.numel(), device="cuda").reshape_as(actual),
        torch.zeros_like(actual),
        torch.randn_like(actual) * 0.1,
    ]
    for step, gradient in enumerate(gradients, start=1):
        expected.grad = gradient.clone()
        optimizer.step()
        group = optimizer.param_groups[0]
        _triton_matrix_step(
            actual,
            gradient,
            row_state,
            col_state,
            beta2=1.0 - step ** group["decay_rate"],
            eps=float(group["eps"][0]),
            clip_threshold=float(group["clip_threshold"]),
            learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            seed=step,
        )
        state = optimizer.state[expected]
        torch.testing.assert_close(row_state, state["exp_avg_sq_row"], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(col_state, state["exp_avg_sq_col"], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape,beta1", [((257,), None), ((17, 19), 0.9)])
def test_triton_public_step_falls_back_for_unsupported_bf16_parameters(shape, beta1):
    pytest.importorskip("triton")
    from musubi_tuner.modules.adafactor_triton import patch_adafactor_triton

    torch.manual_seed(73)
    initial = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    gradient = torch.randn_like(initial)
    actual = torch.nn.Parameter(initial.clone())
    expected = torch.nn.Parameter(initial.clone())
    kwargs = dict(lr=2e-3, relative_step=False, scale_parameter=False, warmup_init=False)
    if beta1 is not None:
        kwargs["beta1"] = beta1
    actual_optimizer = Adafactor([actual], **kwargs)
    expected_optimizer = Adafactor([expected], **kwargs)
    patch_adafactor_triton(actual_optimizer)
    actual.grad = gradient.clone()
    expected.grad = gradient.clone()
    torch.manual_seed(111)
    actual_optimizer.step_param(actual, actual_optimizer.param_groups[0])
    torch.manual_seed(111)
    from musubi_tuner.modules.adafactor_fused import adafactor_step_param

    adafactor_step_param(expected_optimizer, expected, expected_optimizer.param_groups[0])
    assert torch.equal(actual, expected)
    for key, value in expected_optimizer.state[expected].items():
        actual_value = actual_optimizer.state[actual][key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual_value, value)
        else:
            assert actual_value == value

    untouched = torch.nn.Parameter(initial.clone())
    no_grad_optimizer = Adafactor([untouched], **kwargs)
    patch_adafactor_triton(no_grad_optimizer)
    before = untouched.detach().clone()
    no_grad_optimizer.step_param(untouched, no_grad_optimizer.param_groups[0])
    assert torch.equal(untouched, before)
    assert no_grad_optimizer.state[untouched] == {}
