"""training/replica_trainer.py: single-process multi-GPU LoRA training (kohya-ss/sd-scripts dev 88f1c38, ported for the
MOSS-Image LoRA trainer). The pure helpers as kohya tests them; the dispatch and the gradient sum on the CPU with a tiny
DiT and a replica on the master's device (the GPU copies and the real loop run in the server smoke test)."""

import argparse
import contextlib
import contextvars
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch.nn as nn

from musubi_tuner.moss_image.moss_model import MossImage, MossImageParams
from musubi_tuner.training import replica_trainer
from musubi_tuner.training.replica_trainer import (
    AcceleratorDeviceProxy,
    PinnedStager,
    Replica,
    ReplicaTrainerMixin,
    ShapeTracker,
    build_schedule,
    input_shape_signature,
    parse_replica_devices,
)

PARAMS = dict(dim=64, ffn_dim=96, num_heads=2, num_layers=4, text_dim=48, freq_dim=16, latent_channels=8)


# ------------------------------------------------------------------------------------------------------- pure helpers
def test_parse_replica_devices():
    assert parse_replica_devices("1") == [torch.device("cuda:1")]
    assert parse_replica_devices("cuda:1, 2") == [torch.device("cuda:1"), torch.device("cuda:2")]
    for bad in ("cpu", "cuda", " , ", "1,1"):
        with pytest.raises(ValueError):
            parse_replica_devices(bad)


def test_build_schedule_replicas_first_master_last():
    assert build_schedule(1) == [0]
    assert build_schedule(2) == [1, 0]
    assert build_schedule(3) == [1, 2, 0]


def test_accelerator_proxy_delegates_everything_but_device_and_prepare():
    real = types.SimpleNamespace(device=torch.device("cuda:9"), num_processes=1, print=lambda *a: "printed", unwrap_model=lambda m: "unwrapped")
    proxy = AcceleratorDeviceProxy(real, torch.device("cpu"))
    assert proxy.device == torch.device("cpu") and proxy.num_processes == 1 and proxy.print() == "printed"
    module = nn.Linear(2, 2)
    assert proxy.prepare(module) is module and proxy.prepare(module, device_placement=[False]) is module
    assert proxy.prepare(module, module) == (module, module) and proxy.unwrap_model(module) is module


def test_wrap_trainer_class():
    class Base:
        pass

    assert replica_trainer.wrap_trainer_class(Base, argparse.Namespace(replica_devices=None)) is Base
    assert replica_trainer.wrap_trainer_class(Base, argparse.Namespace(replica_devices="")) is Base
    wrapped = replica_trainer.wrap_trainer_class(Base, argparse.Namespace(replica_devices="1"))
    assert wrapped.__mro__[1:3] == (ReplicaTrainerMixin, Base)


def test_move_to_device_recurses_and_keeps_non_tensors():
    data = {"a": torch.zeros(2), "b": [torch.ones(1), "text"], "c": (torch.ones(1),), "d": None}
    moved = replica_trainer.move_to_device(data, torch.device("cpu"))
    assert moved["b"][1] == "text" and moved["d"] is None and isinstance(moved["c"], tuple)


def test_shape_tracker_follows_automatic_dynamic_shapes():
    auto = ShapeTracker("auto")
    assert auto.is_new(0, (1, 64, 64))  # first shape
    assert not auto.is_new(0, (1, 64, 64))
    assert auto.is_new(0, (1, 64, 80))  # second value of a dimension: recompiled as dynamic
    assert not auto.is_new(0, (1, 64, 96))  # dynamic now
    assert auto.is_new(1, (1, 64, 64))  # per device
    static = ShapeTracker("false")
    static.is_new(0, (1, 64))
    static.is_new(0, (1, 80))
    assert static.is_new(0, (1, 96))
    dynamic = ShapeTracker("true")
    dynamic.is_new(0, (1, 64))
    assert not dynamic.is_new(0, (1, 96))


def test_input_shape_signature():
    inputs = {"hidden": torch.zeros(1, 8, 4, 6), "contexts": [torch.zeros(7, 48)], "ref_latents": [[torch.zeros(8, 2, 3)]]}
    assert input_shape_signature(inputs) == (1, 8, 4, 6, 7, 8, 2, 3)


def test_pinned_stager_same_device_copy_and_accumulate():
    stager = PinnedStager()
    src = [torch.ones(2, 3), torch.full((4,), 2.0)]
    dst = [torch.zeros(2, 3), torch.zeros(4)]
    stager.copy_flat(src, dst, key="w")
    assert torch.equal(dst[0], src[0]) and torch.equal(dst[1], src[1])
    stager.copy_flat(src, dst, key="g", accumulate=True)
    assert torch.equal(dst[1], torch.full((4,), 4.0))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs")
def test_pinned_stager_cross_device_copy():
    stager = PinnedStager()
    src = [torch.randn(3, 5, device="cuda:1"), torch.randn(7, device="cuda:1", dtype=torch.bfloat16)]
    dst = [torch.zeros(3, 5, device="cuda:0"), torch.zeros(7, device="cuda:0", dtype=torch.bfloat16)]
    stager.copy_flat(src, dst, key="w")
    stager.copy_flat(src, dst, key="w", accumulate=True)
    torch.cuda.synchronize()
    assert torch.equal(dst[0].cpu(), 2 * src[0].cpu()) and torch.equal(dst[1].cpu(), 2 * src[1].cpu())


def test_replica_submit_runs_in_the_callers_context():
    """kohya #2481: torch._dynamo / torch._inductor config overrides live in contextvars, which a worker thread does not
    inherit; each micro-step runs in a copy of the caller's context."""
    flag = contextvars.ContextVar("flag", default="worker default")
    replica = Replica(index=1, device=torch.device("cpu"), proxy=None, transformer=None, network=None, executor=ThreadPoolExecutor(max_workers=1))
    token = flag.set("main thread value")
    try:
        assert replica.submit(flag.get).result() == "main thread value"
    finally:
        flag.reset(token)
    replica.executor.shutdown()


# --------------------------------------------------------------------------------------------------------- the trainer
def _args(**overrides):
    values = dict(
        replica_devices="1", replica_warmup_steps=None, compile=False, compile_dynamic=None, mixed_precision="bf16", fp8_base=False,
        fp8_scaled=False, convrot_int8=False, base_weights=None, gradient_accumulation_steps=2, sample_prompts=None, text_encoder=None,
        network_module="networks.lora_moss_image", network_args=None, weighting_scheme="none", loss_type="mse", huber_delta=1.0,
        gradient_checkpointing=False, max_train_steps=10, network_weights=None, dim_from_weights=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _replica_trainer(monkeypatch, **overrides):
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer = replica_trainer.wrap_trainer_class(MossImageNetworkTrainer, _args(**overrides))()
    args = _args(**overrides)
    trainer.handle_model_specific_args(args)
    return trainer, args


def test_the_accumulation_is_multiplied_by_the_device_count(monkeypatch):
    trainer, args = _replica_trainer(monkeypatch, replica_devices="1,2", gradient_accumulation_steps=2)
    assert args.gradient_accumulation_steps == 6 and trainer._replica_orig_gradient_accumulation_steps == 2
    assert trainer._replica_schedule == [1, 2, 0] and trainer._replica_warmup_steps == 0
    assert trainer.get_checkpoint_metadata(args) == {"ss_gradient_accumulation_steps": 2, "ss_replica_devices": "cuda:1,cuda:2"}
    compiled, _ = _replica_trainer(monkeypatch, compile=True)
    assert compiled._replica_warmup_steps == 2 and compiled._replica_shapes is not None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mixed_precision": "fp16"}, "fp16"),
        ({"fp8_base": True}, "BF16 DiT"),
        ({"convrot_int8": True}, "convrot_int8"),
        ({"base_weights": ["a.safetensors"]}, "base_weights"),
        ({"weighting_scheme": "sigma_sqrt"}, "weighting_scheme"),
        ({"control_adapter": True}, "control_adapter"),
    ],
)
def test_unsupported_settings_are_refused(monkeypatch, overrides, message):
    with pytest.raises(ValueError, match=message):
        _replica_trainer(monkeypatch, **overrides)


class _Accelerator:
    """What the dispatch needs of accelerate: the device, the accumulation's sync flag and backward's loss division."""

    def __init__(self, accumulation: int):
        self.device = torch.device("cpu")
        self.num_processes = 1
        self.sync_gradients = False
        self.gradient_accumulation_steps = accumulation

    def backward(self, loss):
        (loss / self.gradient_accumulation_steps).backward()

    def autocast(self):
        return contextlib.nullcontext()

    def unwrap_model(self, model):
        return model


def _model_and_network(lora_seed=0):
    """The same frozen DiT every time (both copies load one checkpoint), the LoRA initialized from ``lora_seed``."""
    from musubi_tuner.networks import lora_moss_image

    torch.manual_seed(0)
    model = MossImage(MossImageParams(**PARAMS))
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.15)
    model.requires_grad_(False)
    network = lora_moss_image.create_arch_network(1.0, 4, 4, None, [], model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    torch.manual_seed(lora_seed)
    with torch.no_grad():
        for p in network.parameters():
            p.normal_(0.0, 0.05)  # lora_up starts at zero: give every LoRA matrix a gradient
    network.prepare_grad_etc(model)
    return model, network


def _micro_steps(count=4):
    torch.manual_seed(7)
    steps = []
    for _ in range(count):
        latents = torch.randn(1, 8, 3, 4)
        noise = torch.randn_like(latents)
        steps.append({"latents": latents, "noise": noise, "noisy": 0.6 * latents + 0.4 * noise, "t": torch.tensor([400.0]),
                      "batch": {"ctx_vec": [torch.randn(5, 48)]}})
    return steps


def _loop_loss(output):
    """The loop's plain loss of a DiT output (mse, mean)."""
    pred = output.pred.to(device=output.target.device, dtype=torch.float32)
    return torch.nn.functional.mse_loss(pred, output.target)


def test_a_replica_step_sums_the_gradients_of_single_gpu_accumulation(monkeypatch):
    """Master + one replica with accumulation 1 (2 micro-steps per optimizer step) give the master the gradients a
    single-GPU run with accumulation 2 gets from the same micro-steps; the loop's loss on the replica's stand-in is
    the replica's loss and adds nothing to the gradients; the replica pulls the master's weights first."""
    steps = _micro_steps(2)
    args = _args(gradient_accumulation_steps=1)

    # single GPU, accumulation 2
    from musubi_tuner.moss_image_train_network import MossImageNetworkTrainer

    single = MossImageNetworkTrainer()
    single.handle_model_specific_args(_args(gradient_accumulation_steps=2, replica_devices=None))
    model, network = _model_and_network()
    accelerator = _Accelerator(2)
    single_losses = []
    for step in steps:
        out = single.call_dit(args, accelerator, model, step["latents"], step["batch"], step["noise"], step["noisy"], step["t"], torch.float32)
        loss = _loop_loss(out)
        single_losses.append(float(loss))
        accelerator.backward(loss)
    expected = [p.grad.clone() for p in network.parameters() if p.requires_grad]

    # master + replica (the replica on the CPU too, its own copy with other initial weights: it must pull the master's)
    trainer, replica_args = _replica_trainer(monkeypatch, gradient_accumulation_steps=1)
    assert replica_args.gradient_accumulation_steps == 2
    master_model, master_network = _model_and_network()
    replica_model, replica_network = _model_and_network(lora_seed=123)
    replica = Replica(index=1, device=torch.device("cpu"), proxy=None, transformer=replica_model, network=replica_network,
                      executor=ThreadPoolExecutor(max_workers=1))
    trainer._replicas = [replica]
    accelerator = _Accelerator(2)
    trainer._ensure_param_lists(accelerator, master_network)
    losses = []
    for index, step in enumerate(steps):
        accelerator.sync_gradients = index == len(steps) - 1
        out = trainer.call_dit(replica_args, accelerator, master_model, step["latents"], step["batch"], step["noise"], step["noisy"], step["t"], torch.float32)
        loss = _loop_loss(out)
        losses.append(float(loss))
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            trainer.before_gradient_sync(accelerator, master_network)
    got = [p.grad for p in master_network.parameters() if p.requires_grad]
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    # the first replica micro-step waits for its own loss; the master's is its own
    assert losses == pytest.approx(single_losses, rel=1e-5, abs=1e-6)
    assert all(p.grad is None for p in replica.params) and replica.weights_dirty
    replica.executor.shutdown()


def test_the_windows_last_micro_step_runs_on_the_master(monkeypatch):
    trainer, args = _replica_trainer(monkeypatch, replica_devices="1,2", gradient_accumulation_steps=1)
    calls = []
    trainer.prepare_dit_inputs = lambda *a, **k: {"hidden": torch.zeros(1, 8, 2, 2), "contexts": [torch.zeros(3, 48)]}
    trainer.run_dit = lambda args, transformer, device, dtype, inputs: (calls.append("master") or torch.zeros(1), torch.zeros(1))

    class _Fake:
        def __init__(self, index):
            self.index, self.last_future, self.weights_dirty, self.device = index, None, False, torch.device("cpu")

        def submit(self, fn):
            calls.append(f"replica{self.index}")
            future = types.SimpleNamespace(result=lambda: 0.5)
            self.last_future = future
            return future

        def join(self):
            pass

    trainer._replicas = [_Fake(1), _Fake(2)]
    trainer._replica_anchor = nn.Parameter(torch.zeros(1))
    accelerator = _Accelerator(3)
    pattern = []
    # a full window of 3, then an epoch's last, partial window of 2 (sync at its end)
    for sync in (False, False, True, False, True):
        accelerator.sync_gradients = sync
        calls.clear()
        trainer.call_dit(args, accelerator, None, torch.zeros(1, 8, 2, 2), {}, torch.zeros(1, 8, 2, 2), torch.zeros(1, 8, 2, 2), torch.ones(1), torch.float32)
        pattern.append(calls[0])
    assert pattern == ["replica1", "replica2", "master", "replica1", "master"]


def test_setting_up_the_replicas_leaves_every_random_generator_as_it_was(monkeypatch):
    """A replica's network starts from random weights. Drawing them moved torch's CPU generator, which seeds the data
    loader's shuffle at every epoch: the server's first replica run trained other images than the single-GPU run with
    the same seed at most micro-steps (losses 1e-2 to 8e-2 apart; two single-GPU runs 1e-4)."""
    import random

    import numpy as np

    trainer, args = _replica_trainer(monkeypatch)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    cuda_states = {}
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device="cuda": cuda_states.setdefault(device, torch.zeros(4, dtype=torch.uint8)))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device="cuda": cuda_states.__setitem__(device, state))

    def create_replica(args, accelerator, index, device):
        torch.rand(3), random.random(), np.random.rand(2)  # what a network's initialization draws
        return types.SimpleNamespace(index=index, device=device, network=nn.Linear(2, 2), params=[])

    monkeypatch.setattr(trainer, "_create_replica", create_replica)
    accelerator = types.SimpleNamespace(device=torch.device("cuda:0"), num_processes=1, unwrap_model=lambda model: model)

    def seed():
        torch.manual_seed(5), random.seed(5), np.random.seed(5)

    def draws():
        return torch.rand(4), random.random(), np.random.rand(3)

    master_network = nn.Linear(2, 2)
    seed()
    expected = draws()
    seed()
    trainer.post_prepare_network(args, accelerator, None, master_network)
    got = draws()
    assert len(trainer._replicas) == 1 and sorted(cuda_states) == [0, 1]
    assert torch.equal(got[0], expected[0]) and got[1] == expected[1] and np.array_equal(got[2], expected[2])


def test_a_replicas_copies_are_issued_by_the_main_thread_before_its_micro_step(monkeypatch):
    """The master's side of a replica's copies (the updated weights, the inputs prepared on the master's device) runs on
    the master's stream. Issued from the worker thread it queued behind the master's next micro-step: the server's first
    replica run (GPUs 3 + 4) trained at 5.55 s/step against 5.64 on one GPU. call_dit issues them before it submits."""
    import threading

    trainer, args = _replica_trainer(monkeypatch, gradient_accumulation_steps=1)
    trainer.prepare_dit_inputs = lambda *a, **k: {"hidden": torch.ones(1, 8, 2, 2), "contexts": [torch.zeros(3, 48)]}
    master_params = [nn.Parameter(torch.full((3,), 2.0))]
    trainer._replica_master_params = master_params
    trainer._replica_anchor = master_params[0]
    seen = {}

    class _Fake:
        index, last_future, weights_dirty, device = 1, None, True, torch.device("cpu")

        def __init__(self):
            self.params = [nn.Parameter(torch.zeros(3))]

        def submit(self, fn):
            seen.update(weights=self.params[0].detach().clone(), dirty=self.weights_dirty)
            self.last_future = types.SimpleNamespace(result=lambda: 0.5)
            return self.last_future

    callers = []
    real_move = replica_trainer.move_to_device
    monkeypatch.setattr(replica_trainer, "move_to_device", lambda obj, device: callers.append(threading.current_thread()) or real_move(obj, device))
    trainer._replicas = [_Fake()]
    accelerator = _Accelerator(2)
    trainer.call_dit(args, accelerator, None, torch.zeros(1, 8, 2, 2), {}, torch.zeros(1, 8, 2, 2), torch.zeros(1, 8, 2, 2), torch.ones(1), torch.float32)
    assert seen["dirty"] is False and torch.equal(seen["weights"], torch.full((3,), 2.0))
    assert callers and all(thread is threading.main_thread() for thread in callers)


def test_a_replica_micro_step_that_may_compile_leaves_the_generators_as_they_were(monkeypatch):
    """torch.compile draws random numbers while it compiles, and a single-GPU run compiles the master only: the compiled
    replica run on the server (GPUs 3 + 4) matched the compiled single-GPU run in the first epoch and trained other
    images from the second on (losses up to 0.11 apart). A replica micro-step that runs alone (its first, the warmup,
    new shapes) leaves every generator as it was."""
    import random

    import numpy as np

    trainer, args = _replica_trainer(monkeypatch, gradient_accumulation_steps=1)
    trainer.prepare_dit_inputs = lambda *a, **k: {"hidden": torch.ones(1, 8, 2, 2), "contexts": [torch.zeros(3, 48)]}

    def run_dit(args, transformer, device, dtype, inputs):
        torch.rand(3), random.random(), np.random.rand(2)  # what compiling draws
        return torch.zeros(1, 4, requires_grad=True), torch.ones(1, 4)

    trainer.run_dit = run_dit
    replica = Replica(index=1, device=torch.device("cpu"), proxy=None, transformer=None, network=None, executor=ThreadPoolExecutor(max_workers=1))
    replica.weights_dirty = False
    trainer._replicas = [replica]
    trainer._replica_anchor = nn.Parameter(torch.zeros(1))

    def seed():
        torch.manual_seed(3), random.seed(3), np.random.seed(3)

    def draws():
        return torch.rand(4), random.random(), np.random.rand(3)

    seed()
    expected = draws()
    seed()
    out = trainer.call_dit(args, _Accelerator(2), None, torch.zeros(1, 8, 2, 2), {}, torch.zeros(1, 8, 2, 2), torch.zeros(1, 8, 2, 2), torch.ones(1), torch.float32)
    got = draws()
    replica.executor.shutdown()
    assert float(_loop_loss(out)) == pytest.approx(1.0)  # the replica's own loss, waited for
    assert torch.equal(got[0], expected[0]) and got[1] == expected[1] and np.array_equal(got[2], expected[2])


def test_the_stand_in_loss_is_the_replica_loss_with_a_zero_gradient(monkeypatch):
    trainer, _ = _replica_trainer(monkeypatch)
    anchor = nn.Parameter(torch.randn(3, 2))
    trainer._replica_anchor = anchor
    out = trainer._stand_in(0.37, torch.device("cpu"))
    loss = _loop_loss(out)
    loss.backward()
    assert float(loss) == pytest.approx(0.37) and torch.equal(anchor.grad, torch.zeros_like(anchor))
