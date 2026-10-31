"""Single-process multi-GPU LoRA training: the replica trainer, ported from kohya-ss/sd-scripts dev 88f1c38 (PR #2479,
library/replica_trainer.py, with the #2481 fix: each micro-step runs in a copy of the main thread's contextvars) for the
MOSS-Image LoRA trainer on musubi's extended training loop.

One Python process drives one *master* (the accelerator device) and one *replica* per additional GPU
(``--replica_devices``). Each replica holds its own copy of the frozen DiT and of the network. The data loader's
micro-steps are assigned to the devices in turn (replicas first, master last), a replica's micro-step runs in a worker
thread of its own, and at each optimizer step the replicas' gradients are summed into the master's, the master is updated
as usual and the replicas pull the new weights before their next micro-step. No process group: no torchrun, no NCCL /
gloo, so it also works on Windows; the text encoder and the VAE are needed on the master only (their outputs are cached).

``train_batch_size`` is per device, as with DDP: ``gradient_accumulation_steps`` is multiplied by the number of devices,
so the step count, the learning-rate schedule and the progress bar are those of a single-GPU run with the multiplied
accumulation (and of DDP with the same arguments).

Fork differences from kohya's:
- Every random draw of a micro-step (noise, timesteps, caption and control dropout, the caption candidate) is made in the
  main thread in the data loader's order (the trainer's ``prepare_dit_inputs``); a replica only runs the DiT forward,
  the loss and the backward. A replica run therefore feeds each micro-step the inputs a single-GPU run with the
  multiplied accumulation feeds it (kohya's replicas draw their own noise), and their losses match step for step up
  to GPU rounding (``MUSUBI_LOSS_TRACE`` writes each micro-step's loss). Setting up the replicas leaves every random
  generator as it was (their networks' random initial weights moved the data loader's shuffle seed).
- The master's side of a replica's copies (the updated weights, the inputs prepared on the master's device) is issued
  by the main thread before the micro-step is submitted, so it runs on the master's stream before the master's next
  micro-step instead of queueing behind it.
- The plugs are the trainer's ``call_dit`` (the master micro-step runs the normal loop; a replica micro-step returns a
  zero-gradient stand-in whose loss is that replica's previous loss, as kohya logs it), ``post_prepare_network`` and
  ``before_gradient_sync`` (two calls in training/training_loop.py) and ``get_checkpoint_metadata``.
- The accumulation window's last micro-step always runs on the master (an epoch's last, partial window included), and
  every window starts with the first replica.

Limits: CUDA only, MOSS-Image LoRA / DoRA on the BF16 DiT (no FP8 / ConvRot INT8 base, no colour / depth control adapter,
whose embedder trains outside the network), mixed precision bf16 or no, a single process, no --base_weights, loss
weighting none (or logit_normal / mode, which only shape the sampling), no differential guidance, adaptive rank,
Self-Flow or CREPA.

Implementation notes (kohya's measurements): issuing every device's micro-steps from one thread serializes them, hence
a worker thread per replica; torch.compile is not thread-safe while it compiles, so the first ``--replica_warmup_steps``
optimizer steps and every micro-step whose input shapes are new to its device run alone; cross-device copies of the
weights and gradients go through pinned host memory (``PinnedStager``): the direct device-to-device path is slow on
Windows (no P2P).
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import json
import logging
import math
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

LOSS_TRACE_ENV = "MUSUBI_LOSS_TRACE"  # a JSON-lines file: every micro-step's loss (verification of replica runs)
_TRACE_LOCK = threading.Lock()


# region arguments


def add_replica_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--replica_devices",
        type=str,
        default=None,
        help="single-process multi-GPU LoRA training: comma-separated CUDA devices of the additional replicas, e.g. `1` or "
        "`cuda:1,cuda:2` (indices among the visible GPUs). The master runs on the accelerator device. The batch size is "
        "per device and gradient accumulation is multiplied by the device count, as with DDP (no DDP / NCCL; works on "
        "Windows)",
    )
    parser.add_argument(
        "--replica_warmup_steps",
        type=int,
        default=None,
        help="optimizer steps at the start during which the replicas' micro-steps run one at a time (torch.compile is not "
        "thread-safe while compiling). Default: 2 with --compile, 0 otherwise",
    )
    return parser


def parse_replica_devices(spec: str) -> List[torch.device]:
    """``"1,2"`` / ``"cuda:1, cuda:2"`` -> ``[cuda:1, cuda:2]``. Each entry must name a CUDA device with an index."""
    devices = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"cuda:{token}"
        try:
            device = torch.device(token)
        except RuntimeError as exc:
            raise ValueError(f"--replica_devices entries must be CUDA devices with an index (e.g. `1` or `cuda:1`), got `{token}`") from exc
        if device.type != "cuda" or device.index is None:
            raise ValueError(f"--replica_devices entries must be CUDA devices with an index (e.g. `1` or `cuda:1`), got `{token}`")
        devices.append(device)
    if not devices:
        raise ValueError("--replica_devices is empty")
    if len(set(devices)) != len(devices):
        raise ValueError(f"--replica_devices names a device twice: {spec}")
    return devices


def build_schedule(num_devices: int) -> List[int]:
    """Device index (0 = master) of each micro-step within a round: replicas first, master last. The master's micro-step
    runs in the main thread, so dispatching the replicas first lets them run while the master is busy."""
    return list(range(1, num_devices)) + [0]


# endregion

# region accelerator proxy


class AcceleratorDeviceProxy:
    """An ``Accelerator`` look-alike for one replica: ``device`` is the replica's device, ``prepare`` / ``unwrap_model`` are
    the identity (the replica trainer places the models itself and accelerate must not register them); everything else
    is the real accelerator's (``autocast``, ``backward``, ``print``, ...)."""

    def __init__(self, accelerator, device: torch.device):
        self._accelerator = accelerator
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def prepare(self, *args, device_placement=None, **kwargs):
        for index, obj in enumerate(args):
            place = device_placement[index] if device_placement is not None else True
            if place and isinstance(obj, torch.nn.Module):
                obj.to(self._device)
        return args[0] if len(args) == 1 else args

    def unwrap_model(self, model, keep_fp32_wrapper: bool = True):
        return model

    def __getattr__(self, name):
        return getattr(self._accelerator, name)


# endregion

# region cross-device copies


class PinnedStager:
    """Copies a list of tensors from one CUDA device to another through a pinned host buffer. The copies are stream
    ordered on both sides (D2H on the source's current stream, H2D on the destination's after an event) and do not block
    the CPU, except when a staging buffer is reused before its previous H2D finished. One buffer per ``key`` and dtype."""

    def __init__(self):
        self._buffers: Dict[Tuple[Any, torch.dtype], torch.Tensor] = {}
        self._in_flight: Dict[Tuple[Any, torch.dtype], torch.cuda.Event] = {}

    def _buffer(self, key, numel: int, dtype: torch.dtype) -> torch.Tensor:
        k = (key, dtype)
        buf = self._buffers.get(k)
        if buf is None or buf.numel() < numel:
            buf = torch.empty(numel, dtype=dtype, pin_memory=True)
            self._buffers[k] = buf
        return buf

    @torch.no_grad()
    def copy_flat(self, srcs: Sequence[torch.Tensor], dsts: Sequence[torch.Tensor], key, accumulate: bool = False):
        """``dsts[i] (+)= srcs[i]`` for all i; all ``srcs`` on one device, all ``dsts`` on one device (may be the same)."""
        assert len(srcs) == len(dsts)
        if not srcs:
            return
        by_dtype: Dict[torch.dtype, List[int]] = {}
        for index, tensor in enumerate(srcs):
            by_dtype.setdefault(tensor.dtype, []).append(index)
        src_device, dst_device = srcs[0].device, dsts[0].device
        for dtype, indices in by_dtype.items():
            flat = torch.cat([srcs[i].reshape(-1) for i in indices])
            if src_device == dst_device:
                out = flat
            else:
                k = (key, dtype)
                pending = self._in_flight.get(k)
                if pending is not None:
                    pending.synchronize()  # the previous H2D read of this buffer must be done before overwriting it
                buf = self._buffer(key, flat.numel(), dtype)[: flat.numel()]
                buf.copy_(flat, non_blocking=True)  # D2H on the source's current stream
                done = torch.cuda.Event()
                done.record(torch.cuda.current_stream(src_device))
                dst_stream = torch.cuda.current_stream(dst_device)
                dst_stream.wait_event(done)
                with torch.cuda.device(dst_device), torch.cuda.stream(dst_stream):
                    out = buf.to(dst_device, non_blocking=True)
                read = torch.cuda.Event()
                read.record(dst_stream)
                self._in_flight[k] = read
            # one foreach call instead of one kernel launch per tensor (hundreds of LoRA tensors)
            views, targets, offset = [], [], 0
            for i in indices:
                d = dsts[i]
                views.append(out[offset : offset + d.numel()].view_as(d))
                targets.append(d)
                offset += d.numel()
            if accumulate:
                torch._foreach_add_(targets, views)
            else:
                torch._foreach_copy_(targets, views)


# endregion


@contextlib.contextmanager
def preserved_random_states():
    """Python's, NumPy's, torch's CPU and every visible CUDA device's random generator as they were before the block."""
    import random

    python_state = random.getstate()
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a dependency
        np = None
    numpy_state = np.random.get_state() if np is not None else None
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            yield
        finally:
            random.setstate(python_state)
            if np is not None:
                np.random.set_state(numpy_state)


def move_to_device(obj, device: torch.device):
    """A copy of ``obj`` with every tensor (also inside lists / tuples / dicts) on ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(value, device) for value in obj)
    return obj


def input_shape_signature(inputs: dict) -> Tuple[int, ...]:
    """The shapes of a prepared micro-step that can make torch.compile recompile: the latent (batch, channels, h, w),
    each context's length and each reference latent's shape, as a flat tuple (positions identify the dimensions)."""
    signature: List[int] = list(inputs["hidden"].shape)
    signature.extend(int(context.shape[0]) for context in inputs.get("contexts") or [])
    for refs in inputs.get("ref_latents") or []:
        for ref in refs:
            signature.extend(ref.shape)
    return tuple(signature)


class ShapeTracker:
    """Whether a micro-step may make torch.compile (re)compile on a device. Mirrors Dynamo's automatic dynamic shapes: a
    dimension is specialized to its first value, recompiled as dynamic at a second one and free afterwards; with
    ``compile_dynamic`` "true" only the first shape is new, with "false" every new value is."""

    def __init__(self, compile_dynamic: Optional[str]):
        self.mode = str(compile_dynamic or "auto").lower()
        self._seen: Dict[int, List[set]] = {}

    def is_new(self, device_index: int, signature: Tuple[int, ...]) -> bool:
        seen = self._seen.get(device_index)
        if seen is None or len(seen) != len(signature):
            self._seen[device_index] = [{value} for value in signature]
            return True
        if self.mode == "true":
            return False
        new = False
        for values, value in zip(seen, signature):
            if value not in values:
                if self.mode == "false" or len(values) < 2:
                    new = True
                values.add(value)
        return new


class _Timing:
    """Mean durations of the micro-step phases, printed every N optimizer steps for MUSUBI_REPLICA_TIMING=N (any other
    non-empty value: every 10 steps; thread-safe enough for logging)."""

    def __init__(self, interval: int = 10):
        self._values: Dict[str, List[float]] = defaultdict(list)
        self.last_master_start: Optional[float] = None
        self.interval = max(1, int(interval))

    @classmethod
    def from_env(cls) -> Optional["_Timing"]:
        value = os.environ.get("MUSUBI_REPLICA_TIMING", "").strip()
        if not value:
            return None
        return cls(int(value) if value.isdigit() and int(value) > 0 else 10)

    def add(self, **durations: float):
        for key, value in durations.items():
            self._values[key].append(value)

    def summary(self) -> str:
        parts = [f"{key}={sum(values) / len(values):.3f}" for key, values in sorted(self._values.items()) if values]
        self._values.clear()
        return " ".join(parts)


def trace_loss(micro_step: int, device, loss: float) -> None:
    """One line of the MUSUBI_LOSS_TRACE file (when set): the micro-step's index, device and loss."""
    path = os.environ.get(LOSS_TRACE_ENV)
    if not path:
        return
    with _TRACE_LOCK, open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"micro_step": int(micro_step), "device": str(device), "loss": float(loss)}) + "\n")


# region replica


@dataclass
class Replica:
    index: int  # 1-based (0 is the master)
    device: torch.device
    proxy: AcceleratorDeviceProxy
    transformer: torch.nn.Module
    network: torch.nn.Module
    executor: ThreadPoolExecutor
    params: List[torch.nn.Parameter] = field(default_factory=list)
    last_future: Optional[Future] = None
    weights_dirty: bool = True  # pull the master's weights before the next micro-step

    def submit(self, fn) -> Future:
        # in a copy of the caller's context: torch._dynamo / torch._inductor config overrides made in the main thread
        # (recompile and cache size limits) live in contextvars, which a worker thread does not inherit (kohya #2481)
        self.last_future = self.executor.submit(contextvars.copy_context().run, fn)
        return self.last_future

    def join(self):
        """Wait for the in-flight micro-step (re-raises its exception in the caller)."""
        if self.last_future is not None:
            self.last_future.result()


class ReplicaTrainerMixin:
    """In front of the MOSS-Image LoRA trainer (``wrap_trainer_class``); see the module docstring."""

    # region hooks

    def handle_model_specific_args(self, args):
        super().handle_model_specific_args(args)
        if not torch.cuda.is_available():
            raise ValueError("--replica_devices needs CUDA")
        self.replica_devices = parse_replica_devices(args.replica_devices)
        num_devices = 1 + len(self.replica_devices)

        def require(condition: bool, what: str):
            if not condition:
                raise ValueError(f"--replica_devices needs {what}")

        require(str(args.mixed_precision or "no") != "fp16", "--mixed_precision bf16 or no (fp16 is not supported)")
        require(not (args.fp8_base or args.fp8_scaled), "the BF16 DiT (no --fp8_base / --fp8_scaled)")
        require(not getattr(args, "convrot_int8", False), "the BF16 DiT (no --convrot_int8)")
        require(not getattr(args, "nf4_base", False), "the BF16 DiT (no --nf4_base)")
        require(not getattr(self, "control_adapter", False), "a LoRA / DoRA without --control_adapter (its embedder trains outside the network)")
        require(not getattr(args, "base_weights", None), "no --base_weights")
        require(not getattr(args, "differential_guidance", False), "no --differential_guidance")
        require(not getattr(args, "self_flow", False), "no --self_flow")
        require(str(getattr(args, "weighting_scheme", "none") or "none") not in ("sigma_sqrt", "cosmap"), "--weighting_scheme none, logit_normal or mode")
        require(str(getattr(args, "network_module", "") or "").endswith("lora_moss_image"), "--network_module networks.lora_moss_image")

        self._replica_orig_gradient_accumulation_steps = int(args.gradient_accumulation_steps)
        args.gradient_accumulation_steps = self._replica_orig_gradient_accumulation_steps * num_devices
        self._replica_warmup_steps = args.replica_warmup_steps
        if self._replica_warmup_steps is None:
            self._replica_warmup_steps = 2 if args.compile else 0
        # torch.compile is not thread-safe while compiling: a micro-step that may compile runs alone (the warmup, and
        # later the micro-steps whose input shapes are new to their device)
        self._replica_shapes = ShapeTracker(getattr(args, "compile_dynamic", None)) if args.compile else None
        self._replica_timing = _Timing.from_env()
        print(
            f"replica training: master + {len(self.replica_devices)} replica(s) on {[str(d) for d in self.replica_devices]}, "
            f"gradient_accumulation_steps {self._replica_orig_gradient_accumulation_steps} x {num_devices} = "
            f"{args.gradient_accumulation_steps} micro-steps per optimizer step, {self._replica_warmup_steps} serial warmup step(s)"
        )
        self._replicas: List[Replica] = []
        self._replica_schedule = build_schedule(num_devices)
        self._replica_micro_step = 0  # position in the current accumulation window
        self._replica_dispatched = 0  # micro-steps so far (the loss trace's index)
        self._replica_optimizer_steps = 0
        self._replica_master_params: Optional[List[torch.nn.Parameter]] = None
        self._replica_anchor: Optional[torch.nn.Parameter] = None
        self._replica_stager = PinnedStager()

    def post_prepare_network(self, args, accelerator, transformer, network):
        """After the master's DiT and network are prepared: one replica per extra device, set up as the loop set up the
        master."""
        if accelerator.num_processes != 1:
            raise ValueError("--replica_devices cannot be combined with multi-process (DDP) training: launch one process")
        if accelerator.device.type != "cuda":
            raise ValueError(f"--replica_devices needs the master on a CUDA device, got {accelerator.device}")
        unwrapped = accelerator.unwrap_model(network)
        if hasattr(unwrapped, "has_adaptive_rank") and unwrapped.has_adaptive_rank():
            raise ValueError("--replica_devices does not support adaptive rank networks (their parameters change during training)")
        if getattr(self, "_frozen_networks", None):
            raise ValueError("--replica_devices does not support frozen extra networks")
        for device in self.replica_devices:
            if device == accelerator.device:
                logger.warning(f"replica device {device} is the master device; this only makes sense for testing")
            if device.index >= torch.cuda.device_count():
                raise ValueError(f"replica device {device} is not visible ({torch.cuda.device_count()} CUDA devices): set gpu_ids / CUDA_VISIBLE_DEVICES to the master and the replicas")
        # A replica's network starts from random weights (replaced by the master's before its first micro-step): those
        # draws must not move any generator, or the data loader's shuffle (seeded from torch's CPU generator at each
        # epoch) and every later draw differ from the single-GPU run with the same seed
        with preserved_random_states():
            for index, device in enumerate(self.replica_devices, start=1):
                self._replicas.append(self._create_replica(args, accelerator, index, device))
        self._ensure_param_lists(accelerator, network)

    def call_dit(self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs):
        """Prepare the micro-step in the main thread (every random draw), then run it on its device: the master's in the
        normal loop, a replica's in its worker thread (the loop then gets a zero-gradient stand-in)."""
        from musubi_tuner.training.outputs import DiTOutput

        sync = bool(accelerator.sync_gradients)
        target = self._replica_schedule[self._replica_micro_step % len(self._replica_schedule)]
        self._replica_micro_step = 0 if sync else self._replica_micro_step + 1
        if sync:
            target = 0  # the window's last micro-step runs where the optimizer steps (an epoch's partial window too)
        micro_step = self._replica_dispatched
        self._replica_dispatched += 1
        self._mirror_step_start(args)
        inputs = self.prepare_dit_inputs(args, accelerator.device, latents, batch, noise, noisy_model_input, timesteps, network_dtype)

        if target == 0:
            if self._must_run_alone(0, inputs):
                for replica in self._replicas:
                    replica.join()  # a compilation on the master must not overlap a replica's micro-step
            t0 = time.perf_counter()
            if self._replica_timing is not None:
                self._replica_timing.last_master_start = t0
            try:
                pred, target_tensor = self.run_dit(args, transformer, accelerator.device, network_dtype, inputs)
            except RuntimeError as exc:
                raise self._annotate_compile_race(exc) from exc
            if os.environ.get(LOSS_TRACE_ENV):
                trace_loss(micro_step, accelerator.device, self._traced_loss(args, pred, target_tensor, network_dtype))
            if self._replica_timing is not None:
                self._replica_timing.add(master_dispatch=time.perf_counter() - t0)
            return DiTOutput(pred=pred, target=target_tensor)

        replica = self._replicas[target - 1]
        timing = self._replica_timing
        # The master's side of the replica's copies (the updated weights, the prepared inputs) is issued here, in the main
        # thread, so it runs on the master's stream right after the optimizer step and before the master's next
        # micro-step. Issued from the worker thread (kohya's place), it queued behind whatever the main thread had issued
        # by then, and the inputs, prepared on the master's device here, are a cross-device copy on the master's stream.
        t0 = time.perf_counter()
        if replica.weights_dirty:
            self._replica_stager.copy_flat(self._replica_master_params, replica.params, key=("weights", replica.index))
            replica.weights_dirty = False
        moved = move_to_device(inputs, replica.device)
        submitted = time.perf_counter()
        if timing is not None:
            timing.add(replica_stage=submitted - t0)

        def run_micro_step() -> float:
            t1 = time.perf_counter()
            with accelerator.autocast():  # the master's prepared DiT runs under accelerate's autocast too
                pred, target_tensor = self.run_dit(args, replica.transformer, replica.device, network_dtype, moved)
            pred = pred.float()  # accelerate's convert_outputs_to_fp32 on the master's prepared DiT
            loss = self._replica_loss(args, pred, target_tensor, network_dtype)
            accelerator.backward(loss)  # divided by the (multiplied) accumulation, as the master's micro-steps
            t2 = time.perf_counter()
            value = loss.detach().item()  # waits for this replica's GPU only (worker thread)
            trace_loss(micro_step, replica.device, value)
            if timing is not None:
                t3 = time.perf_counter()
                timing.add(replica_start=t1 - submitted, replica_cpu=t2 - t1, replica_item=t3 - t2, replica_total=t3 - t1)
            return value

        previous = replica.last_future
        if previous is None or self._must_run_alone(replica.index, inputs):
            # first micro-step of this replica, warmup or new shapes: wait for this one. Such a micro-step may compile,
            # and compiling draws random numbers (a single-GPU run compiles the master only): keep the generators as
            # they were, or the data loader's shuffle from the next epoch on differs from the single-GPU run's
            with preserved_random_states():
                value = self._wait(replica.submit(run_micro_step))
        else:
            replica.submit(run_micro_step)
            value = self._wait(previous)  # the loss of this replica's previous micro-step (lags by one dispatch)
        return self._stand_in(value, accelerator.device)

    def before_gradient_sync(self, accelerator, network):
        """At the window's last micro-step (the master's), before clipping and the optimizer step: wait for the replicas
        and sum their gradients into the master's; they pull the updated weights before their next micro-step."""
        master_params = self._replica_master_params
        t0 = time.perf_counter()
        timing = self._replica_timing
        if timing is not None and timing.last_master_start is not None:
            timing.add(master_step=t0 - timing.last_master_start)  # the master's forward and backward (CPU side)
        for replica in self._replicas:
            replica.join()
            t1 = time.perf_counter()
            grads = [p.grad for p in replica.params]
            if all(g is None for g in grads):
                continue  # no micro-step on this replica in the window (a partial window at the end of the data)
            if any(g is None for g in grads):
                raise RuntimeError(f"replica {replica.index}: some parameters have no gradient")
            for p in master_params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            self._replica_stager.copy_flat(grads, [p.grad for p in master_params], key=("grads", replica.index), accumulate=True)
            for p in replica.params:
                p.grad = None
            replica.weights_dirty = True
            if timing is not None:
                timing.add(sync_join=t1 - t0, sync_reduce=time.perf_counter() - t1)
        self._replica_optimizer_steps += 1
        if timing is not None and self._replica_optimizer_steps % timing.interval == 0:
            print(f"replica timing (mean s over the last {timing.interval} steps): {timing.summary()}")

    def get_checkpoint_metadata(self, args) -> dict:
        metadata = dict(super().get_checkpoint_metadata(args) or {})
        metadata["ss_gradient_accumulation_steps"] = self._replica_orig_gradient_accumulation_steps
        metadata["ss_replica_devices"] = ",".join(str(device) for device in self.replica_devices)
        return metadata

    # endregion

    def _mirror_step_start(self, args):
        """The loop calls the master network's on_step_start each micro-step; the replicas' networks follow."""
        step = getattr(self, "_current_train_global_step", None)
        for replica in self._replicas:
            network = getattr(replica, "network", None)
            if network is not None and hasattr(network, "on_step_start"):
                network.on_step_start(global_step=step, max_train_steps=args.max_train_steps)

    @staticmethod
    def _replica_loss(args, pred: torch.Tensor, target: torch.Tensor, network_dtype) -> torch.Tensor:
        """The loop's plain loss: the per-element loss of the prediction (in the network's dtype) and the FP32 target, mean
        over every element (the supported weighting schemes do not weight the loss)."""
        from musubi_tuner.training.losses import per_element_loss

        pred = pred.to(device=target.device, dtype=network_dtype)
        return per_element_loss(pred, target, getattr(args, "loss_type", "mse"), getattr(args, "huber_delta", 1.0)).mean()

    def _traced_loss(self, args, pred, target, network_dtype) -> float:
        with torch.no_grad():
            return float(self._replica_loss(args, pred.detach().float(), target, network_dtype))

    def _stand_in(self, value: float, device):
        """A replica micro-step's output for the loop: its plain loss equals ``value`` (the loss log) and its backward
        reaches one master parameter element with a zero gradient (the network's gradients come from the replica)."""
        from musubi_tuner.training.outputs import DiTOutput

        anchor = self._replica_anchor.reshape(-1)[0].float()
        target = torch.zeros(1, device=device, dtype=torch.float32)
        pred = target + math.sqrt(max(float(value), 0.0)) + 0.0 * anchor.to(device)
        return DiTOutput(pred=pred, target=target)

    def _must_run_alone(self, device_index: int, inputs: dict) -> bool:
        if self._replica_optimizer_steps < self._replica_warmup_steps:
            return True
        if self._replica_shapes is None:
            return False
        return self._replica_shapes.is_new(device_index, input_shape_signature(inputs))

    @staticmethod
    def _wait(future: Future):
        try:
            return future.result()
        except RuntimeError as exc:
            raise ReplicaTrainerMixin._annotate_compile_race(exc) from exc

    @staticmethod
    def _annotate_compile_race(exc: RuntimeError) -> RuntimeError:
        if "symbolically trace a dynamo-optimized function" not in str(exc):
            return exc
        return RuntimeError(
            "torch.compile compiled a block while another device was running (not thread-safe). Increase "
            "--replica_warmup_steps, or train without --compile when using --replica_devices"
        )

    def _ensure_param_lists(self, accelerator, network):
        if self._replica_master_params is not None:
            return
        master = accelerator.unwrap_model(network)
        self._replica_master_params = [p for p in master.parameters() if p.requires_grad]
        if not self._replica_master_params:
            raise RuntimeError("replica training: the network has no trainable parameters")
        self._replica_anchor = self._replica_master_params[0]
        for replica in self._replicas:
            replica.params = [p for p in replica.network.parameters() if p.requires_grad]
            if len(replica.params) != len(self._replica_master_params) or any(
                a.shape != b.shape or a.dtype != b.dtype for a, b in zip(replica.params, self._replica_master_params)
            ):
                raise RuntimeError(
                    f"replica {replica.index}: trainable parameters do not match the master ({len(replica.params)} vs "
                    f"{len(self._replica_master_params)})"
                )
        print(f"replica training: {len(self._replica_master_params)} trainable tensors synchronized per step")

    def _create_replica(self, args, accelerator, index: int, device: torch.device) -> Replica:
        """The frozen DiT and the network of one replica, set up as training_loop.train sets up the master's."""
        import importlib

        from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
        from musubi_tuner.training.training_loop import _enable_transformer_gradient_checkpointing
        from musubi_tuner.utils import model_utils
        from musubi_tuner.utils.arg_utils import parse_text_args

        print(f"replica training: creating replica {index} on {device}")
        proxy = AcceleratorDeviceProxy(accelerator, device)
        blocks_to_swap = int(getattr(self, "blocks_to_swap", 0) or 0)
        loading_device = "cpu" if blocks_to_swap > 0 else device
        dit_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)
        attn_mode = "torch" if args.sdpa else "flash" if args.flash_attn else "xformers" if args.xformers else "torch"
        network_dtype = torch.float32
        if args.full_bf16 or args.full_fp16:
            network_dtype = torch.bfloat16 if args.full_bf16 else torch.float16
        network_module = importlib.import_module(args.network_module)
        net_kwargs = parse_text_args(args.network_args) if args.network_args is not None else {}

        with torch.cuda.device(device):
            transformer = self.load_transformer(proxy, args, args.dit, attn_mode, args.split_attn, loading_device, dit_dtype)
            transformer.eval()
            transformer.requires_grad_(False)
            if blocks_to_swap > 0:
                transformer.enable_block_swap(blocks_to_swap, BlockSwapConfig.from_args(args, device, supports_backward=True))
                transformer.move_to_device_except_swap_blocks(device)
            if args.dim_from_weights:
                weights_sd = self.load_network_weights(args.dim_from_weights, network_module)
                network = network_module.create_arch_network_from_weights(
                    1, weights_sd, unet=transformer, neuron_dropout=args.network_dropout, **net_kwargs
                )
            else:
                network = network_module.create_arch_network(
                    1.0, args.network_dim, args.network_alpha, None, None, transformer, neuron_dropout=args.network_dropout, **net_kwargs
                )
            if network is None:
                raise RuntimeError("the network module returned no network for the replica")
            if callable(getattr(network, "prepare_network", None)):
                network.prepare_network(args)
            network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
            if args.gradient_checkpointing:
                _enable_transformer_gradient_checkpointing(
                    transformer, args.gradient_checkpointing_cpu_offload, blocks_to_checkpoint=getattr(args, "blocks_to_checkpoint", -1)
                )
                try:
                    network.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
                except TypeError:
                    network.enable_gradient_checkpointing()
            if network_dtype != torch.float32:
                network.to(network_dtype)
            if blocks_to_swap > 0:
                transformer.move_to_device_except_swap_blocks(device)
                transformer.prepare_block_swap_before_forward()
            else:
                transformer.to(device)
            if args.compile:
                transformer = self.compile_transformer(args, transformer)
            transformer.train() if args.gradient_checkpointing else transformer.eval()
            network.to(device)
            network.prepare_grad_etc(transformer)
            if hasattr(network, "on_epoch_start"):
                network.on_epoch_start(transformer)  # train mode, as the loop does for the master at each epoch
            network.train()

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"replica{index}", initializer=torch.cuda.set_device, initargs=(device,))
        return Replica(index=index, device=device, proxy=proxy, transformer=transformer, network=network, executor=executor)


def wrap_trainer_class(trainer_class: type, args: argparse.Namespace) -> type:
    """``trainer_class`` with ``ReplicaTrainerMixin`` in front when ``--replica_devices`` is given, else as it is."""
    if not str(getattr(args, "replica_devices", None) or "").strip():
        return trainer_class
    return type(f"Replica{trainer_class.__name__}", (ReplicaTrainerMixin, trainer_class), {})


# endregion
