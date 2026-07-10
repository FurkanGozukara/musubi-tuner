# Image Model Full Finetune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full-DiT finetuning entrypoints for FLUX.1 Kontext, FLUX.2, Ideogram 4, and Krea 2 without changing existing LoRA training semantics.

**Architecture:** Add one repository-internal `FullFineTuningTrainerMixin` with a complete full-model lifecycle and four thin model trainers that inherit existing architecture hooks. The lifecycle keeps explicit raw and Accelerate-prepared model identities, uses model-owned dtype/sampling policies, rejects unsupported optimization combinations early, and exports native safetensors checkpoints. Existing LoRA loops remain separate.

**Tech Stack:** Python 3.10+, PyTorch, Accelerate, safetensors, pytest, ruff, existing Musubi dataset/training utilities.

## Global Constraints

- Baseline is `upstream/main` commit `30c658c4f4b0bf05038b3346eff9670259b10fc7` on local branch `finetune`.
- Full finetuning optimizes the DiT/Transformer only; text encoders, VAEs, and optional unconditional models stay frozen.
- Default trainable parameters are fp32; `--full_bf16` requires `--mixed_precision bf16`; full fp16 is unsupported.
- Reject trainable FP8/BNB bases and `--block_swap_h2d_only`.
- Ordinary full-model block swap is single-process only and uses backward-capable offloading.
- Fused backward is single-process, Adafactor `step_param` only, and requires `gradient_accumulation_steps == 1`.
- Multi-process training is supported only without block swap and fused backward.
- Save precision must equal the trainable parameter dtype in this pass.
- FLUX.2 reload still requires explicit matching `--model_version`; checkpoint metadata is provenance only.
- Krea 2 primary model type is `--dit_variant raw|turbo`; full trainers reject temporary `--turbo_dit` swapping.
- Do not add architecture-name conditionals to the shared engine.
- Keep all existing LoRA and existing full-finetune tests passing.

---

## File Map

- Create `src/musubi_tuner/training/full_finetune.py`: shared validation, progress state, raw/prepared model lifecycle, optimizer loop, sampling cleanup, state/export saving.
- Create `src/musubi_tuner/flux_kontext_train.py`: Kontext full trainer and parser composition.
- Create `src/musubi_tuner/flux_2_train.py`: FLUX.2 full trainer and model-version metadata.
- Create `src/musubi_tuner/ideogram4_train.py`: Ideogram conditional-DiT full trainer, quantized-header rejection, unconditional sampling policy.
- Create `src/musubi_tuner/krea2_train.py`: Krea full trainer and Turbo-swap rejection.
- Create four root `*_train.py` shims matching the existing entrypoint style.
- Modify four `*_train_network.py` modules only for behavior-neutral dtype, unwrap, and policy seams shared with full training.
- Create focused tests under `tests/test_full_finetune_*.py`; extend `tests/test_top_level_entrypoints.py` for shims.
- Modify the four architecture documents and `README.md` with full-finetune commands and limitations.

---

### Task 1: Shared Full-Finetune Primitives

**Files:**
- Create: `src/musubi_tuner/training/full_finetune.py`
- Create: `tests/test_full_finetune_core.py`

**Interfaces:**
- Produces: `TrainingProgress`, `add_full_finetune_args`, `resolve_trainable_dtype`, `validate_full_finetune_args`, `normalize_compiled_state_dict`.
- Consumes: `model_utils.str_to_dtype`, `train_utils.resolve_save_dtype`, common parser argument names.

- [ ] **Step 1: Write failing progress and state-dict tests**

```python
def test_training_progress_round_trip():
    progress = TrainingProgress(global_step=7, epoch=2, next_batch=3)
    restored = TrainingProgress()
    restored.load_state_dict(progress.state_dict())
    assert restored == progress


def test_normalize_compiled_state_dict_rejects_collision():
    state = {"blocks.0.weight": torch.ones(1), "blocks.0._orig_mod.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="collision"):
        normalize_compiled_state_dict(state)
```

- [ ] **Step 2: Run the focused tests and confirm missing imports fail**

Run: `python -m pytest tests/test_full_finetune_core.py -q`

Expected: collection fails because `musubi_tuner.training.full_finetune` does not exist.

- [ ] **Step 3: Implement progress, dtype, key normalization, and parser arguments**

```python
@dataclass(eq=True)
class TrainingProgress:
    global_step: int = 0
    epoch: int = 0
    next_batch: int = 0

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.global_step = int(state["global_step"])
        self.epoch = int(state["epoch"])
        self.next_batch = int(state["next_batch"])


def normalize_compiled_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        key = key.removeprefix("_orig_mod.").replace("._orig_mod.", ".")
        if key in normalized:
            raise ValueError(f"compiled state_dict key collision: {key}")
        normalized[key] = value
    return normalized
```

Add `--full_bf16`, `--fused_backward_pass`, `--mem_eff_save`, and `--block_swap_optimizer_patch_params` in `add_full_finetune_args`.

- [ ] **Step 4: Add table-driven validation tests**

Cover non-default LoRA options, FP8 flags, H2D-only swap, multi-process ordinary swap, fused non-Adafactor, fused multi-process, fused accumulation, mismatched save precision, and full-bf16 without bf16 mixed precision. Each test calls `validate_full_finetune_args(args, num_processes)` and asserts the invalid option appears in the error.

- [ ] **Step 5: Implement strict shared validation**

Use a mapping of argument name to its inert default. Do not branch on model architecture. Permit `network_alpha == 1` because that is the common parser default; reject any other value.

- [ ] **Step 6: Run and lint the core tests**

Run: `python -m pytest tests/test_full_finetune_core.py -q`

Expected: all tests pass.

Run: `ruff check src/musubi_tuner/training/full_finetune.py tests/test_full_finetune_core.py`

Expected: no diagnostics.

- [ ] **Step 7: Commit the primitives**

```text
git add src/musubi_tuner/training/full_finetune.py tests/test_full_finetune_core.py
git commit -m "feat: add full finetune training primitives"
```

---

### Task 2: Shared Full-Model Lifecycle

**Files:**
- Modify: `src/musubi_tuner/training/full_finetune.py`
- Create: `tests/test_full_finetune_runtime.py`

**Interfaces:**
- Produces: `FullFineTuningTrainerMixin.train(args)`, `validate_full_finetune_model_args`, `load_full_finetune_transformer`, `full_finetune_metadata`, `save_state_all_ranks`.
- Consumes: Task 1 primitives and inherited `NetworkTrainer` architecture/optimizer/scheduler/sampling helpers.

- [ ] **Step 1: Build a tiny fake trainer test fixture**

The fake transformer exposes `enable_gradient_checkpointing`, block-swap lifecycle methods, and a single `nn.Linear`.
The fake trainer inherits `FullFineTuningTrainerMixin, NetworkTrainer`, returns a two-item cached dataset, and implements
`call_dit` with a scalar MSE target. Its prepared wrapper intentionally has no `.config` attribute and records whether
forward used the wrapper.

- [ ] **Step 2: Write failing lifecycle tests**

Tests assert:

```python
assert trainer.loader_dtype is torch.float32
assert trainer.forward_wrapper_called
assert trainer.compile_called_before_prepare
assert trainer.optimizer_param_ids == {id(p) for p in trainer.raw_model.parameters()}
assert trainer.saved_state_dict.keys() == {"linear.weight", "linear.bias"}
```

Also assert frozen sample VAE parameters never enter the optimizer and a one-step run changes a transformer weight.

- [ ] **Step 3: Run the runtime tests and confirm `train` is missing**

Run: `python -m pytest tests/test_full_finetune_runtime.py -q`

Expected: failures identify the missing lifecycle methods.

- [ ] **Step 4: Implement the raw/prepared training lifecycle**

The implementation order is fixed:

```python
raw_model = self.load_full_finetune_transformer(..., trainable_dtype)
raw_model.requires_grad_(True)
self.on_transformer_loaded(args, accelerator, raw_model)
if args.gradient_checkpointing:
    raw_model.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
if blocks_to_swap:
    raw_model.enable_block_swap(blocks_to_swap, BlockSwapConfig.from_args(args, accelerator.device, supports_backward=True))
    raw_model.move_to_device_except_swap_blocks(accelerator.device)
if args.compile:
    raw_model = self.compile_transformer(args, raw_model)
if blocks_to_swap:
    forward_model = accelerator.prepare(raw_model, device_placement=[False])
else:
    forward_model = accelerator.prepare(raw_model)
raw_model = accelerator.unwrap_model(forward_model, keep_fp32_wrapper=False)
if blocks_to_swap:
    raw_model.move_to_device_except_swap_blocks(accelerator.device)
    raw_model.prepare_block_swap_before_forward()
```

Prepare optimizer/dataloader/scheduler without preparing the model a second time. Use `forward_model` for
`accelerator.accumulate` and `process_batch`; use `raw_model` for config/save/swap operations. Do not manually reduce
DDP gradients.

- [ ] **Step 5: Implement exception-safe sampling**

Snapshot CPU/CUDA RNG state and previous model mode before `sample_images`. In `finally`, call the architecture after
hook, switch the unwrapped model back to training block-swap mode, restore train/eval mode, restore RNG state, and clean
device memory.

- [ ] **Step 6: Implement native checkpoint export**

Copy metadata before adding SAI fields, include `ss_training_type=full-finetune`, `ss_full_finetune=True`, and
`ss_fp8_base=False`, merge `full_finetune_metadata(args)`, normalize compiled keys, and save only the raw transformer.
Use `mem_eff_save_file` when requested and `save_file` otherwise. Only the main process exports or uploads.

- [ ] **Step 7: Run runtime tests, py_compile, and ruff**

Run: `python -m pytest tests/test_full_finetune_runtime.py tests/test_full_finetune_core.py -q`

Expected: all tests pass.

Run: `python -m py_compile src/musubi_tuner/training/full_finetune.py`

Run: `ruff check src/musubi_tuner/training/full_finetune.py tests/test_full_finetune_runtime.py`

Expected: both commands succeed.

- [ ] **Step 8: Commit the lifecycle**

```text
git add src/musubi_tuner/training/full_finetune.py tests/test_full_finetune_runtime.py
git commit -m "feat: add shared full model training lifecycle"
```

---

### Task 3: Resume and Distributed State Semantics

**Files:**
- Modify: `src/musubi_tuner/training/full_finetune.py`
- Create: `tests/test_full_finetune_resume.py`
- Create: `tests/test_full_finetune_distributed.py`

**Interfaces:**
- Produces: deterministic epoch sampler setup and `save_state_all_ranks(accelerator, args, state_dir, retention_callback)`.
- Consumes: `TrainingProgress` and lifecycle from Tasks 1-2.

- [ ] **Step 1: Write an interrupted/resumed equivalence test**

Train the tiny zero-worker fixture for four updates. Compare it with a run that saves after two updates, creates a new
trainer, resumes, and finishes at four. Assert equal model tensors, scheduler state, `global_step`, `epoch`, and
`next_batch`.

- [ ] **Step 2: Implement registered progress and deterministic resume**

Register `TrainingProgress` before `accelerator.load_state`. Seed the dataloader sampler from `args.seed + epoch`; on
resume call `accelerator.skip_first_batches(train_dataloader, progress.next_batch)`. Fail with a progress-specific error
when a resume directory lacks the registered custom checkpoint.

- [ ] **Step 3: Write the two-process DDP and delayed-rank state tests**

Use subprocess workers with the Gloo backend. The first test performs one optimizer update and writes rank weights; both
ranks must match. The second delays rank 1 before completing its state write and records events; main-rank retention must
occur after both `save_state` events and both workers must pass the second barrier before continuing.

- [ ] **Step 4: Implement one state-save helper used by step, epoch, and final paths**

```python
accelerator.save_state(state_dir)
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    retention_callback()
accelerator.wait_for_everyone()
```

Do not duplicate this sequence in three branches.

- [ ] **Step 5: Run resume and distributed tests**

Run: `python -m pytest tests/test_full_finetune_resume.py tests/test_full_finetune_distributed.py -q`

Expected: all supported-platform tests pass; a platform without Gloo skips only the two distributed tests with an
explicit reason.

- [ ] **Step 6: Commit resume/distributed behavior**

```text
git add src/musubi_tuner/training/full_finetune.py tests/test_full_finetune_resume.py tests/test_full_finetune_distributed.py
git commit -m "feat: preserve full finetune resume progress"
```

---

### Task 4: FLUX.1 Kontext Full Finetuning

**Files:**
- Modify: `src/musubi_tuner/flux_kontext_train_network.py`
- Create: `src/musubi_tuner/flux_kontext_train.py`
- Create: `flux_kontext_train.py`
- Create: `tests/test_flux_kontext_full_finetune.py`
- Modify: `docs/flux_kontext.md`

**Interfaces:**
- Consumes: `FullFineTuningTrainerMixin`, `add_full_finetune_args`.
- Produces: `FluxKontextTrainer`, `main`, fp32-safe sampling.

- [ ] **Step 1: Write failing dtype and entrypoint tests**

Spy on `flux_utils.load_flow_model` and assert `dtype=torch.float32` for default full training. Run sample inference with
fp32 fake parameters and `mixed_precision=no`; assert every floating model input is fp32. Assert the root shim text is
exactly the established import-and-main pattern.

- [ ] **Step 2: Make network sampling use `dit_dtype`**

Replace hard-coded bf16 casts for noise, control latents, T5 features, and CLIP pooler with the passed `dit_dtype`.
Existing LoRA calls still pass bf16/fp16 and therefore retain their current dtype.

- [ ] **Step 3: Add the thin full trainer**

`FluxKontextTrainer(FullFineTuningTrainerMixin, FluxKontextNetworkTrainer)` overrides
`load_full_finetune_transformer` and calls `flux_utils.load_flow_model(..., dtype=trainable_dtype,
fp8_scaled=False)`. Compose `setup_parser_common`, `flux_kontext_setup_parser`, and `add_full_finetune_args` in `main`.

- [ ] **Step 4: Add documentation and run tests**

Document cache prerequisites, full-DiT scope, recommended Adafactor/gradient checkpointing, fp32/full-bf16 behavior,
single-process block swap, and unsupported FP8/H2D-only combinations in both existing language sections.

Run: `python -m pytest tests/test_flux_kontext_full_finetune.py -q`

Run: `python flux_kontext_train.py --help`

Expected: both succeed.

- [ ] **Step 5: Commit Kontext support**

```text
git add src/musubi_tuner/flux_kontext_train_network.py src/musubi_tuner/flux_kontext_train.py flux_kontext_train.py tests/test_flux_kontext_full_finetune.py docs/flux_kontext.md
git commit -m "feat: add FLUX Kontext full finetuning"
```

---

### Task 5: FLUX.2 Full Finetuning

**Files:**
- Modify: `src/musubi_tuner/flux_2_train_network.py`
- Create: `src/musubi_tuner/flux_2_train.py`
- Create: `flux_2_train.py`
- Create: `tests/test_flux_2_full_finetune.py`
- Modify: `docs/flux_2.md`

**Interfaces:**
- Consumes: shared full lifecycle.
- Produces: `Flux2Trainer`, explicit version metadata, fp32-safe sampling.

- [ ] **Step 1: Write failing version/dtype tests**

Parameterize all keys in `FLUX2_MODEL_INFO`; assert each reaches `load_flow_model` with the requested trainable dtype.
Assert fp32 sample latents, context, and control tokens with `mixed_precision=no`. Assert metadata records the selected
version but parser behavior still requires users to pass the same version on reload.

- [ ] **Step 2: Replace hard-coded sample bf16 with `dit_dtype`**

Cast context, negative context, initial latents, and packed control tensors to the passed model dtype while keeping VAE
operations in `vae.dtype`.

- [ ] **Step 3: Add trainer, shim, and metadata**

`Flux2Trainer` uses the selected `Flux2ModelInfo`, passes `trainable_dtype` directly into `load_flow_model`, and returns a
string metadata key for the selected version from `full_finetune_metadata`.

- [ ] **Step 4: Document and verify FLUX.2**

State explicitly that exported Klein checkpoints still need matching `--model_version klein-4b` or `klein-9b` during
reload. Include full commands and limitations in both language sections.

Run: `python -m pytest tests/test_flux_2_full_finetune.py -q`

Run: `python flux_2_train.py --help`

Expected: both succeed.

- [ ] **Step 5: Commit FLUX.2 support**

```text
git add src/musubi_tuner/flux_2_train_network.py src/musubi_tuner/flux_2_train.py flux_2_train.py tests/test_flux_2_full_finetune.py docs/flux_2.md
git commit -m "feat: add FLUX 2 full finetuning"
```

---

### Task 6: Ideogram 4 Full Finetuning

**Files:**
- Modify: `src/musubi_tuner/ideogram4_train_network.py`
- Modify: `src/musubi_tuner/ideogram4/sampling_policy.py`
- Create: `src/musubi_tuner/ideogram4_train.py`
- Create: `ideogram4_train.py`
- Create: `tests/test_ideogram4_full_finetune.py`
- Modify: `docs/ideogram4.md`

**Interfaces:**
- Consumes: shared full lifecycle and Ideogram header helpers.
- Produces: `Ideogram4Trainer`, direct dtype loading, conditional metadata, instance sampling policy.

- [ ] **Step 1: Write failing header, dtype, wrapper, and policy tests**

Assert BNB4 and FP8 key headers fail before model construction. Spy on `load_ideogram4_transformer` and prove fp32 is
passed directly. Wrap a fake transformer without `.config`; assert config is read from `accelerator.unwrap_model` while
forward uses the wrapper. Assert LoRA policy remains opt-in and full policy uses any supplied unconditional path.

- [ ] **Step 2: Generalize the instance sampling policy**

Add `use_unconditional_dit_for_sampling(self, args)` to `Ideogram4NetworkTrainer`, returning the existing LoRA helper.
Route prompt encoding and `on_before_sample_images` through this method. The full subclass overrides it with
`bool(args.unconditional_dit)`.

- [ ] **Step 3: Pass loader dtype and separate config from forward model**

Use `dit_weight_dtype` in `load_transformer` rather than discarding it. In `call_dit`, read config from the unwrapped
model and invoke the prepared `transformer(...)` for prediction.

- [ ] **Step 4: Add the full trainer and native metadata**

Reject quantized headers and the LoRA-only unconditional flag. Save `model_type=ideogram4_cond`. Keep the unconditional
model frozen, sampling-only, and absent from optimizer/export state.

- [ ] **Step 5: Document, test, and commit**

Run: `python -m pytest tests/test_ideogram4_full_finetune.py tests/test_ideogram4_lora_sampling.py -q`

Run: `python ideogram4_train.py --help`

Expected: both succeed.

```text
git add src/musubi_tuner/ideogram4_train_network.py src/musubi_tuner/ideogram4/sampling_policy.py src/musubi_tuner/ideogram4_train.py ideogram4_train.py tests/test_ideogram4_full_finetune.py docs/ideogram4.md
git commit -m "feat: add Ideogram 4 full finetuning"
```

---

### Task 7: Krea 2 Full Finetuning

**Files:**
- Modify: `src/musubi_tuner/krea2_train_network.py`
- Create: `src/musubi_tuner/krea2_train.py`
- Create: `krea2_train.py`
- Create: `tests/test_krea2_full_finetune.py`
- Modify: `docs/krea2.md`

**Interfaces:**
- Consumes: shared full lifecycle.
- Produces: `Krea2Trainer`, independent primary-model variant, fp32-safe sampling.

- [ ] **Step 1: Write failing variant/dtype/wrapper tests**

Assert default `dit_variant=raw` uses resolution-aware timesteps, `dit_variant=turbo` uses fixed `mu=1.15`, and the full
trainer rejects `turbo_dit`/`turbo_dit_cache`. Use a wrapper without `.config` to prove config reads unwrap while forward
uses the wrapper. Assert fp32 sample embeds/noise with `mixed_precision=no`.

- [ ] **Step 2: Add `--dit_variant` without changing LoRA defaults**

Add parser choices `raw` and `turbo`, default `raw`. Choose Turbo sampling when either the existing LoRA swap path is
active or the primary variant is Turbo. Do not change existing RAW-train/Turbo-sample restoration behavior.

- [ ] **Step 3: Make Krea config and dtype wrapper-safe**

Read `config.patch` from the unwrapped model, call the prepared model for forward, and replace sample bf16 constants with
`dit_dtype`.

- [ ] **Step 4: Add the full trainer, docs, and shim**

Reject temporary Turbo swapping in `validate_full_finetune_model_args`; accept a Turbo primary checkpoint only with
`--dit_variant turbo`. Record the variant in metadata.

- [ ] **Step 5: Verify and commit Krea support**

Run: `python -m pytest tests/test_krea2_full_finetune.py tests/test_krea2_vram_cleanup.py tests/test_krea2_timesteps.py -q`

Run: `python krea2_train.py --help`

Expected: both succeed.

```text
git add src/musubi_tuner/krea2_train_network.py src/musubi_tuner/krea2_train.py krea2_train.py tests/test_krea2_full_finetune.py docs/krea2.md
git commit -m "feat: add Krea 2 full finetuning"
```

---

### Task 8: Entrypoint, Documentation, and Regression Closure

**Files:**
- Modify: `tests/test_top_level_entrypoints.py`
- Modify: `README.md`
- Modify: files changed in Tasks 1-7 only when verification exposes a defect.

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: complete public surface and verified regression status.

- [ ] **Step 1: Extend the exact-shim test**

Add the four full entrypoints to the expected mapping and retain the exact two-line import/main contract.

- [ ] **Step 2: Update README capability text**

State that the four families support LoRA and full-DiT finetuning, while avoiding claims of text encoder/VAE training or
unverified GPU combinations.

- [ ] **Step 3: Run focused full-finetune tests**

Run: `python -m pytest tests/test_full_finetune_core.py tests/test_full_finetune_runtime.py tests/test_full_finetune_resume.py tests/test_full_finetune_distributed.py tests/test_flux_kontext_full_finetune.py tests/test_flux_2_full_finetune.py tests/test_ideogram4_full_finetune.py tests/test_krea2_full_finetune.py tests/test_top_level_entrypoints.py -q`

Expected: all tests pass, with only an explicit Gloo-unavailable skip permitted.

- [ ] **Step 4: Run the complete regression suite**

Run: `python -m pytest -q`

Expected: all repository tests pass.

- [ ] **Step 5: Run static verification**

Run: `ruff check src/musubi_tuner/training/full_finetune.py src/musubi_tuner/flux_kontext_train.py src/musubi_tuner/flux_2_train.py src/musubi_tuner/ideogram4_train.py src/musubi_tuner/krea2_train.py tests`

Run: `python -m py_compile src/musubi_tuner/training/full_finetune.py src/musubi_tuner/flux_kontext_train.py src/musubi_tuner/flux_2_train.py src/musubi_tuner/ideogram4_train.py src/musubi_tuner/krea2_train.py`

Run each root `*_train.py --help` command.

Expected: every command exits zero.

- [ ] **Step 6: Inspect the final diff and commit closure**

Run: `git diff --check upstream/main...HEAD`

Run: `git status --short`

Expected: no whitespace errors and no untracked generated artifacts.

```text
git add README.md tests/test_top_level_entrypoints.py
git commit -m "docs: document image full finetuning"
```

The final report must distinguish automated CPU/static verification from unavailable multi-billion-parameter CUDA
smoke tests. Do not claim real-model end-to-end validation unless those checkpoints and suitable hardware were used.
