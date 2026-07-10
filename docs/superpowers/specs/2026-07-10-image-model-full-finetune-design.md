# Image Model Full Finetune Design

## Status

- Date: 2026-07-10
- Baseline: `upstream/main` at `30c658c4f4b0bf05038b3346eff9670259b10fc7`
- Working branch: local `finetune`, created from that baseline without an upstream tracking branch
- Decision: approved scope A, meaning full DiT/Transformer finetuning while the text encoders and VAE remain frozen

## Problem

Musubi Tuner has image generation, cache, and LoRA training entrypoints for seven image model families. Qwen-Image,
Z-Image, and HiDream-O1 already have dedicated full-finetune entrypoints. The following four families expose
`*_generate_image.py` and `*_train_network.py`, but no corresponding `*_train.py`:

| Model family | Existing LoRA entrypoint | Full-finetune entrypoint to add |
| --- | --- | --- |
| FLUX.1 Kontext | `flux_kontext_train_network.py` | `flux_kontext_train.py` |
| FLUX.2 dev/Klein | `flux_2_train_network.py` | `flux_2_train.py` |
| Ideogram 4 | `ideogram4_train_network.py` | `ideogram4_train.py` |
| Krea 2 | `krea2_train_network.py` | `krea2_train.py` |

For this design, "full finetune" means that every active parameter in the model's DiT/Transformer is optimized.
It does not mean training the text encoder, VAE, or an auxiliary unconditional model.

## Goals

1. Add a supported full-DiT training entrypoint for all four missing image model families.
2. Reuse each model's existing dataset, timestep, forward, loss, sampling, loading, block-swap, and compile behavior.
3. Keep frozen components out of the optimizer and avoid loading them during training unless sample generation needs them.
4. Save a standalone safetensors DiT checkpoint that the corresponding Musubi inference/training loader can read directly.
5. Reject LoRA-only, frozen-base-only, and quantized-base combinations before allocating the full model.
6. Cover shared behavior with CPU-sized automated tests and document the remaining real-model GPU smoke tests.

## Non-goals

- Text encoder finetuning.
- VAE finetuning.
- Joint training of Ideogram 4 conditional and unconditional DiTs.
- FP8, BNB 4-bit, or other quantized trainable DiT weights.
- Krea 2 RAW-train/Turbo-sample weight swapping during full finetuning.
- Partial-block, layer-selection, freeze-pattern, or parameter-group presets.
- Migrating `qwen_image_train.py`, `zimage_train.py`, or `hidream_o1_train.py` to the new shared implementation.
- Changing existing LoRA behavior or turning `NetworkTrainer` into a public stable API.
- GUI integration.

## Considered Approaches

### 1. Copy an existing full-finetune script four times

This follows the current Qwen-Image and Z-Image pattern with the smallest conceptual change. It would also add roughly
four copies of the dataset setup, optimizer setup, training loop, checkpoint lifecycle, logging, and resume logic. Bug
fixes would then have to land in seven independent full-finetune loops. This is low initial design cost but high ongoing
maintenance cost.

### 2. Add a shared full-finetune engine and four thin model entrypoints

The shared engine owns one complete full-model lifecycle: dataset setup, a transformer optimizer, Accelerate
preparation, the full-model training loop, sampling, state progress, and checkpoint export. It is deliberately separate
from the existing network/LoRA loop, whose control flow assumes a distinct network object. The thin entrypoints inherit
the existing `*NetworkTrainer` classes only for architecture operations with compatible contracts. Existing LoRA and
existing full-finetune scripts stay unchanged. This creates one new full loop rather than four copies, without changing
the training target of every existing architecture.

### 3. Generalize `NetworkTrainer` into LoRA and full-model modes

This could eventually produce the least total code, but `NetworkTrainer` currently assumes a separate network object in
optimizer construction, Accelerate preparation, distributed gradient reduction, epoch/step hooks, checkpoint saving,
and max-norm handling. Making all of those paths polymorphic would affect every architecture. That refactor is outside
the requested feature and is not justified as a prerequisite.

## Architecture

### Shared engine

Add `src/musubi_tuner/training/full_finetune.py` with a `FullFineTuningTrainerMixin`. A model-specific trainer uses
multiple inheritance with the mixin first so its `train` method wins while all architecture hooks continue to come from
the existing network trainer:

```python
class Flux2Trainer(FullFineTuningTrainerMixin, Flux2NetworkTrainer):
    pass
```

The mixin is repository-internal. It relies on the following existing `NetworkTrainer` contract:

- `architecture` and `architecture_full_name`
- `handle_model_specific_args`
- `process_sample_prompts` and `sample_images`
- `load_vae` and `load_transformer`
- `compile_transformer`
- `scale_shift_latents`
- `process_batch`, which delegates to `call_dit` and `compute_loss`
- `on_transformer_loaded`, `on_before_sample_images`, and `on_after_sample_images`
- optimizer, scheduler, logging, timestep, and resume helpers inherited from `NetworkTrainer`

The mixin also defines narrow full-finetune hooks:

- `validate_full_finetune_model_args(args)`, a default no-op used for format and model-specific incompatibilities;
- `load_full_finetune_transformer(...)`, which loads plain weights directly in the requested trainable dtype instead of
  relying on a lossy post-load cast;
- `full_finetune_metadata(args)`, a default empty mapping used for loader-required checkpoint metadata.

Architecture hooks used by the full engine obey two additional contracts:

- sampling tensors use the passed `dit_dtype` rather than a hard-coded bf16 dtype;
- configuration reads unwrap a prepared model, while the prediction still calls the prepared model.

The shared engine must not branch on architecture names. FLUX.2 model-version metadata, Ideogram checkpoint validation,
and Krea Turbo restrictions live in the thin model trainer through these hooks. This keeps the shared control flow the
same for all four models and makes model ownership explicit.

Sampling hooks receive `network=None`; the four covered trainers must not dereference the network argument on the
full-finetune path. The shared engine wraps the before/after hooks in `try/finally`. Its finalizer restores RNG state,
ordinary block-swap training mode, and the transformer's previous train/eval mode even when prompt processing or model
inference raises; calling the architecture's after hook alone is not sufficient because the current shared sampler does
not restore all of those states on every exceptional exit.

The shared engine must not import a LoRA module, call `network.apply_to`, construct a network object, or call network
methods such as `on_step_start`, `save_weights`, or max-norm regularization. It is an independent full-model loop, not a
claim that the current network loop can be reused unchanged.

### Raw and prepared model ownership

After Accelerate preparation, the engine keeps an explicit model pair:

- `forward_model`: the object returned by `accelerator.prepare`; all training forwards and
  `accelerator.accumulate(...)` calls use it so DDP hooks remain active;
- `raw_model`: `accelerator.unwrap_model(forward_model, keep_fp32_wrapper=False)`; configuration reads, block-swap
  control, checkpoint extraction, and state-dict normalization use it.

The thin trainers never call the raw model for a training forward. Krea 2 and Ideogram 4 may read `config` from the
unwrapped object, but call `forward_model(...)` for the prediction. `compile_transformer` runs on the raw model before
Accelerate/DDP wrapping; Accelerate then wraps the compiled model once.

### Training lifecycle

The engine performs these steps in order:

1. Validate common arguments, cheap checkpoint headers, and model-specific arguments before allocating model weights.
2. Set CUDA TF32/cuDNN flags, resolve the seed, prepare Accelerate, and resolve the trainable parameter dtype:
   - fp32 parameters by default;
   - bf16 parameters only with `--full_bf16`, which also requires `--mixed_precision bf16`;
   - full fp16 parameters are unsupported.
3. Build the cached training dataset using the model architecture key and a deterministic epoch-seeded sampler.
4. Encode sample prompts and load a frozen VAE only when `--sample_prompts` is present.
5. Use `load_full_finetune_transformer` to load plain weights directly in the selected trainable dtype, run
   `on_transformer_loaded`, and set all intended DiT parameters to `requires_grad=True`. A post-load cast is only a
   defensive assertion path; it is not allowed to hide a loader that first discarded fp32 precision.
6. Enable ordinary backward-capable block swap and gradient checkpointing when requested. Before Accelerate wrapping,
   move only non-swapped modules to the accelerator device. Then run `compile_transformer` on the raw model when
   `--compile` is set.
7. Build one optimizer parameter group from named trainable transformer parameters and fail if the group is empty.
8. Prepare the transformer, optimizer, dataloader, and scheduler with Accelerate, create the raw/forward model pair,
   register training progress for state checkpoints, and only then load `--resume` state. Transformer preparation has
   two explicit paths:
   - without block swap, use ordinary Accelerate device placement;
   - with block swap, prepare the transformer separately with `device_placement=[False]`, then call
     `raw_model.move_to_device_except_swap_blocks(accelerator.device)` and
     `raw_model.prepare_block_swap_before_forward()` after wrapping.
9. For every batch, use `forward_model` with `scale_shift_latents` and `process_batch`, backpropagate the returned scalar
   loss, clip transformer gradients, step the optimizer/scheduler, and log shared and model-specific metrics.
10. Preserve step/epoch sampling, exported-checkpoint retention, tracker behavior, and Hugging Face upload behavior.
11. Save the unwrapped raw transformer at step/epoch boundaries and at the end of training. State checkpoints are a
   separate all-rank operation.

The training engine calls `transformer.train()` for optimization. Sampling may temporarily use eval mode through the
existing sampling implementation, after which the transformer must be returned to train mode.

Accelerate/DDP owns gradient synchronization for the prepared transformer. The shared engine must not copy
`NetworkTrainer`'s manual network-gradient reduction loop; doing both would reduce gradients twice.

### Optimizer and block swap

Full finetuning supports the existing optimizer factory. Adafactor remains the documented recommendation because a
full optimizer state for these models is large.

The new parser extension adds the same full-finetune controls used by existing scripts:

- `--full_bf16`
- `--fused_backward_pass`
- `--mem_eff_save`
- `--block_swap_optimizer_patch_params`

`--fused_backward_pass` is valid only for an optimizer implementing the patched Adafactor `step_param` interface and
only for a single training process. The per-parameter hook clears gradients before ordinary DDP reduction can safely
consume them, so multi-process fused backward is rejected until it has a dedicated distributed implementation and test.
It is also rejected when `--gradient_accumulation_steps` is greater than one, because the current post-accumulate hook
would update and clear each parameter on every microbatch. The script rejects incompatible optimizers, process counts,
or accumulation settings before the first backward pass.

Ordinary `--blocks_to_swap` remains supported with `supports_backward=True`. When an optimizer expects gradients on the
same device as its parameters, `--block_swap_optimizer_patch_params` moves each gradient to its parameter device before
`optimizer.step`; documentation must retain the existing warning that this workaround is intended for AdamW and
Adafactor, not 8-bit or arbitrary third-party optimizers.

Full-model block swap is initially single-process only. `num_processes > 1` together with `--blocks_to_swap` is rejected
because DDP assumes a stable device placement while the offloader migrates trainable parameters between CPU and CUDA.
Multi-process training without block swap remains supported through the raw/forward model boundary and requires a
two-process gradient-synchronization test.

`--block_swap_h2d_only` must be rejected. That mode deliberately keeps an immutable CPU master copy and is correct only
for a frozen base model, so using it for full finetuning would discard or overwrite parameter updates.

### Checkpoint format

Each checkpoint is a standalone safetensors state dict of the unwrapped DiT/Transformer. It contains no LoRA prefix and
no text encoder or VAE weights.

Before saving, compiled-module key segments are normalized generically:

- `._orig_mod.` becomes `.`;
- a leading `_orig_mod.` is removed;
- key collisions after normalization are an error.

This must work for every model rather than checking for one Qwen-specific sentinel key.

The save dtype matches the trainable parameter dtype: fp32 by default and bf16 with `--full_bf16`. An explicit
`--save_precision` is accepted only when it resolves to that same dtype; a different value is rejected in this pass.
This avoids constructing a second full-model cast in memory and keeps `--mem_eff_save` truthful. The repository's
streaming safetensors writer handles model export; Accelerate state saving remains separate and may still require normal
optimizer-state memory.

Saved metadata includes the existing session, dataset, optimizer, timestep, architecture, and SAI model-spec fields,
plus:

- `ss_training_type=full-finetune`
- `ss_full_finetune=True`
- `ss_fp8_base=False`

The SAI metadata call uses `is_lora=False`. Ideogram 4 checkpoints additionally retain
`model_type=ideogram4_cond`, because its loader validates that field before reading tensors. FLUX.2 metadata records the
selected model version. Metadata dictionaries must be copied before SAI fields are merged so saving with
`--no_metadata` cannot mutate the full metadata used by later checkpoints.

### State progress and resume

Accelerate state includes the prepared transformer, optimizer, scheduler, scaler, and RNG state. The full engine also
registers a `TrainingProgress` checkpoint object containing `global_step`, zero-based epoch, and the next batch offset in
that epoch. A deterministic sampler is reseeded from `(seed, epoch)`; after resume the engine restores the epoch and uses
`accelerator.skip_first_batches` to continue at the saved local batch offset without replaying model/RNG work.

One shared `save_state_all_ranks` helper owns step, epoch, and final state saves. Every rank calls
`accelerator.save_state`, all ranks immediately synchronize, the main rank then performs retention/upload, and all ranks
synchronize again before training continues. This prevents the main rank from reading or uploading a state directory
while a delayed rank is still writing RNG or progress files. Only the main rank performs exported safetensors work.
The progress file is required for new full-trainer resume; a state directory without it fails clearly instead of
silently restarting counters from zero. Tests compare uninterrupted training with an interrupted/resumed run using a
deterministic zero-worker dataset. Bitwise equivalence is not promised for arbitrary third-party optimizers or datasets
with untracked worker-side randomness.

A normal exported safetensors checkpoint contains model weights only and is not a replacement for `--save_state` when
optimizer/scheduler continuation is required.

## Common Validation

Validation happens before loading model weights. The full-finetune entrypoints must:

- require `--dataset_config` and `--dit`;
- reject `--fp8_base` and `--fp8_scaled` rather than silently turning them off;
- reject `--block_swap_h2d_only`;
- reject non-default LoRA/network options, including `--network_weights`, `--network_module`, `--network_dim`,
  `--network_alpha` when changed from its default, `--network_dropout`, `--network_args`, `--dim_from_weights`,
  `--scale_weight_norms`, `--base_weights`, and `--base_weights_multiplier`;
- retain common attention-backend validation, including the existing SageAttention training rejection;
- reject `--full_bf16` unless mixed precision is bf16;
- reject fused backward with an unsupported optimizer, multiple processes, or gradient accumulation greater than one;
- reject multiple processes combined with full-model block swap;
- reject an explicit save precision that differs from the trainable parameter dtype or that safetensors cannot represent;
- require the new full-trainer progress file when `--resume` is used.

Error messages name the invalid option and explain the full-finetune alternative. No option in the accepted command line
may be silently ignored.

Text encoder, VAE, and text-encoder precision arguments remain accepted for compatibility with shared config files. If
they are supplied without `--sample_prompts`, the trainer logs that they are unused; it does not silently load frozen
components.

## Model-specific Behavior

### FLUX.1 Kontext

- Train every parameter of the Kontext `Flux` transformer loaded from `--dit`.
- Reuse the existing control-latent, T5, CLIP, guidance, timestep, and flow-matching path.
- Require the same cached latent/text outputs as LoRA training; the reference/control latent remains part of every
  training batch.
- The full loader passes fp32/bf16 directly into `flux_utils.load_flow_model`; it does not reuse the current network
  loader's `dtype=None` behavior.
- Training-time sampling casts noise, control latents, and text conditions to the passed `dit_dtype`, so fp32 parameters
  also work with `--mixed_precision no`.
- Keep `--fp8_t5` available only for the frozen sample-prompt encoder.
- Load T5, CLIP, and the VAE only when training-time sampling requires them.

### FLUX.2

- Support every `--model_version` already present in `FLUX2_MODEL_INFO`, including dev and Klein variants, through the
  same model-version resolution used by LoRA training.
- Train every parameter of the selected `Flux2` transformer.
- Reuse optional control-image tokens, model-version guidance behavior, and `flux2_shift` timestep behavior.
- Load transformer weights directly in the full trainable dtype and cast sampling latents/context to `dit_dtype` rather
  than hard-coded bf16.
- Keep `--fp8_text_encoder` available only for the frozen sample-prompt encoder and retain its existing model-version
  restrictions.
- Store the selected model version in checkpoint metadata as provenance. Reload and inference still require the same
  explicit `--model_version`; the current FLUX.2 loader does not infer architecture from safetensors metadata.

### Ideogram 4

- Train the plain conditional DiT only. The checkpoint passed to `--dit` must be fp32, fp16, or bf16.
- Detect and reject BNB 4-bit and prequantized FP8 conditional checkpoints before constructing a trainable optimizer.
- Pass the selected full trainable dtype directly to `load_ideogram4_transformer`. The full entrypoint must not apply the
  network entrypoint's default bf16 `args.dit_dtype` and then upcast already-truncated weights.
- Reuse Ideogram 4 latent normalization, inverse timestep convention, custom mean-MSE loss, and optional loss metrics.
- Preserve `model_type=ideogram4_cond` in every exported checkpoint.
- A model supplied with `--unconditional_dit` is frozen and used only for asymmetric-CFG sample generation. The full
  trainer uses it automatically; it does not require the LoRA-only `--use_unconditional_dit_for_lora_sampling` flag.
- Passing `--use_unconditional_dit_for_lora_sampling` to the full trainer is an error because that compatibility switch
  has no full-finetune meaning.

The Ideogram trainer must expose an instance sampling-policy hook. The LoRA implementation preserves its current
two-argument opt-in rule; the full trainer override returns true whenever `--unconditional_dit` is supplied. Prompt
encoding and before-sample loading both call that hook. This is a required behavior-neutral refactor, not an optional
follow-up.

### Krea 2

- Train every parameter of the Krea 2 `SingleStreamDiT` loaded from `--dit`.
- Reuse the varlen Qwen3-VL cache batching, Krea timestep path, flow-matching target, Qwen-Image VAE sampling, gradient
  checkpointing, compile, and ordinary block-swap implementations.
- Reject `--turbo_dit` and `--turbo_dit_cache` on the full trainer. The LoRA implementation can restore a static RAW
  base after sampling because that base is frozen; doing the same in full finetuning would overwrite learned updates.
- Add `--dit_variant raw|turbo`, defaulting to `raw`, independently of the LoRA-only temporary `--turbo_dit` path. It
  controls the Krea sampling schedule but never swaps weights.
- Sample from the current trainable checkpoint. A Turbo checkpoint is finetuned by passing it as `--dit` together with
  `--dit_variant turbo`, which selects the fixed Turbo `mu=1.15` schedule.
- Sampling embeds and noise use the passed `dit_dtype`, not a hard-coded bf16 dtype.
- Keep the Qwen3-VL text encoder frozen and load it only for sample-prompt encoding.

## Entrypoints and Files

New production files:

- `src/musubi_tuner/training/full_finetune.py`
- `src/musubi_tuner/flux_kontext_train.py`
- `src/musubi_tuner/flux_2_train.py`
- `src/musubi_tuner/ideogram4_train.py`
- `src/musubi_tuner/krea2_train.py`
- root shims `flux_kontext_train.py`, `flux_2_train.py`, `ideogram4_train.py`, and `krea2_train.py`

Required behavior-neutral changes to existing network trainers are limited to:

- use the passed model dtype during FLUX.1 Kontext, FLUX.2, and Krea 2 sampling;
- unwrap only for Krea 2/Ideogram configuration reads while keeping prepared-model forwards;
- pass the requested loader dtype through Ideogram 4;
- expose the Ideogram unconditional sampling-policy hook;
- separate Krea's primary model variant from its LoRA-only temporary Turbo weight swap.

The existing Qwen-Image, Z-Image, HiDream-O1, and all LoRA entrypoints remain on their current training loops. Tests must
show that the default LoRA behavior of each changed architecture is unchanged.

Documentation changes:

- Add full-finetune commands, cache prerequisites, precision rules, block-swap limitations, and checkpoint semantics to
  `docs/flux_kontext.md`, `docs/flux_2.md`, `docs/ideogram4.md`, and `docs/krea2.md` in both existing language sections.
- Update the README capability description so it no longer describes the repository as LoRA-only for these models.

## Error Handling

- Argument incompatibilities fail before model allocation.
- Empty cached datasets fail with the existing cache-missing guidance.
- An empty trainable parameter set fails before optimizer construction.
- Quantized Ideogram checkpoints fail with a format-specific error.
- Checkpoint key normalization fails on collisions instead of overwriting tensors.
- Sampling cleanup and post-sampling restoration run in `finally` blocks.
- All ranks participate in Accelerate state saves, synchronize before main-rank retention/upload, and synchronize again
  afterward; only the main process writes exported model files, applies retention, or uploads artifacts.
- A failed export is never reported or uploaded as successful.

## Testing Strategy

### Shared CPU tests

Create a tiny fake architecture trainer and transformer to exercise the shared engine without downloading model
weights. Tests cover:

1. The optimizer receives all and only trainable transformer parameters.
2. One synthetic optimization step changes a transformer weight.
3. The VAE remains frozen and is not added to the optimizer.
4. fp32 is the default trainable dtype and `--full_bf16` enforces bf16 mixed precision.
5. LoRA arguments, FP8 base training, H2D-only block swap, multi-process block swap, incompatible fused backward, and
   fused backward with gradient accumulation fail before the loader runs.
6. The single-process ordinary block-swap path requests backward support, disables Accelerate device placement, and
   calls both post-prepare block placement/initialization methods. A block-swap plus compile tiny step covers their
   combined ordering.
7. The exported safetensors file reloads, contains normalized native keys, contains no LoRA/text/VAE keys, uses the
   requested save dtype, and contains full-finetune metadata.
8. Step, epoch, final-save, retention, and `--no_metadata` paths do not share or corrupt metadata dictionaries.
9. A sampling exception still invokes the after-sampling hook and returns the transformer to training mode.
10. A sampling exception restores CPU/CUDA RNG state and switches an unwrapped block-swapped transformer back to its
    training configuration.
11. A wrapper without architecture attributes proves config is read from `raw_model` while forward is invoked through
    `forward_model`.
12. `--compile` calls the model compile hook before Accelerate preparation, completes a tiny training step, and exports
    normalized keys.
13. A two-process CPU/Gloo test compares full-model gradients/weights across ranks and proves there is no second manual
    reduction. It runs without block swap or fused backward.
14. An interrupted/resumed deterministic run matches an uninterrupted run in weights, scheduler state, global step,
    epoch, and next-batch offset.
15. A two-process state-save test delays the non-main rank and proves upload/retention starts only after the first
    post-save barrier, with a second barrier before either rank resumes training.

### Model-specific tests

- Assert all four module entrypoints and root shims exist and import the expected `main` function.
- Parse `--help` and a minimal config for every new entrypoint.
- Monkeypatch each full loader to verify the requested trainable dtype reaches model construction and that all
  transformer parameters become trainable.
- Exercise FLUX.1 Kontext, FLUX.2, and Krea 2 sample-at-first with fp32 parameters and `mixed_precision=no`; no sample
  tensor may remain hard-coded bf16.
- Verify FLUX.2 forwards each supported model version into the existing model-info lookup.
- Verify FLUX.2 metadata is provenance only and docs/validation still require the same explicit model version on reload.
- Verify Ideogram rejects BNB 4-bit/FP8 headers, preserves `model_type=ideogram4_cond`, and automatically uses a supplied
  frozen unconditional DiT only for sampling. A loader-spy test proves fp32 is passed into the real loader rather than
  produced by upcasting bf16 parameters afterward.
- Verify Krea 2 rejects Turbo swapping, selects the RAW schedule by default, and selects fixed `mu=1.15` when a Turbo
  primary checkpoint is paired with `--dit_variant turbo`.
- Keep the existing LoRA tests green, including Ideogram sampling-policy and Krea 2 sample-weight swap tests.

### Verification commands

The implementation is not complete until these pass:

```text
python -m pytest -q
ruff check src/musubi_tuner/training/full_finetune.py \
  src/musubi_tuner/flux_kontext_train.py src/musubi_tuner/flux_2_train.py \
  src/musubi_tuner/ideogram4_train.py src/musubi_tuner/krea2_train.py tests
python -m py_compile src/musubi_tuner/training/full_finetune.py \
  src/musubi_tuner/flux_kontext_train.py src/musubi_tuner/flux_2_train.py \
  src/musubi_tuner/ideogram4_train.py src/musubi_tuner/krea2_train.py
python flux_kontext_train.py --help
python flux_2_train.py --help
python ideogram4_train.py --help
python krea2_train.py --help
```

### Real-model smoke tests

CPU synthetic tests cannot establish that multi-billion-parameter kernels, block swapping, or checkpoint compatibility
work on real hardware. Before an upstream release, run one optimizer step and one exported-checkpoint inference for each
family on a suitable CUDA machine. For FLUX.2, cover at least one dev and one Klein model. For Ideogram 4, cover both
single-DiT sampling and optional frozen-unconditional sampling. For Krea 2, confirm that a saved full checkpoint reloads
as the primary `--dit` model.

If the development machine cannot run these models, the final implementation report must say so explicitly rather than
describing static or synthetic coverage as an end-to-end GPU validation.

## Acceptance Criteria

1. All four new `*_train.py` entrypoints expose full-DiT training and require no LoRA module argument.
2. Text encoders, VAEs, and Ideogram's optional unconditional DiT never enter the optimizer or exported checkpoint.
3. Plain fp32/bf16 training works through the shared engine; quantized trainable bases and H2D-only block swap fail
   early with actionable messages.
4. Single-process backward-capable block swap, gradient checkpointing, compile, sampling, progress-aware Accelerate
   state resume, multi-process training without block swap, and full checkpoint export meet their stated restrictions.
5. Exported files use native model keys, reload through the corresponding loader, and identify themselves as full
   finetunes in metadata.
6. Krea 2 cannot accidentally restore stale RAW weights over learned parameters.
7. Existing LoRA and existing full-finetune entrypoints retain their current behavior and all automated tests pass.
8. The four architecture documents and README accurately distinguish LoRA from full finetuning and state the tested
   precision/memory limitations.
