> 📝 Click on the language section to expand / 言語をクリックして展開

# Kandinsky 6

## Overview / 概要

This is an unofficial LoRA and full fine-tuning trainer and generator for [Kandinsky 6](https://github.com/kandinskylab/kandinsky-6). It supports the released regular Lite and Pro checkpoints, joint text-to-video-with-audio (`t2av`), first-frame-conditioned video-with-audio (`ti2av`), multi-reference video-with-audio (`r2av`), and image datasets for appearance or identity LoRAs. R2AV adds reference embeddings that need full fine-tuning before they can be used with LoRA.

Distilled checkpoints are available for inference, but training them is rejected because they use a different PiFlow objective. Train with a regular Lite or Pro checkpoint.

<details><summary>日本語</summary>

[Kandinsky 6](https://github.com/kandinskylab/kandinsky-6) 向けの非公式LoRA学習・生成機能です。通常版Lite／Pro、音声付き動画の `t2av`、先頭フレーム条件付き `ti2av`、外見・人物同一性LoRA用の画像データセットに対応します。

distilled版は推論できますが、別のPiFlow目的関数を使うため学習は拒否されます。学習には通常版LiteまたはProを使用してください。
</details>

## Installation / インストール

Follow the [main installation guide](../README.md#installation), then run:

```bash
pip install -e ".[kandinsky6]"
```

The runtime is bundled; no separate Kandinsky source checkout is needed. ConvRot INT8 and `--adafactor_triton` additionally require Triton (`triton` on Linux or a compatible `triton-windows` package on Windows).

<details><summary>日本語</summary>

[メインのインストールガイド](../README.ja.md#インストール)に従い、上のコマンドで依存関係を追加します。ランタイムは同梱済みです。ConvRot INT8には追加でTritonが必要です（Linuxは `triton`、Windowsは互換性のある `triton-windows`）。
</details>

## Model download / モデルのダウンロード

Download a regular checkpoint from the official [Kandinsky 6.0 Diffusers collection](https://huggingface.co/collections/kandinskylab/kandinsky-60-diffusers):

- [Lite](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers)
- [Pro](https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers)

```bash
hf auth login
hf download kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers --local-dir /models/k6-lite
hf download kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers --local-dir /models/k6-pro
```

Sign in to Hugging Face and accept the model repository's access conditions before downloading. Choose another absolute local directory on Windows or when `/models` is unavailable, then substitute that directory in every example.

Each snapshot contains `transformer/diffusion_pytorch_model.safetensors` plus `vae`, `audio_vae`, `text_encoder`, `text_encoder_2`, and `vocoder`. Examples use `/models/k6-lite`; substitute the Pro directory when needed. `--model_variant auto` detects the DiT geometry, while `lite` or `pro` enforces it.

<details><summary>日本語</summary>

Hugging Faceへログインし、モデルリポジトリの利用条件に同意してから、公式コレクションの通常版LiteまたはProをダウンロードします。snapshotにはDiTとVAE、audio VAE、2つのtext encoder、vocoderが含まれます。Windowsなどで `/models` を使用できない場合は別の絶対パスを選び、以下の例でも同じパスを指定してください。`--model_variant auto` はDiT構成を自動判定し、`lite`／`pro` は構成を固定して不一致を検出します。
</details>

## Dataset / データセット

Kandinsky 6 uses 24 fps and `4n+1` frame counts; 121 frames are about five seconds. Audio is selected from JSONL `audio_path`, a same-stem sidecar, then the video container, and decoded as 44.1 kHz mono. A video without decodable audio is rejected.

```toml
[general]
resolution = [864, 480]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
video_directory = "/data/clips"
cache_directory = "/data/k6-cache"
target_frames = [121]
frame_extraction = "head"
```

JSONL supports explicit audio and is required for an explicit `ti2av` control image:

```json
{"video_path":"clips/boat.mp4","audio_path":"audio/boat.wav","caption":"A boat crosses a lake."}
{"video_path":"clips/forest.mp4","control_path":"starts/forest.png","caption":"Wind moves through a forest."}
```

```toml
[[datasets]]
video_jsonl_file = "/data/train.jsonl"
cache_directory = "/data/k6-cache"
target_frames = [121]
frame_extraction = "head"
```

Every `ti2av` item must contain one still image in `control_path`. Image datasets are cached as one-frame videos with masked silence, so no audio sidecar is needed:

```toml
[general]
resolution = [512, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
image_directory = "/data/images"
cache_directory = "/data/k6-image-cache"
num_repeats = 10
```

See [Dataset Configuration](dataset_config.md) for captions, bucketing, cropping, and path rules.

<details><summary>日本語</summary>

Kandinsky 6は24 fpsと `4n+1` フレームを使用します（121フレームは約5秒）。音声はJSONLの `audio_path`、同名sidecar、動画内音声の順に選ばれ、44.1 kHz monoで読み込まれます。音声をデコードできない動画はエラーです。

上のTOMLは動画ディレクトリ、JSONLは明示的な音声および制御画像の例です。`ti2av` の全項目に `control_path` の静止画が1枚必要です。画像データセットは1フレーム動画とloss対象外の無音としてキャッシュされるため、音声sidecarは不要です。詳細は[Dataset Configuration](dataset_config.md)を参照してください。
</details>

## Pre-caching / 事前キャッシュ

Cache video/audio latents, then both text encoders:

```bash
python kandinsky6_cache_latents.py \
  --dataset_config dataset.toml \
  --vae /models/k6-lite/vae \
  --audio_vae /models/k6-lite/audio_vae \
  --audio_vae_scaling_factor 0.5302

python kandinsky6_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder_qwen /models/k6-lite/text_encoder \
  --text_encoder_clip /models/k6-lite/text_encoder_2 \
  --max_length 1024
```

Regular Lite and Pro use audio VAE scale `0.5302`. Rebuild caches after changing media, captions, crop geometry, controls, the scale, or encoders. `--skip_broken` skips invalid media. `--skip_existing` checks only that a cache file exists.

<details><summary>日本語</summary>

上の順番で動画・音声latentとtext encoder出力をキャッシュします。通常版Lite／Proのaudio VAE scaleは `0.5302` です。素材、caption、crop、control、scale、encoderを変更したら再作成してください。`--skip_broken` は不正素材をskipします。`--skip_existing` はcache fileの存在だけを確認します。
</details>

## Training and resume / 学習と再開

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 kandinsky6_train_network.py \
  --dataset_config dataset.toml \
  --dit /models/k6-lite/transformer/diffusion_pytorch_model.safetensors \
  --task t2av --model_variant auto \
  --network_module networks.lora_kandinsky6 \
  --network_dim 16 --network_alpha 16 \
  --learning_rate 1e-4 --optimizer_type AdamW \
  --gradient_checkpointing --mixed_precision bf16 \
  --max_train_steps 1000 \
  --output_dir output --output_name k6-lora
```

Use `--task ti2av` only with control-image caches. Video and audio diffusion times are sampled independently by default, with the same `--scheduler_scale` shifted-uniform schedule applied to each modality. Use `--no-independent_time` for synchronous training with a shared diffusion time. When resuming an existing run, select the same timing mode used by that run. `--audio_loss_weight` scales audio loss; `--video_only` disables it. To save and resume complete state:

```bash
# add to training
--save_every_n_steps 250 --save_state

# add when resuming
--resume output/k6-lora-state --max_train_steps 500
```

`--max_train_steps` is the **additional** step budget after resume, and the displayed counter restarts at zero. Resume from the state directory, not a LoRA file. `--base_weights` starts a new run after merging a LoRA into BF16; it does not restore optimizer state and cannot be combined with `--convrot_int8` or `--fp8_base`.

<details><summary>日本語</summary>

上のコマンドで学習します。control画像cacheでのみ `--task ti2av` を使います。動画と音声の拡散時刻はデフォルトで独立にサンプリングされ、それぞれに同じ `--scheduler_scale` のshifted-uniform scheduleが適用されます。共通の拡散時刻で同期学習する場合は `--no-independent_time` を指定します。既存の学習を再開する際は、その学習と同じ時刻設定を選択してください。`--audio_loss_weight` は音声lossを調整し、`--video_only` は無効化します。完全なstateは `--save_state` で保存し、LoRA fileではなくstate directoryから再開します。

再開時の `--max_train_steps` は**追加step数**で、表示counterは0から始まります。`--base_weights` はLoRAをBF16へmergeした新規runであり、optimizer状態は復元せず、INT8／FP8とは併用できません。
</details>

## Memory optimization / メモリ最適化

- `--gradient_checkpointing` recomputes activations; `--gradient_checkpointing_cpu_offload` also moves saved activations to CPU.
- `--blocks_to_swap N` swaps frozen blocks. The maximum is 30 for Lite and 58 for Pro.
- `--block_swap_h2d_only` avoids redundant D2H copies of frozen weights.
- `--use_pinned_memory_for_block_swap` speeds transfers but consumes pinned host memory; on Windows it can also consume shared GPU memory.
- `--convrot_int8` and `--fp8_base` are mutually exclusive.

```bash
--gradient_checkpointing \
--blocks_to_swap 30 \
--block_swap_h2d_only \
--use_pinned_memory_for_block_swap
```

Use 58 blocks for maximum Pro swapping. H2D-only still requires host RAM.

<details><summary>日本語</summary>

`--gradient_checkpointing` はactivationを再計算し、CPU offload optionは保存activationもCPUへ移します。block swap上限はLite 30、Pro 58です。H2D-onlyは不要なD2Hコピーを省きますがhost RAMは必要です。pinned memoryは転送を速めますが、Windowsではshared GPU memoryも消費する場合があります。INT8とFP8は併用できません。
</details>

## ConvRot INT8 prequantization / ConvRot INT8事前量子化

`--convrot_int8` quantizes supported frozen DiT linear weights while loading. `--convrot_int8_bwd bf16` keeps activation-gradient matmul in BF16; `int8` quantizes it too. Export once to avoid repeating quantization:

```bash
python kandinsky6_prequantize.py \
  /models/k6-lite/transformer/diffusion_pytorch_model.safetensors \
  /models/k6-lite/transformer/k6-lite-convrot-int8.safetensors \
  --quant-device cuda
```

Use the exported file as `--dit`. The trainer automatically detects ComfyUI INT8 metadata; `--convrot_int8` may also be supplied explicitly:

```bash
--dit /models/k6-lite/transformer/k6-lite-convrot-int8.safetensors --convrot_int8
```

Only regular Lite/Pro weights are accepted. Existing output is protected unless `--overwrite` is passed.

Prequantized Lite or Pro DiTs in the ComfyUI `int8_tensorwise` format are supported. Each quantized Linear layer must provide an INT8 `.weight`, an FP32 `.weight_scale`, and a `.comfy_quant` metadata tensor. Checkpoints can contain both rotated ConvRot and plain per-channel INT8 layers. Their supplied weights and scales are preserved; plain layers do not apply Hadamard rotation. LoRA weights remain floating point.

Pass the checkpoint file to `--dit` in the training command:

```bash
--dit /models/k6/transformer-convrot-int8.safetensors
```

Use the matching original Lite or Pro snapshot for VAEs, text encoders, and vocoder if these components are not included with the quantized DiT. `--fp8_base` and `--base_weights` cannot be combined with a prequantized INT8 base.

<details><summary>日本語</summary>

`--convrot_int8` は読み込み時に対応Linear weightを量子化します。backward精度は `--convrot_int8_bwd bf16` または `int8` で選択します。上のexportを一度実行し、学習では出力ファイルを `--dit` に指定します。ComfyUIのINT8メタデータは自動検出され、`--convrot_int8` を明示することもできます。通常版Lite／Pro専用で、既存出力の置換には `--overwrite` が必要です。

ComfyUIの `int8_tensorwise` 形式で事前量子化されたLite／ProのDiTに対応します。各量子化Linear層にはINT8の `.weight`、FP32の `.weight_scale`、および `.comfy_quant` メタデータが必要です。ConvRot回転済み層と通常のチャネル単位INT8層を混在させることができます。元の重みとスケールは保持され、通常のINT8層にはHadamard回転を適用しません。LoRAは浮動小数点のまま学習します。

上の例のようにチェックポイントファイルを学習時の `--dit` に指定してください。VAE、テキストエンコーダー、vocoderが同梱されていない場合は、対応する通常版Lite／Proから取得します。事前量子化INT8モデルに `--fp8_base` や `--base_weights` は併用できません。
</details>

## Standalone generation / 単体生成

The generator has no `--dit` option. Bundled presets resolve the official components. Base generation with sound:

```bash
python kandinsky6_generate_video.py \
  --config lite \
  --prompt "A boat crosses a lake, with soft splashing water." \
  --audio --offload module --attention_engine sdpa \
  --seed 42 --output output/boat.mp4
```

LoRA generation:

```bash
python kandinsky6_generate_video.py \
  --config lite \
  --prompt "A boat crosses a lake." \
  --lora_weight output/k6-lora.safetensors \
  --lora_multiplier 0.8 \
  --audio --seed 42 --output output/boat-lora.mp4
```

Standalone generation and training previews share the official negative prompt by default. Override it with `--negative_prompt` for generation or the `negative_prompt` field in sample prompts; an explicit empty string disables the negative prompt.

Use `--no-audio` for silence. For TI2AV add `--image starts/boat.png` and a compatible `--visual_cond_scheme`. Multiple LoRAs and matching multipliers may be listed.

For fully local generation, copy the complete bundled `lite.yaml` or `pro.yaml` from `src/musubi_tuner/kandinsky6/runtime/configs/checkpoints/`. Keep every section and replace all six component locations with absolute paths:

```yaml
paths:
  dit: /models/k6-lite/transformer/k6-lite-convrot-int8.safetensors
  vae: /models/k6-lite/vae
  qwen: /models/k6-lite/text_encoder
  clip: /models/k6-lite/text_encoder_2
audio_vae:
  tod_vae_ckpt: /models/k6-lite/audio_vae
  scaling_factor: 0.5302
vocoder:
  ckpt: /models/k6-lite/vocoder
```

A YAML containing `paths` is a full checkpoint mapping, so a partial override is insufficient. Keep `checkpoint`, `dit`, `generation`, `text_embedder`, `scheduler`, and `piflow` from the copied preset. Run the exported INT8 DiT with:

```bash
python kandinsky6_generate_video.py \
  --config /models/k6-lite/local-convrot.yaml \
  --prompt "A boat crosses a lake." \
  --convrot_int8 --audio \
  --output output/boat-int8.mp4
```

LoRAs remain floating adapters over INT8. ConvRot/LoRA generation rejects AOTI configs using `paths.dit_export`.

<details><summary>日本語</summary>

単体生成と学習中のプレビューは、デフォルトで同じ公式のネガティブプロンプトを使用します。生成時の `--negative_prompt`、またはサンプル設定の `negative_prompt` フィールドで上書きできます。空文字列を明示するとネガティブプロンプトを無効にします。

generatorには `--dit` がありません。通常のbase／LoRA生成は上のコマンドを使います。無音は `--no-audio`、TI2AVは `--image` と対応する `--visual_cond_scheme` を追加します。

完全なローカル生成では同梱 `lite.yaml`／`pro.yaml` を**全体ごと**コピーし、上記6箇所を絶対パスにします。`paths` を含むYAMLは完全な設定として扱われるため、部分的な上書きはできません。事前量子化DiTは `paths.dit` に指定し、生成時にも `--convrot_int8` を付けます。LoRAはINT8上の浮動小数点adapterです。`paths.dit_export` のAOTI設定では利用できません。
</details>

## Sampling during training / 学習中のサンプル生成

Create `sample_prompts.txt` using the standard prompt syntax. `--f` is the pixel-frame count and must follow `4n+1`; `--i` supplies the first frame for TI2AV:

```text
A boat crosses a lake, with soft splashing water. --w 864 --h 480 --f 121 --d 42 --s 30 --g 5.0
A forest comes alive in the wind. --w 864 --h 480 --f 121 --d 7 --s 30 --g 5.0 --i starts/forest.png
```

Add the following to the training command:

```bash
--sample_prompts sample_prompts.txt \
--sample_every_n_steps 250 \
--vae /models/k6-lite/vae \
--audio_vae /models/k6-lite/audio_vae \
--vocoder /models/k6-lite/vocoder \
--text_encoder_qwen /models/k6-lite/text_encoder \
--text_encoder_clip /models/k6-lite/text_encoder_2
```

`--sample_at_first` and `--sample_every_n_epochs` are alternatives for scheduling. See [Sampling during training](sampling_during_training.md). `--quantized_qwen` works for text caching but is rejected for in-training sampling because NF4 weights are device-bound.

<details><summary>日本語</summary>

上の形式で `sample_prompts.txt` を作成します。`--f` は出力フレーム数で `4n+1` にし、TI2AVでは `--i` で先頭画像を指定します。学習コマンドには例のsampling option、VAE、audio VAE、vocoder、両text encoderを追加します。実行時期は `--sample_at_first` や `--sample_every_n_epochs` でも指定できます。詳細は[Sampling during training](sampling_during_training.md)を参照してください。`--quantized_qwen` はtext cacheでは利用できますが、device固定のため学習中samplingでは利用できません。
</details>

## Full fine-tuning

`kandinsky6_train.py` trains the transformer weights directly in BF16 for Lite or Pro.
The video/audio VAEs and text encoders stay frozen. The defaults enable gradient
checkpointing and fused Adafactor updates: each tensor is updated during backward,
then its gradient is freed. Use a single process/GPU and `--max_grad_norm 0`.

Add `--flash_attn` for FlashAttention 2 or `--flash3` for FlashAttention 3 on the
unmasked visual, audio, and cross-attention operations in full fine-tuning or LoRA
training. Select at most one. FlashAttention 3 requires its optional Hopper package,
an H100/H800 GPU, and CUDA 12.3 or newer (upstream recommends CUDA 12.8). It imports the current `flash_attn_3`
namespace, with compatibility for older top-level `flash_attn_interface` installs.
Missing packages fail at startup with an actionable error. Text attention that
carries a padding mask continues to use PyTorch SDPA so padding semantics are
preserved. Omit both flags to keep the default SDPA path.

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 kandinsky6_train.py \
  --dataset_config dataset.toml --dit /models/k6-pro/transformer/diffusion_pytorch_model.safetensors \
  --task t2av --learning_rate 1e-6 --max_train_steps 2000 \
  --blocks_to_swap 40 --use_pinned_memory_for_block_swap \
  --block_swap_trainable_ring --adafactor_triton \
  --output_dir output --output_name k6-pro-full
```

`--adafactor_triton` enables the CUDA/Triton fast path for BF16 matrices, with the
ordinary fused Adafactor implementation handling other tensor shapes. It requires
manual learning rates (`scale_parameter=False relative_step=False warmup_init=False`,
the defaults). Omit it to use PyTorch fused Adafactor throughout.

`--block_swap_trainable_ring` streams trainable block weights through pinned CPU
masters and reusable GPU buffers, including writeback after updates. It requires
`--blocks_to_swap`, `--use_pinned_memory_for_block_swap`, gradient checkpointing, and
fused backward. Omit it for ordinary bidirectional block swapping. Frozen-weight
`--block_swap_h2d_only`, FP8 and ConvRot INT8 bases cannot train the full transformer.

`--gradient_checkpointing_cpu_offload` stores checkpointed block-input activations
in pinned CPU memory and restores them for backward recomputation. Block outputs
and shared conditioning tensors stay on the GPU. Omit this option when activation
memory fits on the GPU; gradient checkpointing remains available independently.

`--k6_freeze_blocks 0-9,59` freezes selected visual blocks; `--k6_freeze_params` takes
parameter-name regexes. Frozen weights are still included in the dense checkpoint.
`--no-fused_backward_pass` permits other optimizers only without block swapping.
Gradient accumulation is supported. Dense checkpoints use native model keys and
can be loaded with `--dit`; `--save_state` / `--save_state_on_train_end` and `--resume`
also restore optimizer, scheduler, training RNG and the next batch position.
`--max_train_steps` remains the total target after resume. Keep the original seed,
gradient accumulation, dataset length and trainable parameter selection when
resuming; incompatible changes are rejected. Adafactor moments retain FP32
precision on resume. Checkpoints stream tensor by tensor by default; use
`--no-mem_eff_save` for the ordinary safetensors writer.

Accumulation keeps dense gradients until the update boundary; block swapping
offloads weights, not those gradients. Use one microstep per update to limit
gradient memory residency.

## Multi-reference generation (R2AV) / 複数参照生成 (R2AV)

`--task r2av` teaches Kandinsky 6 to generate a video (with audio) from a prompt plus up to 8 reference images. The prompt refers to them as `@Image 1`, `@Image 2`, ... in reference order. The text encoders stay frozen: the tags are plain text, and each reference gets a learned slot embedding (`visual_ref_slot_embeddings[k]` for `@Image k+1`) plus a new token type, so the DiT learns which image a tag means. References are letterboxed onto the target canvas, VAE-encoded, and appended after the generated frames as clean, masked frames at RoPE times after the clip. This is the same mechanism TI2AV uses for its first frame. Video references are not supported yet.

A released checkpoint has no reference tables, so R2AV must start with **full fine-tuning** (`kandinsky6_train.py`, BF16). That run adds and trains the new tables. LoRA, FP8 and ConvRot INT8 can only be used on a checkpoint that already has trained reference slots.

**1. Dataset.** Each JSONL video record takes an extra `references` list:

```json
{"video_path": "/data/v.mp4", "caption": "@Image 1 throws a ball to @Image 2 in a park", "references": [{"type": "image", "path": "/data/girl.png"}, {"type": "image", "path": "/data/dog.png"}]}
```

Choose your own video and reference-image dataset. Point `video_jsonl_file` in `dataset.toml` at the JSONL file. The order of `references` determines the `@Image k` tags used in each caption.

**2. Caching.** Run `kandinsky6_cache_latents.py` as usual. It encodes the references of each record onto that record's bucket canvas (`--ref_pad_value`, default 255 = white letterbox; `--max_references`, default 8). `--skip_existing` checks reference contents, paths, order, canvas and padding, and rebuilds caches when these change or references are removed. Older reference caches without a fingerprint are rebuilt. Cache the text encoder outputs as usual; regenerate the text cache if captions change.

**3. Training** (batch size 1 recommended, because a batch cannot mix reference counts):

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 kandinsky6_train.py \
  --dataset_config dataset.toml \
  --dit /models/k6-lite/transformer/diffusion_pytorch_model.safetensors \
  --task r2av --max_reference_slots 8 --ref_drop_prob 0.1 \
  --optimizer_type Adafactor --learning_rate 1e-5 --max_grad_norm 0 \
  --gradient_checkpointing --blocks_to_swap 16 --use_pinned_memory_for_block_swap \
  --block_swap_trainable_ring --adafactor_triton \
  --max_train_steps 2000 --output_dir output --output_name k6-r2av
```

`--ref_drop_prob` drops the whole reference set for some steps. This keeps a reference-free branch for `--ref_guidance_scale`. The saved checkpoint contains the reference tables, and later runs detect them automatically: omit `--max_reference_slots`, or pass the same value.

**4. Sampling.** In-training sample prompts take repeated `--ref` options (cannot be combined with `--i`). `--ref_guidance_scale` (default 0) adds extra guidance toward the references, at the cost of one extra forward per step:

```text
@Image 1 throws a ball to @Image 2 in a park. --w 768 --h 512 --f 121 --s 30 --g 4.0 --ref refs/girl.png --ref refs/dog.png
```

Standalone generation:

```bash
python kandinsky6_generate_reference_video.py --dit output/k6-r2av.safetensors \
  --vae /models/k6-lite/vae --audio_vae /models/k6-lite/audio_vae --vocoder /models/k6-lite/vocoder \
  --text_encoder_qwen /models/k6-lite/text_encoder --text_encoder_clip /models/k6-lite/text_encoder_2 \
  --ref refs/girl.png --ref refs/dog.png --prompt "@Image 1 throws a ball to @Image 2 in a park" \
  --width 768 --height 512 --frame_count 121 --steps 50 --output out.mp4 [--blocks_to_swap 16] [--lora_weight ...]
```

Use the same `--ref_pad_value` as during caching.

<details><summary>日本語</summary>

`--task r2av` は、プロンプトと最大8枚の参照画像から音声付き動画を生成するよう学習します。プロンプトでは参照順に `@Image 1`、`@Image 2` … と呼びます。text encoderは凍結したままです。各参照には学習されるslot embeddingと新しいtoken typeが付き、参照は対象canvasにletterboxしてVAE encodeした後、生成フレームの後ろにclean・maskedフレームとして追加されます(TI2AVの先頭フレームと同じ仕組み)。動画参照は未対応です。

公開checkpointには参照用tableがないため、最初は**full fine-tuning**(`kandinsky6_train.py`、BF16)が必要です。LoRA／FP8／ConvRot INT8は、参照slotが学習済みのcheckpointでのみ使えます。

任意の動画・参照画像データセットを使用できます。JSONLの各レコードに `references` を追加し、`dataset.toml` の `video_jsonl_file` にそのJSONLを指定します。`references` の順序をcaption内の `@Image k` と対応させてください。`kandinsky6_cache_latents.py` は参照も同じbucket canvasでencodeします。学習はbatch size 1を推奨します。サンプルプロンプトでは `--ref` を繰り返し指定し、単体生成には `kandinsky6_generate_reference_video.py` を使います。`--ref_pad_value` はキャッシュ時と同じ値にしてください。
</details>
