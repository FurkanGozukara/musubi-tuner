"""Kandinsky 6 multi-reference (R2AV) generation.

Loads an R2AV-trained transformer (full fine-tune output, optionally plus LoRA) and
generates a video with audio from a prompt and up to ``reference_slots`` reference
images, referred to in the prompt as ``@Image 1``, ``@Image 2``, ... in ``--ref`` order::

    python kandinsky6_generate_reference_video.py --dit r2av.safetensors --vae ... --audio_vae ... \\
        --vocoder ... --text_encoder_qwen ... --text_encoder_clip ... \\
        --ref girl.png --ref dog.png --prompt "@Image 1 walks @Image 2 on a beach" --output out.mp4

Sampling uses the same reference packing, RoPE and guidance code as training-time
sampling (``musubi_tuner.kandinsky6.reference``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from musubi_tuner.kandinsky6.defaults import DEFAULT_NEGATIVE_PROMPT
from musubi_tuner.kandinsky6.reference import DEFAULT_REF_PAD_VALUE, encode_reference_paths, sample_reference_latents

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dit", required=True, help="R2AV Kandinsky 6 transformer (safetensors with reference slots)")
    parser.add_argument("--vae", required=True, help="Kandinsky 6 HunyuanVideo VAE directory")
    parser.add_argument("--audio_vae", required=True, help="Kandinsky 6 MMAudio VAE checkpoint")
    parser.add_argument("--audio_vae_scaling_factor", type=float, default=0.5302)
    parser.add_argument("--vocoder", required=True, help="Kandinsky 6 BigVGAN checkpoint")
    parser.add_argument("--text_encoder_qwen", required=True)
    parser.add_argument("--text_encoder_clip", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--quantized_qwen", action="store_true")
    parser.add_argument("--ref", action="append", default=[], help="reference image; repeat in @Image 1, @Image 2, ... order")
    parser.add_argument("--ref_pad_value", type=int, default=DEFAULT_REF_PAD_VALUE, help="letterbox grey level (match caching)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", required=True, help="output .mp4")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frame_count", type=int, default=121, help="pixel frames at 24 fps (snapped to 4k+1)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--ref_guidance_scale", type=float, default=0.0, help="extra guidance toward the references")
    parser.add_argument("--scheduler_scale", type=float, default=1.0)
    parser.add_argument("--visual_rope_scale", type=float, nargs=3, default=(1.0, 2.0, 2.0), metavar=("T", "H", "W"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="stream this many DiT blocks from CPU")
    parser.add_argument("--fp8_base", action="store_true", help="load the DiT Linear weights as FP8")
    parser.add_argument("--convrot_int8", action="store_true", help="load the DiT Linear weights as ConvRot INT8")
    parser.add_argument("--lora_weight", nargs="*", default=[], help="LoRA(s) trained on top of this R2AV checkpoint")
    parser.add_argument("--lora_multiplier", type=float, nargs="*")
    return parser.parse_args(argv)


def load_reference_dit(args: argparse.Namespace, device: torch.device):
    from musubi_tuner.kandinsky6 import inspect_reference_layout, load_dit, load_dit_convrot_int8, load_dit_fp8

    if args.fp8_base and args.convrot_int8:
        raise ValueError("--fp8_base and --convrot_int8 are mutually exclusive")
    _, slots = inspect_reference_layout(args.dit)
    if not slots:
        raise ValueError(f"{args.dit} has no reference slots; train it with --task r2av first")
    loading_device = "cpu" if args.blocks_to_swap else device
    if args.convrot_int8:
        dit = load_dit_convrot_int8(args.dit, device=loading_device, quant_device=device)
    elif args.fp8_base:
        dit = load_dit_fp8(args.dit, device=loading_device, quant_device=device)
    else:
        dit = load_dit(args.dit, dtype=torch.bfloat16, device=loading_device)
    if args.lora_weight:
        from musubi_tuner.kandinsky6_generate_video import apply_lora_weights

        apply_lora_weights(dit, args.lora_weight, args.lora_multiplier, merge=not (args.convrot_int8 or args.fp8_base))
    dit.requires_grad_(False)
    dit.eval()
    if args.blocks_to_swap:
        from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

        dit.enable_block_swap(args.blocks_to_swap, BlockSwapConfig(device=device, supports_backward=False))
        dit.move_to_device_except_swap_blocks(device)
        dit.switch_block_swap_for_inference()
    else:
        dit.to(device)
    return dit


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    from musubi_tuner.kandinsky6.runtime.core.algo.mux import mux_video_audio
    from musubi_tuner.kandinsky6.runtime.core.algo.postprocess_audio import postprocess_audio
    from musubi_tuner.kandinsky6.runtime.core.algo.postprocess_video import postprocess_video
    from musubi_tuner.kandinsky6.runtime.core.components.text_embedder import Kandinsky6TextEmbedder
    from musubi_tuner.kandinsky6.runtime.core.components.vae_audio import build_audio_vae, build_vocoder
    from musubi_tuner.kandinsky6.runtime.core.components.vae_video import build_vae
    from musubi_tuner.kandinsky6_train_network import sample_canvas_size, sample_latent_frame_count

    device = torch.device(args.device)
    width, height = sample_canvas_size(args.width, args.height)
    _, video_frames = sample_latent_frame_count(args.frame_count)
    cpu = torch.device("cpu")

    # Text and reference encoding first, so their models leave the GPU before the DiT loads.
    text_embedder = Kandinsky6TextEmbedder(
        args.text_encoder_qwen,
        args.text_encoder_clip,
        max_length=args.max_length,
        device=device,
        quantized_qwen=args.quantized_qwen,
        text_token_padding=False,
    )
    text, _, attention_mask = text_embedder.encode([args.prompt])
    null, _, null_attention_mask = text_embedder.encode([args.negative_prompt])
    text = {key: value.to(device, torch.bfloat16) for key, value in text.items()}
    null = {key: value.to(device, torch.bfloat16) for key, value in null.items()}
    attention_mask = None if attention_mask is None else attention_mask.to(device)
    null_attention_mask = None if null_attention_mask is None else null_attention_mask.to(device)
    del text_embedder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    video_vae = build_vae(args.vae, device=device)
    video_vae.requires_grad_(False).eval()
    references = None
    if args.ref:
        latents = encode_reference_paths(video_vae, args.ref, width, height, args.ref_pad_value)
        references = latents.permute(0, 2, 3, 1).unsqueeze(0)  # [1,N,h,w,C]
    video_vae.to(cpu)

    dit = load_reference_dit(args, device)
    if references is not None and references.shape[1] > dit.reference_slots:
        raise ValueError(f"{references.shape[1]} references given, but the model has {dit.reference_slots} slots")
    result = sample_reference_latents(
        dit,
        references,
        text,
        null,
        attention_mask=attention_mask,
        null_attention_mask=null_attention_mask,
        video_frames=video_frames,
        latent_height=height // 8,
        latent_width=width // 8,
        seed=args.seed,
        device=device,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        reference_guidance_scale=args.ref_guidance_scale if references is not None else 0.0,
        scheduler_scale=args.scheduler_scale,
        visual_rope_scale=tuple(args.visual_rope_scale),
    )
    del dit
    if device.type == "cuda":
        torch.cuda.empty_cache()

    video_vae.to(device)
    frames = postprocess_video(result, video_vae, bs=1)
    video_vae.to(cpu)
    audio_vae = build_audio_vae(
        tod_vae_ckpt=args.audio_vae,
        mode="44k",
        need_vae_encoder=False,
        need_vae_decoder=True,
        scaling_factor=args.audio_vae_scaling_factor,
        device=device,
    )
    vocoder = build_vocoder(ckpt=args.vocoder, device=device)
    audio = postprocess_audio(result, audio_vae, vocoder)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mux_video_audio(frames[0], audio[0], output, fps=24, audio_sample_rate=44100)
    logger.info("Saved Kandinsky 6 R2AV video to %s", output)


if __name__ == "__main__":
    main()
