"""LoRA / DoRA training for MOSS-Image (Base or SFT; see moss_image/moss_utils.py for the model, text and VAE details).

Latents come from moss_image_cache_latents.py (the FLUX.2 VAE), contexts from moss_image_cache_text_encoder_outputs.py
(varlen (L, 4096) per caption; a batch is right-padded and the padding masked out of every attention, which gives the
vendor's per-item results). The loss is the rectified-flow velocity ``noise - latents`` with sigma 1 = noise; the DiT
takes the timestep on the vendor's 0..1000 scale. Samples follow the vendor's sampler: shift 5 snapped to the training
grid, CFG whose negative branch skips block 22, an empty negative prompt.
"""

import argparse
import logging
from typing import Optional

import torch
from accelerate import Accelerator
from diffusers.utils.torch_utils import randn_tensor

from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE, ARCHITECTURE_MOSS_IMAGE_FULL
from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.moss_image import moss_model, moss_utils
from musubi_tuner.training.accelerator_setup import clean_memory_on_device
from musubi_tuner.training.outputs import DiTOutput
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sample_decode import decode_sample
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_ext import NetworkTrainer
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MossImageNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MOSS_IMAGE

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MOSS_IMAGE_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = moss_utils.DEFAULT_GUIDANCE
        self.default_discrete_flow_shift = moss_utils.DEFAULT_SHIFT
        if args.sample_prompts and not args.text_encoder:
            raise ValueError("MOSS-Image: sampling needs --text_encoder (Qwen3.5-9B)")

    @staticmethod
    def _negative(prompt_dict: dict) -> str:
        """The negative prompt; none, or a blank one (a bare ``--n``), is the empty prompt, as in the vendor's sampler."""
        return str(prompt_dict.get("negative_prompt") or "").strip()

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        device = accelerator.device
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)
        texts = []
        for prompt_dict in prompts:
            for text in (prompt_dict.get("prompt", ""), self._negative(prompt_dict)):
                if text not in texts:
                    texts.append(text)

        text_encoder, tokenizer = moss_utils.load_text_encoder(
            args.text_encoder, dtype=torch.bfloat16, device=device, stream=bool(getattr(args, "low_vram_text_encoding", False))
        )
        logger.info("Encoding with Qwen3.5-9B")
        contexts = moss_utils.encode_prompts(text_encoder, tokenizer, texts)
        encoded = {text: ctx.cpu() for text, ctx in zip(texts, contexts)}
        del text_encoder, contexts
        clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["ctx"] = encoded[prompt_dict.get("prompt", "")]
            prompt_dict_copy["negative_ctx"] = encoded[self._negative(prompt_dict)]
            sample_parameters.append(prompt_dict_copy)
        clean_memory_on_device(accelerator.device)
        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """The vendor's sampler: fp32 Euler on the shift-5 grid snapped to the training sigmas, CFG 3 whose negative
        branch skips block 22, an empty negative prompt."""
        model: moss_model.MossImage = transformer
        device = accelerator.device
        ctx, ctx_mask = moss_utils.pad_contexts([sample_parameter["ctx"]], device, dit_dtype)
        neg_ctx, neg_mask = moss_utils.pad_contexts([sample_parameter["negative_ctx"]], device, dit_dtype)
        guidance = cfg_scale if cfg_scale is not None else guidance_scale
        shift = discrete_flow_shift if discrete_flow_shift else moss_utils.DEFAULT_SHIFT

        noise = randn_tensor((1, 128, height // 16, width // 16), generator=generator, device=device, dtype=torch.float32)
        latents = moss_utils.denoise(model, noise, ctx, ctx_mask, neg_ctx, neg_mask, sample_steps, guidance, shift)

        vae.to(device)
        vae.eval()
        logger.info(f"Decoding image from latents: {latents.shape}")
        with torch.no_grad():
            pixels = decode_sample(vae, vae.decode, latents.to(vae.dtype))
        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)
        vae.to("cpu")
        clean_memory_on_device(device)
        return pixels.unsqueeze(2)  # B C 1 H W

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info(f"Loading the FLUX.2 VAE from {args.vae}")
        return flux2_utils.load_ae(args.vae, dtype=vae_dtype, device="cpu", disable_mmap=True)

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        return moss_utils.load_dit(
            accelerator.device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )

    def compile_transformer(self, args, transformer):
        transformer: moss_model.MossImage = transformer
        return model_utils.compile_transformer(
            args,
            transformer,
            [transformer.layers, transformer.text_embedder.blocks],
            disable_linear=self.blocks_to_swap > 0,
            offloaders=[transformer.offloader if self.blocks_to_swap > 0 else None, None],
        )

    def scale_shift_latents(self, latents):
        return latents  # the FLUX.2 VAE normalizes its packed latents itself

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        model: moss_model.MossImage = transformer
        device = accelerator.device

        hidden = noisy_model_input.to(device=device, dtype=network_dtype)
        ctx, ctx_mask = moss_utils.pad_contexts(batch["ctx_vec"], device, network_dtype)
        if args.gradient_checkpointing:  # the checkpointed blocks' inputs must carry gradients
            hidden.requires_grad_(True)
            ctx.requires_grad_(True)

        t = timesteps.to(device=device, dtype=torch.float32)  # the vendor's 0..1000 scale, 1000 = pure noise
        model_pred = model(hidden, t, ctx, ctx_mask)

        target = noise - latents.to(device=device, dtype=network_dtype)
        return DiTOutput(pred=model_pred, target=target)

    # endregion model specific


def moss_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """MOSS-Image specific options."""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3.5-9B ComfyUI file, needed for sampling")
    parser.add_argument(
        "--low_vram_text_encoding",
        action="store_true",
        help=(
            "encode the sample prompts with Qwen3.5-9B kept in CPU RAM and run on the GPU one decoder layer at a time: "
            "under 1 GB of VRAM instead of about 18 GB, fast on any CPU"
        ),
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = moss_image_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if not str(args.vae_dtype or "").strip():
        args.vae_dtype = "float32"

    trainer = MossImageNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
