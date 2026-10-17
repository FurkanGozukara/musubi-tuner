"""LoRA / DoRA training for FLUX.3 [klein] (base model; see flux_3/flux3_utils.py for the model, text and VAE details).

Latents come from flux_3_cache_latents.py (the FLUX.2 VAE), contexts from flux_3_cache_text_encoder_outputs.py (varlen
(L, 20480) per caption, zero-padded per batch). The loss is the rectified-flow velocity ``noise - latents`` at t = 1 for
noise. Datasets with reference images (``latents_control_<i>``) train editing: the references enter the DiT's
``image_cond`` stream as clean tokens at t = 0 and l = -1, -2, ...; without references that stream's five blocks stay on
the CPU and out of the LoRA. Samples run the prompt and its negative as separate forwards at their own lengths, as
ComfyUI does (the DiT attends to every text token, so padding the shorter one would change it).
"""

import argparse
import logging
from typing import Optional

import torch
from accelerate import Accelerator
from diffusers.utils.torch_utils import randn_tensor

from musubi_tuner.dataset.architectures import ARCHITECTURE_FLUX_3, ARCHITECTURE_FLUX_3_FULL
from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.flux_3 import flux3_model, flux3_utils
from musubi_tuner.training.accelerator_setup import clean_memory_on_device
from musubi_tuner.training.outputs import DiTOutput
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sample_decode import decode_sample
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_ext import NetworkTrainer
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def reference_latents(batch: dict) -> list[torch.Tensor]:
    """The batch's reference latents ``latents_control_0``, ``_1``, ... (each (B, 128, h, w)), in order."""
    out = []
    while f"latents_control_{len(out)}" in batch:
        out.append(batch[f"latents_control_{len(out)}"])
    return out


def pack_references(latents_list: list[torch.Tensor], device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference latents -> (B, sum h_i * w_i, 128) tokens and their ids, reference i at l = -1 - i."""
    tokens, ids = [], []
    for i, latents in enumerate(latents_list):
        batch, _, height, width = latents.shape
        tokens.append(flux3_utils.pack_latents(latents.to(device=device, dtype=dtype)))
        ids.append(flux3_model.image_ids(batch, height, width, -1 - i, device))
    return torch.cat(tokens, dim=1), torch.cat(ids, dim=1)


class Flux3NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_FLUX_3

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_FLUX_3_FULL

    def configure_reference_target_ranges_from_datasets(self, datasets) -> None:
        """Called with the datasets before the DiT loads: editing datasets (reference images) train the reference
        stream; text-to-image runs leave it on the CPU and out of the LoRA (its blocks never run)."""
        self._trains_references = any(getattr(d, "has_control", False) or getattr(d, "control_directory", None) for d in datasets)

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        self._i2v_training = False
        self._control_training = False  # video control; reference images are handled in call_dit
        self.default_guidance_scale = flux3_utils.DEFAULT_GUIDANCE
        self.default_discrete_flow_shift = flux3_utils.DEFAULT_SHIFT
        if args.sample_prompts and not args.text_encoder:
            raise ValueError("FLUX.3: sampling needs --text_encoder (Qwen3-VL-4B)")

    @staticmethod
    def _negative(prompt_dict: dict) -> str:
        """The negative prompt; none, or a blank one (a bare ``--n``), is the empty prompt, as in BFL's sampler."""
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

        text_encoder, tokenizer = flux3_utils.load_text_encoder(args.text_encoder, dtype=torch.bfloat16, device=device)
        logger.info("Encoding with Qwen3-VL")
        contexts = flux3_utils.encode_prompts(text_encoder, tokenizer, texts)
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
        """FLUX.3 [klein] base sampling: Euler on the shifted schedule with classifier-free guidance (the base model
        is not guidance-distilled; its reference sampler uses 50 steps, guidance 3.5, shift 6.93)."""
        model: flux3_model.Flux3 = transformer
        device = accelerator.device
        ctx = sample_parameter["ctx"].unsqueeze(0).to(device=device, dtype=dit_dtype)
        neg_ctx = sample_parameter["negative_ctx"].unsqueeze(0).to(device=device, dtype=dit_dtype)
        guidance = cfg_scale if cfg_scale is not None else guidance_scale
        alpha = discrete_flow_shift if discrete_flow_shift else flux3_utils.DEFAULT_SHIFT

        image_cond = image_cond_ids = None
        if "control_image_path" in sample_parameter:
            vae.to(device)
            vae.eval()
            paths = sample_parameter["control_image_path"]
            limit_size = (2024, 2024) if len(paths) == 1 else (1024, 1024)
            latents_list = []
            with torch.no_grad():
                for path in paths:
                    tensor, _, _ = flux2_utils.preprocess_control_image(path, limit_size)
                    latents_list.append(vae.encode(tensor.to(device, vae.dtype)))
            vae.to("cpu")
            clean_memory_on_device(device)
            self._ensure_reference_stream(accelerator.unwrap_model(model), device)
            image_cond, image_cond_ids = pack_references(latents_list, device, dit_dtype)

        noise = randn_tensor((1, 128, height // 16, width // 16), generator=generator, device=device, dtype=torch.float32)
        latents = flux3_utils.denoise(
            model, noise, ctx, neg_ctx, sample_steps, guidance, alpha, image_cond=image_cond, image_cond_ids=image_cond_ids, dtype=dit_dtype
        )

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
        model = flux3_utils.load_flow_model(
            accelerator.device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        trains_references = getattr(self, "_trains_references", True)
        model.trains_references = trains_references  # read by networks/lora_flux_3.py
        model.keep_image_cond_on_cpu(not trains_references)
        return model

    @staticmethod
    def _ensure_reference_stream(model: flux3_model.Flux3, device: torch.device):
        if model.image_cond_on_cpu:
            logger.info("FLUX.3: reference images in use, moving the reference stream to the GPU")
            model.content_mode_blocks["image_cond"].to(device)
            model.keep_image_cond_on_cpu(False)

    def compile_transformer(self, args, transformer):
        transformer: flux3_model.Flux3 = transformer
        return model_utils.compile_transformer(
            args,
            transformer,
            [transformer.single_blocks, transformer.txt_mode_blocks, transformer.content_mode_blocks["image"]],
            disable_linear=self.blocks_to_swap > 0,
            offloaders=[transformer.offloader_single if self.blocks_to_swap > 0 else None, None, None],
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
        model: flux3_model.Flux3 = transformer
        device = accelerator.device
        batch_size, _, height, width = latents.shape

        tokens = flux3_utils.pack_latents(noisy_model_input).to(device=device, dtype=network_dtype)
        ids = flux3_model.image_ids(batch_size, height, width, 0, device)
        ctx = flux3_utils.pad_contexts(batch["ctx_vec"], device, network_dtype)

        image_cond = image_cond_ids = None
        refs = reference_latents(batch)
        if refs:
            image_cond, image_cond_ids = pack_references(refs, device, network_dtype)

        if args.gradient_checkpointing:  # the checkpointed blocks' inputs must carry gradients
            tokens.requires_grad_(True)
            ctx.requires_grad_(True)
            if image_cond is not None:
                image_cond.requires_grad_(True)

        t = (timesteps / 1000.0).to(device=device, dtype=torch.float32)
        model_pred = model(ctx, tokens, ids, t, image_cond=image_cond, image_cond_ids=image_cond_ids)
        model_pred = flux3_utils.unpack_latents(model_pred, height, width)

        target = noise - latents.to(device=device, dtype=network_dtype)
        return DiTOutput(pred=model_pred, target=target)

    # endregion model specific


def flux3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """FLUX.3 [klein] specific options."""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3-VL-4B safetensors, needed for sampling")
    return parser


def main():
    parser = setup_parser_common()
    parser = flux3_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if not str(args.vae_dtype or "").strip():
        args.vae_dtype = "float32"  # the reference runs the VAE in float32

    trainer = Flux3NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
