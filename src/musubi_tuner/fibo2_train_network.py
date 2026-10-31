"""LoRA / DoRA training for FIBO 2.0 Dev, text-to-image and editing (see fibo2/fibo2_utils.py for the model, text,
reference and VAE details).

Latents come from fibo2_cache_latents.py (the FLUX.2 VAE), contexts from fibo2_cache_text_encoder_outputs.py (per
caption the (L, 15 * 2560) Qwen3-VL taps, expanded to the DiT's (L, 46080) bundles; a batch is right-padded and the
padding masked, which gives each item's own forward). Datasets with reference images (``latents_control_<i>``) train
editing as ai-toolkit does, on the same Dev model: the captions were encoded as Bria's edit prompts with the references'
pixels, and the DiT sees the references as clean latents on their own RoPE planes. The loss is the rectified-flow
velocity ``noise - latents``; the DiT takes ``1000 * sigma``. Samples use ComfyUI's sampler (shift 1.15 = alpha 3.158,
the simple scheduler, Euler, CFG 5 against the null caption, a non-empty negative as written, references in both CFG
passes, a single reference cropped to the sample's size, several each to their ~1 MP size).
"""

import argparse
import logging
from typing import Optional

import torch
from accelerate import Accelerator
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from musubi_tuner.dataset.architectures import ARCHITECTURE_FIBO2, ARCHITECTURE_FIBO2_FULL
from musubi_tuner.fibo2 import fibo2_model, fibo2_utils
from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.training.accelerator_setup import clean_memory_on_device
from musubi_tuner.training.outputs import DiTOutput
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sample_decode import decode_sample
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import DIRECT_TIMESTEP_SAMPLING_METHODS
from musubi_tuner.training.trainer_ext import NetworkTrainer
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.model_utils import cast_for_cache, cpu_text_encoder_dtype, mark_cpu_fp32_fallback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def reference_latents(batch: dict) -> list[torch.Tensor]:
    """The batch's reference latents ``latents_control_0``, ``_1``, ... (each (B, 128, h, w)), in order."""
    out = []
    while f"latents_control_{len(out)}" in batch:
        out.append(batch[f"latents_control_{len(out)}"])
    return out


def sample_size(prompt_dict: dict) -> tuple[int, int]:
    """(width, height) of a training sample: sample_image_inference's multiple of 8, then the latent grid's 16."""
    width = int(prompt_dict.get("width", 256)) // 8 * 8 // 16 * 16
    height = int(prompt_dict.get("height", 256)) // 8 * 8 // 16 * 16
    return max(16, width), max(16, height)


class Fibo2TrainerMixin:
    """FIBO 2.0's model hooks: Fibo2NetworkTrainer below runs them on the extended training loop (LoRA / DoRA),
    fibo2_train.py on trainer_base's loop, whose hooks the dense trainer overrides (full fine-tuning)."""

    _convrot_int8 = False

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_FIBO2

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_FIBO2_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        self._i2v_training = False
        self._control_training = False  # video control; reference images are handled in call_dit
        self.default_guidance_scale = fibo2_utils.DEFAULT_GUIDANCE
        self.default_discrete_flow_shift = fibo2_utils.DEFAULT_SHIFT
        if args.sample_prompts and not args.text_encoder:
            raise ValueError("FIBO 2.0: sampling needs --text_encoder (Qwen3-VL-4B)")
        if args.fp8_base and not args.fp8_scaled:
            raise ValueError("FIBO 2.0: --fp8_base needs --fp8_scaled (a plain FP8 cast would also cast the embedders and norms)")
        if getattr(args, "convrot_int8", False) and (args.fp8_base or args.fp8_scaled):
            raise ValueError("FIBO 2.0: --convrot_int8 and --fp8_base / --fp8_scaled quantize the same weights; choose one")

    @staticmethod
    def _negative(prompt_dict: dict) -> str:
        """The negative prompt; none, or a blank one (a bare ``--n``), is the null caption, as in ComfyUI's
        TextEncodeFibo2."""
        return str(prompt_dict.get("negative_prompt") or "").strip()

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        device = accelerator.device
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        te_dtype = cpu_text_encoder_dtype(device, torch.bfloat16, args.text_encoder, "Qwen3-VL text encoder")
        text_encoder, tokenizer, image_processor = fibo2_utils.load_text_encoder(args.text_encoder, dtype=te_dtype, device=device)
        mark_cpu_fp32_fallback(text_encoder, torch.bfloat16, te_dtype)
        logger.info("Encoding with Qwen3-VL")

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            width, height = sample_size(prompt_dict)
            crops = [Image.open(path).convert("RGB") for path in prompt_dict.get("control_image_path") or []]
            if crops:  # as ComfyUI's TextEncodeFibo2: one reference at the result's size, several at their own ~1 MP sizes
                sizes = fibo2_utils.sample_reference_sizes(crops, width, height)
                crops = [fibo2_utils.resize_and_crop(image, w, h) for image, (w, h) in zip(crops, sizes)]
            positive, negative = fibo2_utils.encode_prompts(
                text_encoder,
                tokenizer,
                image_processor,
                [prompt_dict.get("prompt", ""), self._negative(prompt_dict)],
                [[fibo2_utils.vlm_image(crop) for crop in crops] or None, None],
                as_written=[False, True],
            )
            prompt_dict_copy["ctx"] = cast_for_cache(text_encoder, positive.cpu())
            prompt_dict_copy["negative_ctx"] = cast_for_cache(text_encoder, negative.cpu())
            prompt_dict_copy["reference_pixels"] = [fibo2_utils.pil_to_tensor(crop) for crop in crops]
            sample_parameters.append(prompt_dict_copy)

        del text_encoder
        clean_memory_on_device(device)
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
        """ComfyUI's sampler for FIBO 2.0 (see the module docstring)."""
        model: fibo2_model.Fibo2 = transformer
        device = accelerator.device
        width, height = sample_size({"width": width, "height": height})
        ctx, ctx_mask = fibo2_utils.stack_contexts([sample_parameter["ctx"]], device, dit_dtype)
        neg_ctx, neg_mask = fibo2_utils.stack_contexts([sample_parameter["negative_ctx"]], device, dit_dtype)
        guidance = cfg_scale if cfg_scale is not None else guidance_scale
        shift = discrete_flow_shift if discrete_flow_shift else fibo2_utils.DEFAULT_SHIFT

        ref_latents = None
        pixels = sample_parameter.get("reference_pixels") or []
        if pixels:
            vae.to(device)
            vae.eval()
            with torch.no_grad():
                ref_latents = [vae.encode(p.to(device, vae.dtype)).to(dit_dtype) for p in pixels]
            vae.to("cpu")
            clean_memory_on_device(device)

        noise = randn_tensor((1, 128, height // 16, width // 16), generator=generator, device=device, dtype=torch.float32)
        latents = fibo2_utils.denoise(
            model, noise, ctx, ctx_mask, neg_ctx, neg_mask, sample_steps, guidance, shift, ref_latents=ref_latents, dtype=dit_dtype
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
        model = fibo2_utils.load_dit(
            accelerator.device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
            convrot_int8=bool(getattr(args, "convrot_int8", False)),
            convrot_int8_bwd=getattr(args, "convrot_int8_bwd", None) or "bf16",
            allow_int8_file=not getattr(self, "trains_full_transformer", False),  # an INT8 file trains LoRA only
        )
        self._convrot_int8 = bool(getattr(model, "is_convrot_int8", False))
        return model

    def compile_transformer(self, args, transformer):
        transformer: fibo2_model.Fibo2 = transformer
        return model_utils.compile_transformer(
            args,
            transformer,
            [transformer.layers, transformer.noise_refiner, transformer.perceiver.layers],
            # ConvRot INT8 layers run a custom autograd Function with Triton kernels, which dynamo cannot trace
            disable_linear=self.blocks_to_swap > 0 or getattr(self, "_convrot_int8", False),
            offloaders=[transformer.offloader if self.blocks_to_swap > 0 else None, None, None],
        )

    def scale_shift_latents(self, latents):
        return latents  # the FLUX.2 VAE normalizes its packed latents itself

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dtype):
        # the DiT is conditioned on t = 1000 * sigma, as Bria's pipeline, ai-toolkit and ComfyUI pass it; the shared direct
        # samplers return 1000 * sigma + 1 (HunyuanVideo's 1..1000 index). The scheduler path's grid pairs them exactly.
        noisy_model_input, timesteps = super().get_noisy_model_input_and_timesteps(
            args, noise, latents, timesteps, noise_scheduler, device, dtype
        )
        if args.timestep_sampling in DIRECT_TIMESTEP_SAMPLING_METHODS:
            timesteps = timesteps - 1
        return noisy_model_input, timesteps

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
        model: fibo2_model.Fibo2 = transformer
        device = accelerator.device

        hidden = noisy_model_input.to(device=device, dtype=network_dtype)
        ctx, ctx_mask = fibo2_utils.stack_contexts(batch["ctx_vec"], device, network_dtype)
        refs = [ref.to(device=device, dtype=network_dtype) for ref in reference_latents(batch)]

        t = timesteps.to(device=device, dtype=torch.float32)  # 0..1000, 1000 = pure noise
        model_pred = model(hidden, t, ctx, ctx_mask, refs or None)

        # in FP32: in full fine-tuning network_dtype is BF16, which would round the latents
        target = noise.to(device=device, dtype=torch.float32) - latents.to(device=device, dtype=torch.float32)
        return DiTOutput(pred=model_pred, target=target)

    # endregion model specific


class Fibo2NetworkTrainer(Fibo2TrainerMixin, NetworkTrainer):
    """LoRA / DoRA training on the extended training loop."""


def fibo2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """FIBO 2.0 specific options."""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument(
        "--convrot_int8",
        action="store_true",
        help=(
            "LoRA: quantize the frozen DiT's block linears (all but the adaLN modulations) to ConvRot INT8 while loading "
            "(Hadamard rotation, row-wise INT8: half their BF16 memory; fused Triton INT8 GEMM, a slower dequantizing fallback "
            "without Triton). Comfy-Org's fibo2_dev_int8_convrot file is used as it is without this option. Not combinable with FP8"
        ),
    )
    parser.add_argument(
        "--convrot_int8_bwd",
        type=str,
        default="bf16",
        choices=["bf16", "int8"],
        help="backward of the ConvRot INT8 layers: bf16 (default, a transient dequantized matmul) or int8 (the fused INT8 GEMM, faster, needs Triton)",
    )
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3-VL-4B safetensors, needed for sampling")
    return parser


def main():
    parser = setup_parser_common()
    parser = fibo2_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if not str(args.vae_dtype or "").strip():
        args.vae_dtype = "float32"

    trainer = Fibo2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
