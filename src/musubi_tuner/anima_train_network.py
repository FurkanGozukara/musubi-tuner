"""LoRA / DoRA training for Anima (circlestone-labs/Anima Base; see anima/anima_utils.py for the files, the text
conditioning and the sampler).

Latents come from anima_cache_latents.py (the Qwen-Image VAE, normalized), text from anima_cache_text_encoder_outputs.py
(per caption the Qwen3-0.6B states and the T5 ids). The DiT's LLM adapter turns them into the 512-token context inside
every forward (a batch of captions of different lengths runs masked, which gives each caption's own unpadded result);
it stays frozen and out of the LoRA unless the network arguments say ``train_llm_adapter=True``. The loss is the
rectified-flow velocity ``noise - latents`` with sigma 1 = noise, and the DiT takes sigma (0..1) as its timestep.
Samples are ComfyUI's default Anima workflow: er_sde, simple scheduler, shift 3, 30 steps, CFG 4, ComfyUI's seeded noise.
"""

import argparse
import logging
from typing import Optional

import torch
from accelerate import Accelerator

from musubi_tuner.anima import anima_model, anima_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_ANIMA, ARCHITECTURE_ANIMA_FULL
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


class AnimaTrainerMixin:
    """Anima's model hooks (shared with a dense fine-tuner on trainer_base's loop)."""

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_ANIMA

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_ANIMA_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = anima_utils.DEFAULT_GUIDANCE
        self.default_discrete_flow_shift = anima_utils.DEFAULT_SHIFT  # samples; --discrete_flow_shift sets training only
        if args.sample_prompts and not args.text_encoder:
            raise ValueError("Anima: sampling needs --text_encoder (Qwen3-0.6B, qwen_3_06b_base.safetensors)")
        if args.fp8_base and not args.fp8_scaled:
            raise ValueError("Anima: --fp8_base needs --fp8_scaled (a plain FP8 cast of the whole DiT has no FP8 kernels here)")
        if getattr(args, "convrot_int8", False) and (args.fp8_base or args.fp8_scaled):
            raise ValueError("Anima: --convrot_int8 and --fp8_base / --fp8_scaled quantize the same weights; choose one")
        sampler = str(getattr(args, "sample_solver", None) or anima_utils.DEFAULT_SAMPLER)
        if sampler not in anima_utils.SAMPLERS:
            raise ValueError(f"Anima: --sample_solver must be one of {anima_utils.SAMPLERS}, got {sampler!r}")

    @staticmethod
    def _negative(prompt_dict: dict) -> str:
        """The prompt's --n text; without one ComfyUI's default Anima negative (the model card's workflow), a bare --n
        (blank) the empty prompt."""
        negative = prompt_dict.get("negative_prompt")
        if negative is None:
            return anima_utils.DEFAULT_NEGATIVE_PROMPT
        return str(negative).strip()

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        device = accelerator.device
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)
        texts = []
        for prompt_dict in prompts:
            for text in (prompt_dict.get("prompt", ""), self._negative(prompt_dict)):
                if text not in texts:
                    texts.append(text)

        te_dtype = cpu_text_encoder_dtype(device, torch.bfloat16, args.text_encoder, "Qwen3-0.6B text encoder")
        text_encoder = anima_utils.load_text_encoder(args.text_encoder, dtype=te_dtype, device=device)
        mark_cpu_fp32_fallback(text_encoder, torch.bfloat16, te_dtype)
        qwen_tokenizer, t5_tokenizer = anima_utils.load_tokenizers()
        logger.info("Encoding with Qwen3-0.6B")
        encoded = anima_utils.encode_prompts(text_encoder, qwen_tokenizer, t5_tokenizer, texts, batch_size=1)
        by_text = {text: (cast_for_cache(text_encoder, states.cpu()), ids) for text, (states, ids) in zip(texts, encoded)}
        del text_encoder, encoded
        clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["anima_text"] = by_text[prompt_dict.get("prompt", "")]
            prompt_dict_copy["anima_negative_text"] = by_text[self._negative(prompt_dict)]
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
        """ComfyUI's default Anima sampling: each prompt's context through the adapter on its own (as ComfyUI encodes
        one conditioning at a time), er_sde (or euler) on the simple schedule of shift 3, CFG with the negative."""
        model: anima_model.Anima = accelerator.unwrap_model(transformer)
        device = accelerator.device
        width, height = max(16, width // 16 * 16), max(16, height // 16 * 16)
        guidance = cfg_scale if cfg_scale is not None else guidance_scale
        shift = discrete_flow_shift if discrete_flow_shift else anima_utils.DEFAULT_SHIFT
        sampler = str(getattr(args, "sample_solver", None) or anima_utils.DEFAULT_SAMPLER)

        def context(entry):
            states, ids = entry
            return model.encode_context(states[None].to(device=device, dtype=dit_dtype), ids[None].to(device))

        with torch.no_grad():
            ctx = context(sample_parameter["anima_text"])
            neg_ctx = context(sample_parameter["anima_negative_text"]) if guidance != 1.0 else None
            latents = anima_utils.denoise(
                model, ctx, neg_ctx, height, width, sample_steps, guidance, shift, sampler, seed=sample_parameter.get("seed"), generator=generator
            )

        vae.to(device)
        vae.eval()
        logger.info(f"Decoding image from latents: {latents.shape}")
        with torch.no_grad():
            pixels = decode_sample(vae, vae.decode_to_pixels, latents.to(vae.dtype))  # (B, C, H, W) in 0..1
        pixels = pixels.to(torch.float32).cpu()
        vae.to("cpu")
        clean_memory_on_device(device)
        return pixels.unsqueeze(2)  # B C 1 H W

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info(f"Loading the Qwen-Image VAE from {args.vae}")
        return anima_utils.load_vae(args.vae, device="cpu", dtype=vae_dtype)

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
        model = anima_utils.load_dit(
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
            # block swap moves tensors between blocks: every block in one layout (blocks 0, 1 and 27 quantized too)
            convrot_int8_all_blocks=(getattr(self, "blocks_to_swap", 0) or 0) > 0,
        )
        self._convrot_int8 = bool(getattr(model, "is_convrot_int8", False))
        return model

    def compile_transformer(self, args, transformer):
        transformer: anima_model.Anima = transformer
        return model_utils.compile_transformer(
            args,
            transformer,
            [transformer.blocks],
            # ConvRot INT8 layers run a custom autograd Function with Triton kernels, which dynamo cannot trace
            disable_linear=self.blocks_to_swap > 0 or getattr(self, "_convrot_int8", False),
            offloaders=[transformer.offloader if self.blocks_to_swap > 0 else None],
        )

    def scale_shift_latents(self, latents):
        return latents  # the Qwen-Image VAE's latents are normalized at caching

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dtype):
        # the DiT is conditioned on sigma; the shared direct samplers return 1000 * sigma + 1 (HunyuanVideo's 1..1000
        # index), so take the 1 back (as MOSS-Image does) and the DiT sees the noise level it gets
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
        inputs = self.prepare_dit_inputs(args, accelerator.device, latents, batch, noise, noisy_model_input, timesteps, network_dtype)
        pred, target = self.run_dit(args, transformer, accelerator.device, network_dtype, inputs)
        return DiTOutput(pred=pred, target=target)

    def prepare_dit_inputs(self, args, device, latents, batch, noise, noisy_model_input, timesteps, network_dtype) -> dict:
        """Everything a DiT step needs (Anima makes no random draw here): the replica trainer
        (training/replica_trainer.py) prepares a step in the main thread and runs it on another GPU with run_dit."""
        return {
            "hidden": noisy_model_input,
            "qwen_states": list(batch["ctx_vec"]),
            "t5_ids": list(batch["t5_ids"]),
            "timesteps": timesteps,
            "noise": noise,
            "latents": latents,
        }

    def run_dit(self, args, transformer, device, network_dtype, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """The DiT forward of a step prepared by prepare_dit_inputs, on ``device``: (prediction, FP32 target)."""
        model: anima_model.Anima = transformer
        states, states_mask, t5_ids, t5_mask = anima_utils.batch_text(inputs["qwen_states"], inputs["t5_ids"], device, network_dtype)
        hidden = inputs["hidden"].to(device=device, dtype=network_dtype)
        if args.gradient_checkpointing:  # the checkpointed blocks' inputs carry gradients, as in the other trainers
            hidden.requires_grad_(True)

        sigma = (inputs["timesteps"].to(device=device, dtype=torch.float32) / 1000.0).reshape(-1)
        # the adapter runs inside the forward (kohya: correct DDP gradient synchronization with an adapter LoRA)
        model_pred = model(hidden, sigma, states, t5_ids=t5_ids, t5_mask=t5_mask, context_mask=states_mask)

        # in FP32 (BF16 networks rounded the latents before, MOSS-Image's finding)
        target = inputs["noise"].to(device=device, dtype=torch.float32) - inputs["latents"].to(device=device, dtype=torch.float32)
        return model_pred, target

    # endregion model specific


class AnimaNetworkTrainer(AnimaTrainerMixin, NetworkTrainer):
    """LoRA / DoRA training on the extended training loop."""


def anima_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Anima specific options."""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument(
        "--convrot_int8",
        action="store_true",
        help=(
            "LoRA: quantize the frozen DiT's block projections (blocks 2-26; every block with block swap) to ConvRot INT8 while loading (Hadamard rotation, "
            "row-wise INT8; fused Triton INT8 GEMM, a slower dequantizing fallback without Triton). A comfy_quant INT8 ConvRot "
            "DiT file (ruwwww/Anima-ConvRot) is used as it is without this option. Not combinable with FP8"
        ),
    )
    parser.add_argument(
        "--convrot_int8_bwd",
        type=str,
        default="bf16",
        choices=["bf16", "int8"],
        help="backward of the ConvRot INT8 layers: bf16 (default, a transient dequantized matmul) or int8 (the fused INT8 GEMM, faster, needs Triton)",
    )
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3-0.6B base (qwen_3_06b_base.safetensors), needed for sampling")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default=anima_utils.DEFAULT_SAMPLER,
        choices=list(anima_utils.SAMPLERS),
        help="sampler of the training samples: er_sde (ComfyUI's default Anima workflow) or euler; both on the simple schedule",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = anima_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if not str(args.vae_dtype or "").strip():
        args.vae_dtype = "bfloat16"  # the Qwen-Image VAE file's dtype, as kohya and ComfyUI run it

    trainer = AnimaNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
