"""Qwen-Image 2.1 LoRA training."""

import argparse
import gc
import logging
import os

import torch
from musubi_tuner.utils.model_utils import cpu_text_encoder_dtype, mark_cpu_fp32_fallback
from safetensors.torch import load_file
from torch.nn import functional as F

from musubi_tuner.dataset.architectures import ARCHITECTURE_QWEN_IMAGE_21, ARCHITECTURE_QWEN_IMAGE_21_FULL
from musubi_tuner.networks import lora_qwen_image_21
from musubi_tuner.qwen_image_21 import qwen_image_21_model, qwen_image_21_sampling, qwen_image_21_utils
from musubi_tuner.training.base_weights_sampling import SampleBaseWeights, add_sample_multiplier_argument, sample_multiplier_deltas
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer, comfy_lora_path
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def dit_lora_names(dit_path: str) -> set[str]:
    """The LoRA module names (``lora_unet_<layer>``) of every weight in a Qwen-Image 2.1 DiT file or shard folder, in this
    model's layout (ComfyUI prefixes removed, the fused MLP gate / up split), read from the safetensors headers only."""
    from safetensors import safe_open

    paths = [dit_path]
    if os.path.isdir(dit_path):
        paths = sorted(os.path.join(dit_path, name) for name in os.listdir(dit_path) if name.endswith(".safetensors"))
    names = set()
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        for key in keys:
            for name in qwen_image_21_model.canonical_weight_hook(key, None)[0]:
                if name.endswith(".weight"):
                    names.add("lora_unet_" + name[: -len(".weight")].replace(".", "_"))
    return names


def load_base_weights(path: str, layer_names: set[str] | None = None) -> dict[str, torch.Tensor]:
    """A ``--base_weights`` file in this trainer's LoRA layout: ai-toolkit / ComfyUI / Diffusers keys are converted (the
    Turbo training adapter fuses the MLP gate and up layers), and every LoRA module must match a layer of the DiT. The merge
    while loading skips a key without a layer with only a warning, which would train on a partly merged adapter."""
    weights = load_file(path)
    converted = lora_qwen_image_21.convert_weight_keys(weights)
    weights = converted if converted is not None else weights
    if layer_names is not None:
        modules = {key.split(".", 1)[0] for key in weights if "." in key}
        unmatched = sorted(modules - layer_names)
        if unmatched:
            raise ValueError(
                f"--base_weights {path}: {len(unmatched)} of {len(modules)} LoRA modules have no layer in the Qwen-Image 2.1 "
                f"DiT (e.g. {', '.join(unmatched[:3])}); is it a Qwen-Image 2.1 LoRA?"
            )
    return weights


def prepare_conditioning(batch: dict, device: torch.device, dtype: torch.dtype):
    """Check reference cache consistency, pad text features, and pack reference latents."""
    embeds = batch["vl_embed"]
    lengths = [e.shape[0] for e in embeds]
    if not lengths or min(lengths) < 1 or any(e.ndim != 2 or e.shape[-1] != 4096 for e in embeds):
        raise ValueError("Expected Qwen3-VL 4096-dimensional text features; rebuild 2.1 text caches")
    text = torch.stack([F.pad(e, (0, 0, 0, max(lengths) - e.shape[0])) for e in embeds]).to(device=device, dtype=dtype)
    refs = []
    shapes = []
    control_keys = sorted((k for k in batch if k.startswith("latents_control_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    if control_keys != [f"latents_control_{i}" for i in range(len(control_keys))]:
        raise ValueError("Reference cache indices must be contiguous")
    for key in control_keys:
        latent = batch[key]
        refs.append(qwen_image_21_utils.pack_latents(latent).to(device=device, dtype=dtype))
        shapes.append((1, latent.shape[-2], latent.shape[-1]))
    if "image_slots" not in batch or "reference_grids" not in batch:
        raise ValueError("Missing 2.1 reference metadata; rebuild text caches even for text-to-image")
    slots, grids = batch["image_slots"], batch["reference_grids"]
    if len(slots) != len(embeds) or len(grids) != len(embeds):
        raise ValueError("Reference metadata batch size mismatch")
    expected = [list(shape[1:]) for shape in shapes]
    for row_slots, row_grids in zip(slots, grids):
        if len(row_slots) != len(refs) or row_grids.tolist() != expected:
            raise ValueError("Reference VLM and VAE cache layouts disagree; rebuild both caches with the same dataset config")
    if refs:
        if "reference_hashes" not in batch or "vlm_reference_hashes" not in batch:
            raise ValueError("Missing reference content hashes; rebuild both Qwen-Image 2.1 caches")
        vae_hashes, vlm_hashes = batch["reference_hashes"], batch["vlm_reference_hashes"]
        if len(vae_hashes) != len(embeds) or len(vlm_hashes) != len(embeds):
            raise ValueError("Reference fingerprint batch size mismatch")
        for left, right in zip(vae_hashes, vlm_hashes):
            if left.shape != (len(refs), 32) or not torch.equal(left.cpu(), right.cpu()):
                raise ValueError("Reference contents/order differ between VAE and VLM caches; rebuild both caches")
    return text, lengths, refs, shapes, [s.tolist() for s in slots]


class QwenImage21NetworkTrainer(NetworkTrainer):
    """Trainer for Qwen-Image 2.1."""

    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1
        self._sample_base_weights = SampleBaseWeights([])

    @property
    def architecture(self):
        return ARCHITECTURE_QWEN_IMAGE_21

    @property
    def architecture_full_name(self):
        return ARCHITECTURE_QWEN_IMAGE_21_FULL

    def on_post_save(self, args, accelerator, network, transformer, ckpt_name, save_dtype, metadata, force_sync_upload):
        """A ComfyUI / SwarmUI copy of each checkpoint, ``<name>.comfy.safetensors``, with diffusers keys
        (``transformer.<module>.lora_A.weight``, DoRA's ``dora_scale`` beside them). Comfy-Org's Qwen-Image 2.1 checkpoints
        fuse each block's MLP gate and up layers (``img_mlp.gate_up``) and ComfyUI maps LoRA weights onto the two halves
        only for such keys: with the trainer's own ``lora_unet_`` keys it skipped ``img_mlp.proj`` and
        ``img_mlp.gate_layer`` ("lora key not loaded"), two of the seven trained layers of every block
        (kohya-ss/musubi-tuner #1163). The trainer's own file stays as it is (resume, musubi's tools)."""
        if not accelerator.is_main_process:
            return
        from safetensors.torch import save_file

        from musubi_tuner.convert_lora import convert_to_diffusers

        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        comfy_file = comfy_lora_path(ckpt_file)
        weights = convert_to_diffusers("lora_unet_", "transformer", load_file(ckpt_file))
        save_file(weights, comfy_file, metadata={key: str(value) for key, value in (metadata or {}).items()})
        accelerator.print(f"saved the ComfyUI / SwarmUI LoRA: {comfy_file}")
        if args.huggingface_repo_id is not None:
            from musubi_tuner.utils import huggingface_utils

            huggingface_utils.upload(args, comfy_file, "/" + os.path.basename(comfy_file), force_sync_upload=force_sync_upload)

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = None
        if args.sample_prompts and (not args.text_encoder or not args.vae):
            raise ValueError("Sampling requires --text_encoder and --vae")
        self._sample_base_weights = SampleBaseWeights(sample_multiplier_deltas(args))
        if getattr(args, "dit_variant", "base") == "turbo" and not args.base_weights:
            logger.warning(
                "Qwen-Image 2.1 Turbo without a training adapter (--base_weights): training breaks the step distillation "
                "quickly; ostris/qwen_image_2_turbo_training_adapter keeps it while the LoRA trains."
            )

    def load_transformer(self, accelerator, args, dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype):
        lora_weights_list = None
        if args.base_weights:
            names = dit_lora_names(dit_path)
            lora_weights_list = [load_base_weights(path, names) for path in args.base_weights]
            # the trainer's INFO logging is not shown in the GUI: print what is merged
            print(
                "Qwen-Image 2.1: merging --base_weights while loading: "
                + ", ".join(f"{os.path.basename(path)} ({len({k.split('.', 1)[0] for k in sd})} modules)" for path, sd in zip(args.base_weights, lora_weights_list))
            )
        model = qwen_image_21_model.load_model(
            dit_path,
            device=accelerator.device,
            loading_device=loading_device,
            dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
            config_path=args.dit_config,
            disable_numpy_memmap=args.disable_numpy_memmap,
            attn_mode=attn_mode,
            lora_weights_list=lora_weights_list,
            lora_multipliers=args.base_weights_multiplier,
            num_layers=args.num_layers,
        )
        model.img_in_txt_in_offloading = args.img_in_txt_in_offloading
        return model

    def merge_base_weights(self, args, accelerator, transformer, network_module, weight_dtype):
        # load_model merges base weights before FP8 quantization.
        pass

    def on_before_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        if self._sample_base_weights:
            # built at the first sample, after the trained network: it wraps the trained layers' forward
            self._sample_base_weights.build(lora_qwen_image_21, accelerator.unwrap_model(transformer), load_base_weights)
            self._sample_base_weights.before_sampling(accelerator.device, dit_dtype)

    def on_after_sample_images(self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype):
        if self._sample_base_weights:
            self._sample_base_weights.after_sampling()

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(
            args,
            transformer,
            [transformer.transformer_blocks],
            disable_linear=self.blocks_to_swap > 0,
            offloaders=[getattr(transformer, "offloader", None) if self.blocks_to_swap > 0 else None],
        )

    def scale_shift_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Latents are normalized during caching.
        return latents

    def latent_tokens_for_timestep_sampling(self, latents: torch.Tensor) -> int:
        return latents.shape[-2] * latents.shape[-1]

    def load_vae(self, args, vae_dtype, vae_path):
        return qwen_image_21_utils.load_vae(vae_path, "cpu", vae_dtype, args.vae_tiling)

    def call_dit(
        self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
    ):
        target = qwen_image_21_utils.pack_latents(noisy_model_input).to(device=accelerator.device, dtype=network_dtype)
        text, lengths, refs, shapes, slots = prepare_conditioning(batch, accelerator.device, network_dtype)
        if args.gradient_checkpointing:
            target.requires_grad_(True)
            text.requires_grad_(True)
        shapes = [shapes + [(1, latents.shape[-2], latents.shape[-1])]]
        with accelerator.autocast():
            pred = transformer(
                hidden_states=target,
                encoder_hidden_states=text,
                timestep=timesteps / 1000.0,
                img_shapes=shapes,
                txt_seq_lens=lengths,
                reference_latents=refs,
                image_slots=slots,
            )
        pred = qwen_image_21_utils.unpack_latents(pred, latents.shape[-2], latents.shape[-1])
        return DiTOutput(pred=pred, target=(noise - latents).to(device=accelerator.device, dtype=network_dtype))

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        te_dtype = torch.bfloat16
        if not args.fp8_vl:
            te_dtype = cpu_text_encoder_dtype(accelerator.device, te_dtype, args.text_encoder, "Qwen3-VL text encoder")
        processor, encoder = qwen_image_21_utils.load_text_encoder(
            args.text_encoder, device=accelerator.device, dtype=te_dtype, fp8_vl=args.fp8_vl
        )
        mark_cpu_fp32_fallback(encoder, torch.bfloat16, te_dtype)
        prompts = load_prompts(sample_prompts)
        logger.info("Encoding with VLM")
        prompts = qwen_image_21_sampling.encode_sample_prompts(processor, encoder, prompts)
        del processor, encoder
        gc.collect()
        clean_memory_on_device(accelerator.device)
        return prompts

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
        with accelerator.autocast():
            pixels = qwen_image_21_sampling.sample_image(
                transformer,
                vae,
                sample_parameter=sample_parameter,
                device=accelerator.device,
                dit_dtype=dit_dtype,
                width=width,
                height=height,
                sample_steps=sample_steps,
                generator=generator,
                discrete_flow_shift=discrete_flow_shift,
                cfg_scale=cfg_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                # Turbo samples with the 8-step schedule its checkpoint ships, whatever the step count says
                sigmas=qwen_image_21_sampling.TURBO_SAMPLE_SIGMAS if getattr(args, "dit_variant", "base") == "turbo" else None,
            )
        return pixels.unsqueeze(2)


def setup_parser() -> argparse.ArgumentParser:
    parser = setup_parser_common()
    parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled FP8 for DiT weights")
    parser.add_argument("--fp8_vl", action="store_true", help="Store sampling VLM language block linears in FP8")
    parser.add_argument("--num_layers", type=int, default=None, help="Override DiT depth; checkpoint must match")
    parser.add_argument("--dit_config", help="Optional Diffusers transformer config.json")
    parser.add_argument("--text_encoder", help="Qwen3-VL-8B directory or BF16/FP32 safetensors file, needed only for sampling")
    parser.add_argument("--vae_tiling", action="store_true", help="Enable spatial tiling for the Qwen-Image 2.1 VAE")
    parser.add_argument(
        "--dit_variant",
        type=str,
        choices=["base", "turbo"],
        default="base",
        help="the --dit checkpoint: base, or the 8-step Turbo (samples with its fixed 8-step schedule; train it with "
        "ostris' training adapter as --base_weights and --base_weights_sample_multiplier 0)",
    )
    add_sample_multiplier_argument(parser)
    parser.set_defaults(network_module="networks.lora_qwen_image_21", dit_dtype="bfloat16", vae_dtype="bfloat16", split_attn=True)
    return parser


def main():
    parser = setup_parser()
    args = read_config_from_file(parser.parse_args(), parser)
    QwenImage21NetworkTrainer().train(args)


if __name__ == "__main__":
    main()
