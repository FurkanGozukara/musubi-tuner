"""LoRA / DoRA training for MOSS-Image (Base or SFT; see moss_image/moss_utils.py for the model, text and VAE details).

Latents come from moss_image_cache_latents.py (the FLUX.2 VAE), contexts from moss_image_cache_text_encoder_outputs.py
(varlen (L, 4096) per caption; a batch is right-padded and the padding masked out of every attention, which gives the
vendor's per-item results). The loss is the rectified-flow velocity ``noise - latents`` with sigma 1 = noise; the DiT
takes the timestep on the vendor's 0..1000 scale. Samples follow the vendor's sampler: shift 5 snapped to the training
grid, CFG whose negative branch skips block 22, an empty negative prompt.

``--control_adapter`` trains a colour / depth control adapter instead (moss_image/moss_control.py): each image's maps
(control images ``<name>_0`` colour regions, ``<name>_1`` depth) are cached as latents beside it, the DiT's patch
embedder takes ``[noisy | colour | depth]``, and every checkpoint is written in the format ComfyUI's MOSS-Image
Color/Depth Control node loads.

``--moss_edit`` trains an edit LoRA (ai-toolkit's layout; ComfyUI: Text Encode MOSS-Image Edit, edit_format "ai-toolkit",
or "ai-toolkit (kv_cache off)" with ``--moss_edit_kv_cache false``): the dataset's control images are the references,
cached as clean latents at their own framing (control_resolution) and encoded into the captions' contexts through
Qwen3.5's vision tower (moss_image_cache_text_encoder_outputs.py --moss_edit); the DiT sees them after the text at t = 0.

``--caption_dropout_rate`` trains that share of samples with the empty caption (an edit sample keeps its references);
caption files with several captions (``---===---`` lines) train one of them per step, chosen when the batch is loaded.
"""

import argparse
import logging
import os
from typing import Optional

import torch
from accelerate import Accelerator
from PIL import Image

from musubi_tuner.dataset.architectures import ARCHITECTURE_MOSS_IMAGE, ARCHITECTURE_MOSS_IMAGE_FULL
from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.moss_image import moss_control, moss_model, moss_utils
from musubi_tuner.training.accelerator_setup import clean_memory_on_device
from musubi_tuner.training import replica_trainer
from musubi_tuner.training.outputs import DiTOutput
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sample_decode import decode_sample
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import DIRECT_TIMESTEP_SAMPLING_METHODS
from musubi_tuner.training.trainer_ext import NetworkTrainer
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _true(value) -> bool:
    """A flag from the command line or a runtime TOML (a bool, or "true" / "false" text)."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def batch_reference_latents(batch: dict, batch_size: int) -> Optional[list]:
    """Per sample, the list of cached reference latents (``latents_control_0``, ``latents_control_1``, ... in the batch,
    each (B, 128, h, w)), or None without references. Cached latents of a batch are stacked, so every sample of a batch
    holds the same number of references at the same sizes (the GUI keeps edit batches at 1)."""
    indices = sorted(int(key.rsplit("_", 1)[1]) for key in batch if key.startswith("latents_control_") and key.rsplit("_", 1)[1].isdigit())
    if not indices:
        return None
    if indices != list(range(len(indices))):
        raise ValueError(f"reference latents {indices} in the batch are not numbered 0..{len(indices) - 1}")
    stacks = [batch[f"latents_control_{index}"] for index in indices]
    for stack in stacks:
        if stack.shape[0] != batch_size:
            raise ValueError(
                f"a batch of {batch_size} holds {stack.shape[0]} reference latents per slot: every sample of an edit batch needs "
                "the same number of references (use dataset batch size 1)"
            )
    return [[stack[sample] for stack in stacks] for sample in range(batch_size)]


class MossImageTrainerMixin:
    """MOSS-Image's model hooks: MossImageNetworkTrainer below runs them on the extended training loop (LoRA / DoRA,
    control adapters), moss_image_train.py on trainer_base's loop, whose hooks the dense trainer overrides (full
    fine-tuning)."""

    control_adapter = False
    moss_edit = False
    edit_isolate_refs = True
    _control_embedder: Optional[moss_control.MossControlEmbedder] = None
    _control_module_names: dict = {}
    _control_shares = moss_control.DEFAULT_CONTROL_DROPOUT
    _caption_dropout = 0.0
    _empty_ctx: Optional[torch.Tensor] = None
    _convrot_int8 = False

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MOSS_IMAGE

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MOSS_IMAGE_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        if args.mixed_precision == "fp16":
            # KBlueLeaf/MOSS-Image-FP16-fixed measured the residual stream near 62,000 at block 26 (FP16 ends at 65,504)
            logger.warning(
                "MOSS-Image in FP16: its residual stream comes close to FP16's limit (about 62,000 of 65,504 measured at "
                "block 26), so a LoRA or a larger image can overflow to inf / NaN. Train in BF16 where the GPU supports it."
            )
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = moss_utils.DEFAULT_GUIDANCE
        self.default_discrete_flow_shift = moss_utils.DEFAULT_SHIFT
        if args.sample_prompts and not args.text_encoder:
            raise ValueError("MOSS-Image: sampling needs --text_encoder (Qwen3.5-9B)")
        if getattr(args, "convrot_int8", False) and (args.fp8_base or args.fp8_scaled):
            raise ValueError("MOSS-Image: --convrot_int8 and --fp8_base / --fp8_scaled quantize the same weights; choose one")
        self.control_adapter = bool(getattr(args, "control_adapter", False))
        self.moss_edit = bool(getattr(args, "moss_edit", False))
        self.edit_isolate_refs = _true(getattr(args, "moss_edit_kv_cache", True))
        if self.moss_edit and self.control_adapter:
            raise ValueError(
                "MOSS-Image: --moss_edit (an edit LoRA) and --control_adapter (a colour / depth adapter) both read the dataset's "
                "control images; choose one"
            )
        if self.moss_edit and getattr(self, "trains_full_transformer", False):
            raise ValueError("MOSS-Image edit training trains a LoRA / DoRA: use moss_image_train_network.py")
        if self.control_adapter:
            self._validate_control_adapter_args(args)
            self._control_shares = moss_control.parse_control_dropout(getattr(args, "control_dropout", None))
            self._caption_dropout = float(getattr(args, "control_caption_dropout", 0.0) or 0.0)
        rate = getattr(args, "caption_dropout_rate", None)
        if rate is not None and str(rate).strip() != "":
            rate = float(rate)
            if not 0.0 <= rate < 1.0:
                raise ValueError(f"--caption_dropout_rate must be in [0, 1), got {rate}")
            # every dataset type; a control adapter keeps its own --control_caption_dropout unless a rate above 0 is given
            if rate > 0 or not self.control_adapter:
                self._caption_dropout = rate
        if _true(getattr(args, "train_text_embedder_proj", False)):
            if getattr(self, "trains_full_transformer", False):
                logger.info("MOSS-Image full fine-tuning trains text_embedder.proj with every other weight")
            else:
                network_args = list(args.network_args or [])
                if not any(arg.split("=", 1)[0].strip() == "include_text_proj" for arg in network_args):
                    args.network_args = network_args + ["include_text_proj=True"]

    def on_train_datasets(self, args, datasets):
        """Control adapters: the datasets must carry the maps (a cache-only dataset cannot tell). Edit LoRAs: the datasets
        must carry the references. Caption dropout reads the empty caption's context written by the text encoder caching
        (edit captions keep their references: their dropped contexts are cached per item)."""
        has_maps = any(getattr(dataset, "has_control", False) for dataset in datasets)
        cache_only = any(getattr(dataset, "cache_only", False) for dataset in datasets)
        if self.control_adapter and not has_maps and not cache_only:
            raise ValueError(
                "MOSS-Image control adapter training needs colour and depth maps: a control_directory in the dataset config "
                "with <image name>_0.png (one flat colour per region) and <image name>_1.png (depth, near = white) for each "
                "image; moss_image_control_maps.py makes both"
            )
        if self.moss_edit and not has_maps and not cache_only:
            raise ValueError(
                "MOSS-Image edit training needs reference images: a control_directory in the dataset config with each "
                "target's reference(s), <image name>.png or <image name>_0.png, <image name>_1.png, ..."
            )
        if self._caption_dropout > 0 and not self.moss_edit:
            self._empty_ctx = moss_control.load_empty_caption([getattr(dataset, "cache_directory", None) for dataset in datasets])

    def _validate_control_adapter_args(self, args):
        """What a control adapter needs: the trainer's own LoRA / input-axis DoRA (the export reads its keys), weights
        that start from the DiT (the file is the ComfyUI format, so a run continues with --resume), and a DiT whose
        embedder stays in its own dtype (no whole-model FP8 cast). The text refiner's blocks join the LoRA, as in the
        published recipe, unless the network arguments choose."""
        if getattr(self, "trains_full_transformer", False):
            raise ValueError("--control_adapter trains a LoRA / DoRA adapter: use moss_image_train_network.py")
        if str(getattr(args, "network_module", "") or "") not in ("networks.lora_moss_image", "musubi_tuner.networks.lora_moss_image"):
            raise ValueError("--control_adapter needs --network_module networks.lora_moss_image")
        if args.network_weights or getattr(args, "dim_from_weights", False):
            raise ValueError(
                "--control_adapter starts from the DiT: the saved files are ComfyUI control adapters, so continue a run "
                "with --resume (save the training state with --save_state)"
            )
        if args.fp8_base and not args.fp8_scaled:
            raise ValueError("--control_adapter with --fp8_base needs --fp8_scaled (the plain FP8 cast would cast the trained embedder)")
        if not float(getattr(args, "control_lr_ratio", 10.0) or 0.0) > 0:
            raise ValueError("--control_lr_ratio must be above 0")
        dropout = float(getattr(args, "control_caption_dropout", 0.0) or 0.0)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"--control_caption_dropout must be in [0, 1), got {dropout}")
        network_args = list(args.network_args or [])
        if any("dora_axis" in arg and "output" in arg.split("=", 1)[-1].lower() for arg in network_args):
            raise ValueError("--control_adapter writes DoRA as ComfyUI's input-axis dora_scale: drop dora_axis=output")
        if not any(arg.split("=", 1)[0].strip() == "include_text_embedder" for arg in network_args):
            args.network_args = network_args + ["include_text_embedder=True"]

    @staticmethod
    def _negative(prompt_dict: dict) -> str:
        """The negative prompt; none, or a blank one (a bare ``--n``), is the empty prompt, as in the vendor's sampler."""
        return str(prompt_dict.get("negative_prompt") or "").strip()

    @staticmethod
    def _sample_size(prompt_dict: dict) -> tuple[int, int]:
        """A sample's (width, height) on the DiT's 16 px grid (sample_image_inference's defaults)."""
        width = int(prompt_dict.get("width", 256)) // 8 * 8
        height = int(prompt_dict.get("height", 256)) // 8 * 8
        return max(16, width // 16 * 16), max(16, height // 16 * 16)

    def _sample_references(self, prompt_dict: dict) -> list:
        """An edit sample's references: the prompt's ``--ci`` images, each fitted to the sample's pixel count with its own
        aspect (as the training references), as (H, W, 3) uint8 arrays."""
        width, height = self._sample_size(prompt_dict)
        references = []
        for path in prompt_dict.get("control_image_path") or []:
            if moss_control.is_no_map(path):
                continue
            with Image.open(str(path).strip()) as opened:
                references.append(moss_utils.fit_reference(opened.convert("RGB"), width, height))
        if not references:
            logger.warning(f"MOSS-Image edit sample without reference images (--ci): {prompt_dict.get('prompt', '')!r}")
        return references

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
            args.text_encoder,
            dtype=torch.bfloat16,
            device=device,
            stream=bool(getattr(args, "low_vram_text_encoding", False)),
            vision=self.moss_edit,
        )
        logger.info("Encoding with Qwen3.5-9B")
        sample_parameters = []
        if self.moss_edit:
            # every prompt and its negative see the sample's references, as ComfyUI's TextEncodeMOSSImageEdit for both
            for prompt_dict in prompts:
                references = self._sample_references(prompt_dict)
                texts = [prompt_dict.get("prompt", ""), self._negative(prompt_dict)]
                contexts = moss_utils.encode_prompts_with_images(text_encoder, tokenizer, texts, [references, references])
                prompt_dict_copy = prompt_dict.copy()
                prompt_dict_copy["ctx"], prompt_dict_copy["negative_ctx"] = (ctx.cpu() for ctx in contexts)
                prompt_dict_copy["reference_images"] = references
                sample_parameters.append(prompt_dict_copy)
            del text_encoder
        else:
            contexts = moss_utils.encode_prompts(text_encoder, tokenizer, texts)
            encoded = {text: ctx.cpu() for text, ctx in zip(texts, contexts)}
            del text_encoder, contexts
            for prompt_dict in prompts:
                prompt_dict_copy = prompt_dict.copy()
                prompt_dict_copy["ctx"] = encoded[prompt_dict.get("prompt", "")]
                prompt_dict_copy["negative_ctx"] = encoded[self._negative(prompt_dict)]
                sample_parameters.append(prompt_dict_copy)
        clean_memory_on_device(accelerator.device)
        return sample_parameters

    @staticmethod
    @torch.no_grad()
    def _encode_references(vae, references: list, device) -> list:
        """(H, W, 3) uint8 references -> clean (128, h, w) latents with the FLUX.2 VAE (moved back to the CPU after)."""
        if not references:
            return []
        vae.to(device)
        try:
            latents = []
            for image in references:
                pixels = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 127.5 - 1.0
                latents.append(vae.encode(pixels.to(vae.dtype))[0].float())
            return latents
        finally:
            vae.to("cpu")

    @staticmethod
    def sample_noise(generator: torch.Generator, height: int, width: int, device) -> torch.Tensor:
        """The sample's initial noise as ComfyUI draws it for the same seed: float32 normal noise of the 128-channel latent
        from a CPU generator seeded with the sample's seed (comfy.sample.prepare_noise), then moved to the device."""
        cpu = torch.Generator(device="cpu").manual_seed(int(generator.initial_seed()))
        return torch.randn((1, 128, height // 16, width // 16), generator=cpu, dtype=torch.float32).to(device)

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
        branch skips block 22, an empty negative prompt. A control adapter's sample takes its maps from the prompt's
        ``--ci`` entries (colour, then depth; ``--ci none`` leaves one out) on both CFG branches, as ComfyUI's node; an
        edit sample takes its references from them (both branches). The initial noise comes from a CPU generator, so a
        sample's seed gives the same noise as ComfyUI's KSampler with that seed."""
        model: moss_model.MossImage = transformer
        device = accelerator.device
        if self.control_adapter:
            control = moss_control.sample_control(vae, sample_parameter.get("control_image_path"), width, height, device)
            model = moss_control.ControlledDiT(model, control)
        ref_latents = None
        if self.moss_edit:
            references = self._encode_references(vae, sample_parameter.get("reference_images") or [], device)
            ref_latents = [references] if references else None
        ctx, ctx_mask = moss_utils.pad_contexts([sample_parameter["ctx"]], device, dit_dtype)
        neg_ctx, neg_mask = moss_utils.pad_contexts([sample_parameter["negative_ctx"]], device, dit_dtype)
        guidance = cfg_scale if cfg_scale is not None else guidance_scale
        shift = discrete_flow_shift if discrete_flow_shift else moss_utils.DEFAULT_SHIFT

        noise = self.sample_noise(generator, height, width, device)
        latents = moss_utils.denoise(
            model, noise, ctx, ctx_mask, neg_ctx, neg_mask, sample_steps, guidance, shift,
            ref_latents=ref_latents, isolate_refs=self.edit_isolate_refs,
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
        model = moss_utils.load_dit(
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
        if self.control_adapter:
            self._control_embedder = moss_control.widen_video_embedder(model)
            self._control_module_names = moss_control.lora_module_names(model)
            logger.info("MOSS-Image control adapter: video_embedder widened to [noisy | colour | depth] (384 inputs)")
        return model

    def compile_transformer(self, args, transformer):
        transformer: moss_model.MossImage = transformer
        return model_utils.compile_transformer(
            args,
            transformer,
            [transformer.layers, transformer.text_embedder.blocks],
            # ConvRot INT8 Linears compile as well: the fused Triton path is a torch.library custom op (opaque node, no
            # graph break; kohya-ss/sd-scripts a4aaa20), so only block swapping needs the Linear exclusion
            disable_linear=self.blocks_to_swap > 0,
            offloaders=[transformer.offloader if self.blocks_to_swap > 0 else None, None],
        )

    def scale_shift_latents(self, latents):
        return latents  # the FLUX.2 VAE normalizes its packed latents itself

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dtype):
        # the DiT is conditioned on the vendor's t = 1000 * sigma, as its sampler and both reference trainers (kohya's
        # sd-scripts, sdbds's port) pass it; the shared direct samplers return 1000 * sigma + 1 (HunyuanVideo's 1..1000
        # index), which told the model the noise level of sigma + 0.001. The scheduler path's grid pairs them exactly.
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
        if os.environ.get(replica_trainer.LOSS_TRACE_ENV):  # every micro-step's loss, to compare a replica run with this one
            micro_step = getattr(self, "_traced_micro_steps", 0)
            self._traced_micro_steps = micro_step + 1
            with torch.no_grad():
                loss = replica_trainer.ReplicaTrainerMixin._replica_loss(args, pred.detach().float(), target, network_dtype)
            replica_trainer.trace_loss(micro_step, accelerator.device, float(loss))
        return DiTOutput(pred=pred, target=target)

    def prepare_dit_inputs(self, args, device, latents, batch, noise, noisy_model_input, timesteps, network_dtype) -> dict:
        """Everything a DiT step needs, with every random draw of the step (control map dropout, caption dropout) made
        here, in the order a single-GPU run makes them: replica training (training/replica_trainer.py) prepares a step
        in the main thread and runs it on another GPU (run_dit)."""
        hidden = noisy_model_input.to(device=device, dtype=network_dtype)
        contexts = list(batch["ctx_vec"])
        if self.control_adapter:
            # the clean map latents beside the noisy one, each kept or dropped per sample; captions dropped on their own
            maps = [batch.get(f"latents_control_{index}") for index in range(len(moss_control.CONTROL_MAPS))]
            keep = moss_control.control_keep(hidden.shape[0], self._control_shares)
            hidden = moss_control.control_input(hidden, maps, keep)
        if self._caption_dropout > 0 and (self.moss_edit or self._empty_ctx is not None):
            dropped = (torch.rand(len(contexts)) < self._caption_dropout).tolist()
            if any(dropped):
                if not getattr(self, "_logged_caption_dropout", False):
                    self._logged_caption_dropout = True
                    print(f"MOSS-Image caption dropout {self._caption_dropout}: a sample trains with the empty caption" + (" (its references kept)" if self.moss_edit else ""))
                if self.moss_edit:  # the dropped caption keeps its references: its context is cached per item
                    unconds = batch.get("ctx_vec_uncond")
                    if unconds is None or len(unconds) != len(contexts):
                        raise ValueError(
                            "MOSS-Image edit caption dropout needs each caption's references-only context in the text "
                            "encoder caches: run moss_image_cache_text_encoder_outputs.py --moss_edit again"
                        )
                    contexts = [uncond if drop else context for context, uncond, drop in zip(contexts, unconds, dropped)]
                else:
                    contexts = [self._empty_ctx if drop else context for context, drop in zip(contexts, dropped)]
        ref_latents = batch_reference_latents(batch, hidden.shape[0]) if self.moss_edit else None
        if self.moss_edit and ref_latents is None:
            raise ValueError("MOSS-Image edit training: the batch holds no reference latents (latents_control_*); cache the latents again")
        return {
            "hidden": hidden,
            "contexts": contexts,
            "ref_latents": ref_latents,
            "timesteps": timesteps,
            "noise": noise,
            "latents": latents,
        }

    def run_dit(self, args, transformer, device, network_dtype, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """The DiT forward of a step prepared by prepare_dit_inputs, on ``device``: (prediction, FP32 target)."""
        model: moss_model.MossImage = transformer
        hidden = inputs["hidden"].to(device=device, dtype=network_dtype)
        ctx, ctx_mask = moss_utils.pad_contexts(inputs["contexts"], device, network_dtype)
        if args.gradient_checkpointing:  # the checkpointed blocks' inputs must carry gradients
            hidden.requires_grad_(True)
            ctx.requires_grad_(True)

        t = inputs["timesteps"].to(device=device, dtype=torch.float32)  # the vendor's 0..1000 scale, 1000 = pure noise
        ref_kwargs = {}
        if inputs.get("ref_latents") is not None:
            ref_kwargs = {
                "ref_latents": [[ref.to(device=device) for ref in refs] for refs in inputs["ref_latents"]],  # the DiT casts them
                "isolate_refs": self.edit_isolate_refs,
            }
        model_pred = model(hidden, t, ctx, ctx_mask, **ref_kwargs)

        # in FP32 like both reference trainers: in full fine-tuning network_dtype is BF16, which rounded the latents
        target = inputs["noise"].to(device=device, dtype=torch.float32) - inputs["latents"].to(device=device, dtype=torch.float32)
        return model_pred, target

    # endregion model specific


class MossImageNetworkTrainer(MossImageTrainerMixin, NetworkTrainer):
    """LoRA / DoRA training, edit LoRAs and colour / depth control adapters on the extended training loop, whose dataset,
    optimizer-group, checkpoint, gradient-clipping and save hooks carry the control adapter."""

    def _prepare_network_optimizer_params(self, args, network):
        """The network's groups, then the control embedder's: its released columns and bias at the LoRA rate (no weight
        decay: they start from the DiT's values), its control columns at --control_lr_ratio times that rate."""
        params, descriptions = super()._prepare_network_optimizer_params(args, network)
        if not self.control_adapter:
            return params, descriptions
        embedder = self._control_embedder
        embedder.requires_grad_(True)
        lr = float(args.learning_rate)
        released = [embedder.base_weight] + ([embedder.bias] if embedder.bias is not None else [])
        params = list(params) + [
            {"params": released, "lr": lr, "weight_decay": 0.0, "group_name": "control_embedder"},
            {"params": [embedder.control_weight], "lr": lr * float(args.control_lr_ratio), "group_name": "control_maps"},
        ]
        return params, list(descriptions or []) + ["control embedder", "control maps"]

    def pre_train_hook(self, args, accelerator, transformer=None, network=None):
        super().pre_train_hook(args, accelerator, transformer=transformer, network=network)
        if self.control_adapter:
            accelerator.register_for_checkpointing(self._control_embedder)  # --save_state / --resume keep the embedder

    def extra_grad_params(self) -> list:
        if not self.control_adapter or self._control_embedder is None:
            return []
        return [param for param in self._control_embedder.parameters() if param.requires_grad]

    def post_save_checkpoint_hook(self, args, ckpt_file, ckpt_name, accelerator, force_sync_upload=False, **kwargs):
        """Control adapters: the checkpoint becomes the file ComfyUI loads (moss_control.write_comfy_control_file)."""
        super().post_save_checkpoint_hook(args, ckpt_file, ckpt_name, accelerator, force_sync_upload, **kwargs)
        if not self.control_adapter or not accelerator.is_main_process:
            return
        base_model = "moss-sft" if "sft" in os.path.basename(str(args.dit)).lower() else "moss-base"
        metadata = moss_control.control_metadata(self._control_shares, None, base_model)
        metadata["control_lr_ratio"] = str(args.control_lr_ratio)
        metadata["control_caption_dropout"] = str(self._caption_dropout)
        count = moss_control.write_comfy_control_file(ckpt_file, self._control_embedder, self._control_module_names, metadata)
        shares = ", ".join(f"{name} {value:.3f}" for name, value in self._control_embedder.control_share().items())
        accelerator.print(f"saved the ComfyUI MOSS-Image control adapter: {ckpt_file} ({count} tensors; control columns / noisy: {shares})")


def moss_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """MOSS-Image specific options."""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument(
        "--convrot_int8",
        action="store_true",
        help=(
            "LoRA: quantize the joint blocks' projections of the frozen DiT to ConvRot INT8 while loading (Hadamard rotation, "
            "row-wise INT8: half their BF16 memory; fused Triton INT8 GEMM, a slower dequantizing fallback without Triton). "
            "A ComfyUI INT8 ConvRot DiT file is used as it is without this option. Not combinable with FP8"
        ),
    )
    parser.add_argument(
        "--convrot_int8_bwd",
        type=str,
        default="bf16",
        choices=["bf16", "int8"],
        help="backward of the ConvRot INT8 layers: bf16 (default, a transient dequantized matmul) or int8 (the fused INT8 GEMM, faster, needs Triton)",
    )
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3.5-9B ComfyUI file, needed for sampling")
    parser.add_argument(
        "--low_vram_text_encoding",
        action="store_true",
        help=(
            "encode the sample prompts with Qwen3.5-9B kept in CPU RAM and run on the GPU one decoder layer at a time: "
            "under 1 GB of VRAM instead of about 18 GB, fast on any CPU"
        ),
    )
    parser.add_argument(
        "--control_adapter",
        action="store_true",
        help=(
            "train a colour / depth control adapter for ComfyUI's MOSS-Image Color/Depth Control node: the dataset's control "
            "images <name>_0 (one flat colour per region) and <name>_1 (depth, near = white) are the maps; the saved files "
            "are the adapter ComfyUI loads"
        ),
    )
    parser.add_argument(
        "--control_lr_ratio",
        type=float,
        default=10.0,
        help="learning rate of the embedder's new control columns relative to --learning_rate (the published recipe: 10)",
    )
    parser.add_argument(
        "--control_dropout",
        type=str,
        default=None,
        help="shares of steps that keep the colour map only, the depth map only, both, none (default 0.35,0.35,0.2,0.1)",
    )
    parser.add_argument(
        "--control_caption_dropout",
        type=float,
        default=0.1,
        help="share of samples trained with the empty caption, independent of the maps (the published recipe: 0.1)",
    )
    parser.add_argument(
        "--moss_edit",
        action="store_true",
        help=(
            "train an edit LoRA (ai-toolkit's MOSS-Image edit layout): the dataset's control images are the references, "
            "cached as clean latents and seen by the text encoder (cache the text with --moss_edit too); ComfyUI: Text "
            "Encode MOSS-Image Edit, edit_format ai-toolkit"
        ),
    )
    parser.add_argument(
        "--moss_edit_kv_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "edit LoRAs: the references attend only to each other (ai-toolkit's kv_cache; ComfyUI edit_format 'ai-toolkit', "
            "whose references are computed once per image). --no-moss_edit_kv_cache: they also attend to the image "
            "(edit_format 'ai-toolkit (kv_cache off)')"
        ),
    )
    parser.add_argument(
        "--caption_dropout_rate",
        type=float,
        default=None,
        help=(
            "share of samples trained with the empty caption, for every dataset (an edit sample keeps its references); "
            "a value above 0 replaces --control_caption_dropout"
        ),
    )
    parser.add_argument(
        "--train_text_embedder_proj",
        action="store_true",
        help="also train a LoRA on text_embedder.proj, the projection of the Qwen3.5 states into the DiT (network_args include_text_proj=True)",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = moss_image_setup_parser(parser)
    parser = replica_trainer.add_replica_arguments(parser)  # LoRA on several GPUs in one process (no DDP)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if not str(args.vae_dtype or "").strip():
        args.vae_dtype = "float32"

    trainer = replica_trainer.wrap_trainer_class(MossImageNetworkTrainer, args)()
    trainer.train(args)


if __name__ == "__main__":
    main()
