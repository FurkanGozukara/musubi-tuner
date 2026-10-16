"""Where the text encoder encodes the sample prompts.

Training encodes its sample prompts before loading the DiT, with the text encoder on the accelerator's device. A preset
whose text encoder caching runs on the CPU does so because the text encoder does not fit its GPU, and then the sample
prompts cannot fit either (FLUX.2 Dev: Mistral 3 in BF16 peaked at 45.5 GB with the 32 GB preset). With
``--sample_text_encoder_device cpu`` (the GUI sets it from the caching device) ``process_sample_prompts`` sees the CPU
as its device, in BF16: FP8 text encoder weights have no CPU kernels.
"""

from __future__ import annotations

import copy
import logging

import torch

logger = logging.getLogger(__name__)

# text encoder precision flags of the trainers, off for a CPU text encoder
FP8_TEXT_ENCODER_FLAGS = ("fp8_llm", "fp8_vl", "fp8_t5", "fp8_text_encoder")
# quantized loaders that need CUDA (LTX-2 Gemma through bitsandbytes): the sample prompts stay on the accelerator
CUDA_ONLY_TEXT_ENCODER_FLAGS = ("gemma_load_in_8bit", "gemma_load_in_4bit")


class _OnDevice:
    """The accelerator with another ``device``; everything else is the accelerator's own."""

    def __init__(self, accelerator, device: torch.device):
        self._accelerator = accelerator
        self.device = device

    def __getattr__(self, name):
        return getattr(self._accelerator, name)


def sample_text_encoder_on_cpu(args) -> bool:
    if str(getattr(args, "sample_text_encoder_device", None) or "").strip().lower() != "cpu":
        return False
    return not any(getattr(args, name, False) for name in CUDA_ONLY_TEXT_ENCODER_FLAGS)


def sample_prompt_inputs(args, accelerator):
    """``(args, accelerator)`` for ``process_sample_prompts``: unchanged, or with --sample_text_encoder_device cpu a copy
    of ``args`` without FP8 text encoder flags and the accelerator seen with the CPU as its device."""
    if not sample_text_encoder_on_cpu(args):
        return args, accelerator
    cpu_args = copy.copy(args)
    for name in FP8_TEXT_ENCODER_FLAGS:
        if getattr(cpu_args, name, False):
            setattr(cpu_args, name, False)
    logger.info("Sample prompts: the text encoder runs on the CPU (--sample_text_encoder_device cpu)")
    return cpu_args, _OnDevice(accelerator, torch.device("cpu"))
