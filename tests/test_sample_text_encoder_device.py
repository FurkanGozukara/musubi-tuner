"""--sample_text_encoder_device cpu: process_sample_prompts sees the CPU as its device and no FP8 text encoder flag."""

import argparse
import types

import torch

from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training.sample_text_encoder import sample_prompt_inputs


def _accelerator():
    return types.SimpleNamespace(device=torch.device("cuda", 0), is_main_process=True, unwrap_model=lambda model: model)


def test_without_the_setting_nothing_changes():
    args, accelerator = argparse.Namespace(sample_text_encoder_device=None, fp8_llm=True), _accelerator()
    assert sample_prompt_inputs(args, accelerator) == (args, accelerator)
    args.sample_text_encoder_device = "cuda"
    assert sample_prompt_inputs(args, accelerator) == (args, accelerator)
    assert sample_prompt_inputs(argparse.Namespace(), accelerator)[1] is accelerator  # an older config without the key


def test_cpu_gives_a_cpu_device_and_bf16_text_encoder_without_touching_the_run():
    args = argparse.Namespace(sample_text_encoder_device=" CPU ", fp8_llm=True, fp8_vl=True, fp8_t5=True, fp8_text_encoder=True, fp8_base=True)
    accelerator = _accelerator()
    cpu_args, cpu_accelerator = sample_prompt_inputs(args, accelerator)
    assert cpu_accelerator.device == torch.device("cpu")
    assert cpu_accelerator.is_main_process is True and cpu_accelerator.unwrap_model("m") == "m"  # the rest is the accelerator's
    assert not any(getattr(cpu_args, name) for name in ("fp8_llm", "fp8_vl", "fp8_t5", "fp8_text_encoder"))
    assert cpu_args.fp8_base is True  # the DiT's own precision is not the text encoder's
    assert args.fp8_llm and args.fp8_vl and args.fp8_t5 and args.fp8_text_encoder  # training keeps its settings
    assert accelerator.device == torch.device("cuda", 0)


def test_quantized_loaders_that_need_cuda_stay_on_the_accelerator():
    args, accelerator = argparse.Namespace(sample_text_encoder_device="cpu", gemma_load_in_8bit=True), _accelerator()
    assert sample_prompt_inputs(args, accelerator) == (args, accelerator)


def test_every_trainer_parser_takes_the_setting():
    parser = setup_parser_common()
    assert parser.parse_args(["--sample_text_encoder_device", "cpu"]).sample_text_encoder_device == "cpu"
    assert parser.parse_args([]).sample_text_encoder_device is None
