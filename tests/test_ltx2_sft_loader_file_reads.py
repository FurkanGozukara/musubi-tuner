"""LTX-2 models a training process loads from the checkpoint it already maps (sampling: text encoder, VAE, audio)
read the file with plain reads (in_process_checkpoint_loader): safetensors 0.8 could not map a file again while
tensors of an earlier map were alive (Windows, 2026-10-12)."""

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.ltx_2.loader.sft_loader import SafetensorsStateDictLoader, in_process_checkpoint_loader


class _Only:
    def __init__(self, word):
        self.word = word

    def apply_to_key(self, name):
        return name.replace("model.", "") if self.word in name else None

    def apply_to_key_value(self, name, value):
        return ((name, value),)


def test_file_reads_match_the_mapped_load_while_earlier_views_are_alive(tmp_path):
    path = str(tmp_path / "checkpoint.safetensors")
    tensors = {f"model.connector.{i}": torch.randn(64, 32).to(torch.bfloat16) for i in range(3)}
    tensors["model.transformer.0"] = torch.randn(128, 128)
    save_file(tensors, path, metadata={"config": json.dumps({"transformer": {"layers": 1}})})
    with safe_open(path, framework="pt", device="cpu") as handle:
        kept = handle.get_tensor("model.transformer.0")  # the training DiT's view of the file

    mapped = SafetensorsStateDictLoader().load(path, _Only("connector"), torch.device("cpu"))
    read = in_process_checkpoint_loader().load(path, _Only("connector"), torch.device("cpu"))
    assert sorted(read.sd) == sorted(mapped.sd) == ["connector.0", "connector.1", "connector.2"]
    assert all(torch.equal(read.sd[key], mapped.sd[key]) for key in read.sd)
    assert read.size == mapped.size and read.dtype == {torch.bfloat16}
    assert in_process_checkpoint_loader().metadata(path) == {"transformer": {"layers": 1}}
    assert torch.equal(kept, tensors["model.transformer.0"])
