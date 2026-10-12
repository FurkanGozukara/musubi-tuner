import json

import safetensors
import torch
from musubi_tuner.ltx_2.loader.primitives import StateDict, StateDictLoader
from musubi_tuner.ltx_2.loader.sd_ops import SDOps


class SafetensorsStateDictLoader(StateDictLoader):
    """
    Loads weights from safetensors files without metadata support.
    Use this for loading raw weight files. For model files that include
    configuration metadata, use SafetensorsModelStateDictLoader instead.
    """

    def __init__(self, *, cpu_staging: bool = False, file_reads: bool = False):
        """Create a weight loader.

        When ``cpu_staging`` is enabled, tensors are opened on CPU and then
        copied synchronously to the requested device. This avoids relying on
        safetensors' direct device-backed storage path.

        ``file_reads`` reads every tensor with plain file reads (musubi's
        MemoryEfficientSafeOpen without memory maps). A process that already holds
        tensors mapped from the same file (the training DiT, loaded from the
        single-file checkpoint) cannot map it again: safetensors 0.8 raised
        "Attempted to access the data pointer on an invalid python storage" for a
        CUDA open and crashed for a CPU one (Windows, 2026-10-12).
        """
        self.cpu_staging = cpu_staging
        self.file_reads = file_reads

    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def load(self, path: str | list[str], sd_ops: SDOps, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """
        sd = {}
        size = 0
        dtype = set()
        device = device or torch.device("cpu")
        open_device = torch.device("cpu") if self.cpu_staging else device
        model_paths = path if isinstance(path, list) else [path]
        if self.file_reads:
            return self._load_with_file_reads(model_paths, sd_ops, device)
        for shard_path in model_paths:
            with safetensors.safe_open(shard_path, framework="pt", device=str(open_device)) as f:
                safetensor_keys = f.keys()
                for name in safetensor_keys:
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(
                        device=device,
                        non_blocking=not self.cpu_staging,
                        copy=False,
                    )
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value

        return StateDict(sd=sd, device=device, size=size, dtype=dtype)

    def _load_with_file_reads(self, model_paths: list[str], sd_ops: SDOps | None, device: torch.device) -> StateDict:
        from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

        sd = {}
        size = 0
        dtype = set()
        for shard_path in model_paths:
            with MemoryEfficientSafeOpen(shard_path, disable_numpy_memmap=True) as f:
                for name in f.keys():
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(device=device)
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value
        return StateDict(sd=sd, device=device, size=size, dtype=dtype)


class SafetensorsModelStateDictLoader(StateDictLoader):
    """
    Loads weights and configuration metadata from safetensors model files.
    Unlike SafetensorsStateDictLoader, this loader can read model configuration
    from the safetensors file metadata via the metadata() method.
    """

    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None, *, cpu_staging: bool = False):
        if weight_loader is not None and cpu_staging:
            raise ValueError("cpu_staging cannot be combined with a custom weight_loader")
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader(cpu_staging=cpu_staging)

    def metadata(self, path: str) -> dict:
        if getattr(self.weight_loader, "file_reads", False):
            from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

            with MemoryEfficientSafeOpen(path, disable_numpy_memmap=True) as f:
                return json.loads(f.metadata()["config"])
        with safetensors.safe_open(path, framework="pt") as f:
            return json.loads(f.metadata()["config"])

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        return self.weight_loader.load(path, sd_ops, device)


def in_process_checkpoint_loader() -> SafetensorsModelStateDictLoader:
    """For models a training process loads while it holds the DiT mapped from the same checkpoint (sampling: text
    encoder, VAE, audio decoder and vocoder): plain file reads, never a second memory map of the file."""
    return SafetensorsModelStateDictLoader(weight_loader=SafetensorsStateDictLoader(file_reads=True))
