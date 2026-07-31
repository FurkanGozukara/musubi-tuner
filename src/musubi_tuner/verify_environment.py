"""Verify that the installed PyTorch stack can run Musubi Tuner."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Callable


TORCHVISION_COMPATIBILITY = {
    (2, 5): (0, 20),
    (2, 6): (0, 21),
    (2, 7): (0, 22),
    (2, 8): (0, 23),
    (2, 9): (0, 24),
    (2, 10): (0, 25),
    (2, 11): (0, 26),
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def _version_pair(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _wheel_cuda_tag(version: str) -> str | None:
    match = re.search(r"\+(cu\d+)(?:[.-]|$)", version)
    return match.group(1) if match else None


def _runtime_cuda_tag(version: str | None) -> str | None:
    if not version:
        return None
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return None
    return f"cu{match.group(1)}{match.group(2)}"


def _run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name=name, status="ok", detail=check())
    except Exception as exc:
        return CheckResult(name=name, status="error", detail=f"{type(exc).__name__}: {exc}")


def _import_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def verify_environment(*, require_cuda: bool = False, expected_cuda: str | None = None) -> dict[str, Any]:
    checks: list[CheckResult] = []
    versions: dict[str, str] = {}

    def import_core_stack() -> str:
        for module_name in ("torch", "torchvision", "torchaudio"):
            versions[module_name] = _import_version(module_name)
        return ", ".join(f"{name}={version}" for name, version in versions.items())

    checks.append(_run_check("pytorch_imports", import_core_stack))
    if checks[-1].status == "error":
        return _build_report(checks, versions, {}, require_cuda, expected_cuda)

    import torch
    import torchaudio
    import torchvision

    def compatible_versions() -> str:
        torch_pair = _version_pair(versions["torch"])
        vision_pair = _version_pair(versions["torchvision"])
        audio_pair = _version_pair(versions["torchaudio"])
        expected_vision = TORCHVISION_COMPATIBILITY.get(torch_pair)
        if expected_vision is None:
            raise RuntimeError(
                f"torch {versions['torch']} is outside the verified compatibility table; "
                "update this verifier after qualifying its matching torchvision and torchaudio releases"
            )
        if vision_pair != expected_vision:
            raise RuntimeError(
                f"torch {versions['torch']} requires torchvision {expected_vision[0]}.{expected_vision[1]}.x, "
                f"but {versions['torchvision']} is installed"
            )
        if audio_pair != torch_pair:
            raise RuntimeError(
                f"torch {versions['torch']} requires matching torchaudio {torch_pair[0]}.{torch_pair[1]}.x, "
                f"but {versions['torchaudio']} is installed"
            )
        return "torch, torchvision, and torchaudio release lines match"

    checks.append(_run_check("pytorch_versions", compatible_versions))

    runtime_cuda = str(torch.version.cuda) if torch.version.cuda is not None else None
    runtime_tag = _runtime_cuda_tag(runtime_cuda)
    cuda_available = bool(torch.cuda.is_available())
    cuda_info: dict[str, Any] = {
        "available": cuda_available,
        "runtime": runtime_cuda,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [],
    }
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_info["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": f"{properties.major}.{properties.minor}",
                    "total_memory_bytes": properties.total_memory,
                }
            )

    def matching_cuda_builds() -> str:
        wheel_tags = {name: tag for name, version in versions.items() if (tag := _wheel_cuda_tag(version)) is not None}
        if wheel_tags and len(set(wheel_tags.values())) != 1:
            raise RuntimeError(f"PyTorch wheel CUDA tags differ: {wheel_tags}")
        if wheel_tags and runtime_tag and next(iter(wheel_tags.values())) != runtime_tag:
            raise RuntimeError(f"wheel tag {next(iter(wheel_tags.values()))} does not match torch CUDA runtime {runtime_tag}")
        if expected_cuda is not None and runtime_tag != expected_cuda:
            raise RuntimeError(f"expected {expected_cuda}, but torch reports {runtime_tag or 'a CPU build'}")
        return f"CUDA build tags match ({runtime_tag or 'cpu'})"

    checks.append(_run_check("cuda_build", matching_cuda_builds))

    def cuda_runtime() -> str:
        if not cuda_available:
            if require_cuda:
                raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
            return "CUDA is not available; CPU-only checks completed"
        left = torch.randn((32, 32), device="cuda", dtype=torch.float32)
        result = left @ left.T
        if not bool(torch.isfinite(result).all().item()):
            raise RuntimeError("CUDA matrix multiplication produced non-finite values")
        torch.cuda.synchronize()
        return f"CUDA kernel passed on {torch.cuda.get_device_name(0)}"

    checks.append(_run_check("cuda_runtime", cuda_runtime))

    def torchvision_ops() -> str:
        if not torchvision.extension._has_ops():
            raise RuntimeError("TorchVision compiled operators are unavailable")
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.5, 0.5, 2.5, 2.5]])
        scores = torch.tensor([0.9, 0.8])
        kept = torchvision.ops.nms(boxes, scores, 0.5)
        if kept.numel() != 2:
            raise RuntimeError(f"unexpected NMS result: {kept.tolist()}")
        return "compiled NMS operator passed"

    checks.append(_run_check("torchvision_ops", torchvision_ops))

    def torchaudio_ops() -> str:
        waveform = torch.linspace(-1.0, 1.0, 1600).unsqueeze(0)
        resampled = torchaudio.functional.resample(waveform, 16000, 24000)
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=256, n_mels=32)(resampled)
        if resampled.shape[-1] != 2400 or mel.shape[-2] != 32:
            raise RuntimeError(f"unexpected audio shapes: resampled={tuple(resampled.shape)}, mel={tuple(mel.shape)}")
        if not bool(torch.isfinite(mel).all().item()):
            raise RuntimeError("TorchAudio mel transform produced non-finite values")
        return "resampling and mel-spectrogram operations passed"

    checks.append(_run_check("torchaudio_ops", torchaudio_ops))

    def trainer_imports() -> str:
        modules = ("accelerate", "av", "diffusers", "safetensors", "transformers")
        for module_name in modules:
            versions[module_name] = _import_version(module_name)
        return ", ".join(f"{name}={versions[name]}" for name in modules)

    checks.append(_run_check("trainer_imports", trainer_imports))

    optional_capabilities: dict[str, str] = {}
    for module_name in ("bitsandbytes", "flash_attn", "triton"):
        try:
            optional_capabilities[module_name] = _import_version(module_name)
        except Exception as exc:
            optional_capabilities[module_name] = f"unavailable ({type(exc).__name__}: {exc})"

    return _build_report(checks, versions, cuda_info, require_cuda, expected_cuda, optional_capabilities)


def _build_report(
    checks: list[CheckResult],
    versions: dict[str, str],
    cuda: dict[str, Any],
    require_cuda: bool,
    expected_cuda: str | None,
    optional_capabilities: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "ok": all(check.status == "ok" for check in checks),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requirements": {"require_cuda": require_cuda, "expected_cuda": expected_cuda},
        "versions": versions,
        "cuda": cuda,
        "optional_capabilities": optional_capabilities or {},
        "checks": [asdict(check) for check in checks],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Fail unless a CUDA device can execute a torch kernel.")
    parser.add_argument(
        "--expected-cuda",
        choices=("cu124", "cu128", "cu130"),
        help="Fail unless the installed PyTorch stack uses this CUDA release.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = verify_environment(require_cuda=args.require_cuda, expected_cuda=args.expected_cuda)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for check in report["checks"]:
            marker = "PASS" if check["status"] == "ok" else "FAIL"
            print(f"[{marker}] {check['name']}: {check['detail']}")
        for name, detail in report["optional_capabilities"].items():
            print(f"[INFO] {name}: {detail}")
        print("Environment is compatible." if report["ok"] else "Environment is not compatible.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
