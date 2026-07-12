"""Executable example for ``python -m musubi_tuner.torch_compile_toolchain.example``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402

from . import (  # noqa: E402
    compile_module_callable,
    ensure_compile_environment,
)


class DemoModel(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(value) * 2


def main() -> int:
    status = ensure_compile_environment(project_root=PROJECT_ROOT)
    print(json.dumps(status.as_dict(), indent=2))
    if not status.ok:
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DemoModel().to(device)
    compile_result = compile_module_callable(
        model,
        "forward",
        project_root=PROJECT_ROOT,
        on_status=lambda state, detail: print(f"{state}: {detail}"),
    )
    output = model(torch.arange(8, device=device, dtype=torch.float32))
    print(output)
    print(
        json.dumps(
            {
                "compiled": compile_result.compiled,
                "verified": compile_result.verified,
                "detail": compile_result.detail,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
