import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musubi_tuner.hfato import HFATOConfig, degrade_latents, hfato_x0_loss  # noqa: E402


def test_hfato_defaults_match_vibe_stage2_recipe():
    config = HFATOConfig()

    assert config.scale_factor == 0.5
    assert config.interpolation == "trilinear"
    assert config.probability == 1.0


def test_degrade_latents_default_matches_vibe_trilinear_formula():
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 4, 8, 10)

    actual = degrade_latents(latents)
    down = F.interpolate(
        latents,
        scale_factor=(1.0, 0.5, 0.5),
        mode="trilinear",
        align_corners=False,
    )
    expected = F.interpolate(
        down,
        size=(4, 8, 10),
        mode="trilinear",
        align_corners=False,
    )

    assert torch.allclose(actual, expected)


def test_hfato_x0_loss_matches_vibe_reconstruction_formula():
    torch.manual_seed(1)
    clean = torch.randn(2, 3, 4, 8, 10)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.25, 0.75])
    degraded = degrade_latents(clean)
    noisy = (1.0 - sigma.view(-1, 1, 1, 1, 1)) * degraded + sigma.view(-1, 1, 1, 1, 1) * noise
    flow_pred = torch.randn_like(clean)

    actual = hfato_x0_loss(flow_pred, noisy, clean, sigma)
    expected = F.mse_loss((noisy - sigma.view(-1, 1, 1, 1, 1) * flow_pred).float(), clean.float())

    assert torch.allclose(actual, expected)
