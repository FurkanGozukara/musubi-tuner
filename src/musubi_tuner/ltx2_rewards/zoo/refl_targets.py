"""Differentiable rewards registered for the ``--rl_loss refl`` backend.

Unlike the black-box zoo rewards (which return detached Python floats scored on decoded media), a
differentiable reward implements ``score_grad(samples) -> (Tensor[N], info)`` returning a grad-carrying
per-sample reward tensor. The refl backend backprops through one final denoising step into the LoRA.

Two templates:

* ``latent_energy`` is the runnable latent-space example and needs no VAE decode.
* ``pixel_sharpness`` defines the pixel-space reward interface, but the ReFL driver currently rejects
  it because in-graph LTX-2 video decoding has not been implemented or validated there.

Both also implement detached ``score`` methods. A custom ReFL reward is currently runnable only when
its ``needs`` set does not request decoded video or audio.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from ..registry import BaseReward, register_reward

_LAPLACIAN = ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))


def _highpass_energy_2d(x, num_slices: int = 8):
    """Mean squared 3x3-Laplacian response over up to ``num_slices`` (channel*frame) planes.

    ``x`` is ``[P, H, W]`` (P planes). Returns a 0-d tensor that KEEPS the autograd graph of ``x``.
    """
    import torch
    import torch.nn.functional as F

    p, h, w = x.shape
    if num_slices and p > num_slices:
        idx = torch.linspace(0, p - 1, num_slices, device=x.device).round().long()
        x = x.index_select(0, idx)
    kernel = torch.tensor(_LAPLACIAN, device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    hp = F.conv2d(x.unsqueeze(1), kernel, padding=1)  # [P,1,H,W]
    return hp.pow(2).mean()


def _as_planes(t):
    """Collapse a latent ``[C,F,H,W]`` / ``[1,C,F,H,W]`` or video ``[C,T,H,W]`` / ``[1,C,T,H,W]``
    to ``[P, H, W]`` planes (P = leading dims flattened), preserving grad."""
    if t.dim() == 5:
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"expected 4D [C,F,H,W] latent/video (or 5D with batch 1), got {tuple(t.shape)}")
    c, f, h, w = t.shape
    return t.reshape(c * f, h, w)


@register_reward("latent_energy")
class LatentEnergyReward(BaseReward):
    """Latent-space high-pass energy — differentiable ReFL template, NO decode."""

    kind = "differentiable"
    route = "video"
    needs = frozenset()  # scored on the latent directly; the generator decodes nothing for it

    def __init__(self) -> None:
        self._num_slices = 8

    def setup(self, device, *, num_slices=8, **_ignored) -> None:
        self._num_slices = int(num_slices)

    def _energy(self, sample: dict):
        z = sample.get("video_x0")
        if z is None:
            raise KeyError("latent_energy needs sample['video_x0'] (the denoised latent); none provided")
        return _highpass_energy_2d(_as_planes(z).float(), self._num_slices)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        return [float(self._energy(s).detach()) for s in samples], {"reward": "latent_energy"}

    def score_grad(self, samples: List[dict]) -> Tuple[Any, dict]:
        import torch

        return torch.stack([self._energy(s) for s in samples]), {"reward": "latent_energy"}


@register_reward("pixel_sharpness")
class PixelSharpnessReward(BaseReward):
    """Pixel-space interface example; unavailable in ReFL until in-graph decode is implemented."""

    kind = "differentiable"
    route = "video"
    needs = frozenset({"video"})  # requires the ReFL loop to decode sample['video_x0'] -> pixels with grad

    def __init__(self) -> None:
        self._num_frames = 5

    def setup(self, device, *, num_frames=5, **_ignored) -> None:
        self._num_frames = int(num_frames)

    def _sharp(self, sample: dict):
        v = sample.get("video")
        if v is None:
            raise KeyError("pixel_sharpness needs a grad-carrying sample['video'] (decoded frames); none provided")
        return _highpass_energy_2d(_as_planes(v).float(), self._num_frames)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        return [float(self._sharp(s).detach()) for s in samples], {"reward": "pixel_sharpness"}

    def score_grad(self, samples: List[dict]) -> Tuple[Any, dict]:
        import torch

        return torch.stack([self._sharp(s) for s in samples]), {"reward": "pixel_sharpness"}
