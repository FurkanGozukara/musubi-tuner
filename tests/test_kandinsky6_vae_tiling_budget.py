"""Kandinsky 6 latent caching: --vae_tiling_memory_gb plans smaller encode tiles.

The VAE sizes its tiles from the free GPU memory, so on any card a 512x288 clip was encoded untiled
(6.7 GB while caching), which kept the 8 GB and 6 GB preset tiers out of reach although training fit.
"""

import torch

from musubi_tuner.kandinsky6.runtime.core.components.hunyuan_vae import AutoencoderKLHunyuanVideo


def _tiny_vae():
    torch.manual_seed(0)
    return AutoencoderKLHunyuanVideo(block_out_channels=(8, 8, 8, 8), norm_num_groups=4, layers_per_block=1, mid_block_add_attention=False).eval()


def test_budget_plans_smaller_tiles_and_still_covers_the_clip():
    vae = _tiny_vae()
    shape = (1, 3, 49, 288, 512)
    assert vae.get_enc_optimal_tiling(shape, device="cpu")[0] == (1, 17, 288, 512)  # untiled without a budget

    vae.tiling_memory_budget = 3 * 1024**3
    tile, _stride = vae.get_enc_optimal_tiling(shape, device="cpu")
    assert tile[2] * tile[3] < 288 * 512 / 2

    with torch.no_grad():
        latent = vae.encode(torch.randn(shape)).latent_dist.mean
    assert tuple(latent.shape[2:]) == (13, 36, 64)
