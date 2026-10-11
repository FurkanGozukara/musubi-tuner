"""The Z-Image VAE mid-block attention runs in query chunks.

Its single 512-channel head has no fused SDPA kernel, so the math kernel built the full (H*W)^2 score matrix: latent
caching of one 1536x1536 image reserved 19.7 GB on an RTX A6000 (12.2 GB allocated), which kept the 1536 px Z-Image
Turbo presets off 10-16 GB GPUs. Query chunks give the same rows (relative difference 6e-6) with 8.3 GB reserved.
"""

import torch

from musubi_tuner.zimage import zimage_autoencoder


def test_chunked_attention_matches_whole_image_attention(monkeypatch):
    torch.manual_seed(0)
    attention = zimage_autoencoder.Attention(64, groups=8).eval()
    hidden = torch.randn(1, 64, 24, 20)  # 480 query rows
    monkeypatch.setattr(zimage_autoencoder, "ATTENTION_QUERY_CHUNK", 0)
    with torch.no_grad():
        whole = attention(hidden)
    monkeypatch.setattr(zimage_autoencoder, "ATTENTION_QUERY_CHUNK", 100)  # 5 chunks, the last one partial
    with torch.no_grad():
        chunked = attention(hidden)
    assert chunked.shape == whole.shape
    torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-5)
