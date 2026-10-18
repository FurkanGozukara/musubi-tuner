"""Sample references of FLUX.2 / FLUX.3 (flux2_utils.preprocess_control_image): limit_size caps the pixel count as BFL's
reference does (2024 ** 2 for one reference, 1024 ** 2 for several), keeping the aspect ratio, with both sides multiples
of 16. Resizing to (2024, 2024) itself squared a 12 MP phone photo and gave a 253-wide latent that the 2x2 patching
cannot split."""

import pytest
from PIL import Image

from musubi_tuner.flux_2 import flux2_utils


@pytest.mark.parametrize(
    ("size", "limit", "expected"),
    [
        ((4000, 3000), (2024, 2024), (2336, 1744)),  # one 12 MP reference: about 4.1 MP at 4:3
        ((3000, 3000), (2024, 2024), (2016, 2016)),  # square: 2016, not 2024 (latent 252, even)
        ((2000, 1000), (1024, 1024), (1440, 720)),  # one of several references: about 1 MP at 2:1
    ],
)
def test_large_references_keep_their_aspect_within_the_pixel_cap(tmp_path, size, limit, expected):
    path = tmp_path / "reference.png"
    Image.new("RGB", size, "red").save(path)
    tensor, image, _ = flux2_utils.preprocess_control_image(str(path), limit)
    height, width = tensor.shape[-2:]
    assert (width, height) == expected and image.shape[:2] == (height, width)
    assert width % 16 == 0 and height % 16 == 0 and width * height <= limit[0] * limit[1]
    assert (width // 8) % 2 == 0 and (height // 8) % 2 == 0  # the latent splits into 2x2 patches


def test_small_references_keep_their_size_on_the_16_px_grid(tmp_path):
    path = tmp_path / "reference.png"
    Image.new("RGB", (1000, 750), "red").save(path)
    tensor, _, _ = flux2_utils.preprocess_control_image(str(path), (2024, 2024))
    height, width = tensor.shape[-2:]
    assert width % 16 == 0 and height % 16 == 0 and abs(width / height - 4 / 3) < 0.03
