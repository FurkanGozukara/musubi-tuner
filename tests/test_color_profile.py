"""Training images in sRGB (utils.color_profile): a photo with an embedded Display P3 profile is converted, sRGB-tagged
and untagged images are used as they are, alpha survives, and the dataset loaders go through it."""

import io
import struct

import numpy as np
import pytest
from PIL import Image

from musubi_tuner.utils import color_profile

pytest.importorskip("PIL.ImageCms")

# Display P3 colorants adapted to the D50 profile connection space, as in Apple's Display P3 profile
P3_COLORANTS = ((0.515102, 0.241196, -0.001053), (0.291965, 0.692245, 0.041885), (0.157153, 0.066561, 0.784073))
D50 = (0.964203, 1.0, 0.824905)


def _s15(value: float) -> bytes:
    return struct.pack(">i", int(round(value * 65536)))


def _xyz(values) -> bytes:
    return b"XYZ " + b"\0" * 4 + b"".join(_s15(v) for v in values)


def _curve(gamma: float) -> bytes:
    return b"curv" + b"\0" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(gamma * 256))) + b"\0\0"


def _desc(text: str) -> bytes:
    ascii_text = text.encode("ascii") + b"\0"
    return b"desc" + b"\0" * 4 + struct.pack(">I", len(ascii_text)) + ascii_text + b"\0" * 8 + b"\0" * 3 + b"\0" * 67


def matrix_profile(colorants=P3_COLORANTS, gamma=2.2, name="Display P3 (test)") -> bytes:
    """A minimal ICC v2 RGB matrix / TRC display profile."""
    tags = [(b"desc", _desc(name)), (b"wtpt", _xyz(D50)), (b"rXYZ", _xyz(colorants[0])), (b"gXYZ", _xyz(colorants[1])),
            (b"bXYZ", _xyz(colorants[2])), (b"rTRC", _curve(gamma)), (b"gTRC", _curve(gamma)), (b"bTRC", _curve(gamma)),
            (b"cprt", b"text" + b"\0" * 4 + b"test\0")]
    offset = 128 + 4 + 12 * len(tags)
    table, body = b"", b""
    for signature, data in tags:
        while (offset + len(body)) % 4:
            body += b"\0"
        table += signature + struct.pack(">II", offset + len(body), len(data))
        body += data
    size = 128 + 4 + len(table) + len(body)
    header = (struct.pack(">I", size) + b"lcms" + struct.pack(">I", 0x02100000) + b"mntrRGB XYZ " + b"\0" * 12 + b"acsp"
              + b"\0" * 24 + struct.pack(">I", 0) + _s15(D50[0]) + _s15(D50[1]) + _s15(D50[2]) + b"\0" * 48)
    assert len(header) == 128
    return header + struct.pack(">I", len(tags)) + table + body


def _saved(tmp_path, name, mode, color, icc=None):
    image = Image.new(mode, (8, 8), color)
    path = tmp_path / name
    image.save(path, icc_profile=icc) if icc else image.save(path)
    return path


def test_a_display_p3_photo_becomes_sRGB(tmp_path):
    path = _saved(tmp_path, "p3.png", "RGB", (200, 100, 60), matrix_profile())
    pixel = np.array(color_profile.open_rgb_image(path)).reshape(-1, 3)[0].astype(int)
    # the same numbers mean a more saturated colour in P3: in sRGB red rises and green / blue fall
    assert pixel[0] > 200 and pixel[1] < 100 and pixel[2] < 60, pixel


def test_srgb_tagged_and_untagged_images_stay_as_they_are(tmp_path):
    from PIL import ImageCms

    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    for path in (_saved(tmp_path, "srgb.png", "RGB", (200, 100, 60), srgb), _saved(tmp_path, "plain.png", "RGB", (200, 100, 60))):
        assert tuple(np.array(color_profile.open_rgb_image(path))[0, 0]) == (200, 100, 60)


def test_alpha_is_kept_and_modes_are_RGB_or_RGBA(tmp_path):
    rgba = color_profile.open_rgb_image(_saved(tmp_path, "p3a.png", "RGBA", (200, 100, 60, 77), matrix_profile()))
    assert rgba.mode == "RGBA" and np.array(rgba)[0, 0, 3] == 77
    assert color_profile.open_rgb_image(_saved(tmp_path, "grey.png", "L", 128)).mode == "RGB"


def test_a_broken_profile_leaves_the_pixels(tmp_path, caplog):
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    image.info["icc_profile"] = b"not a profile"
    assert color_profile.to_srgb(image, "broken.png") is image
    assert "broken.png" in caplog.text


def test_dataset_loaders_use_it():
    """Every training image / reference loader of the dataset modules goes through open_rgb_image."""
    import inspect

    from musubi_tuner.dataset import datasources, image_video_dataset, media_utils

    for module in (image_video_dataset, datasources, media_utils):
        source = inspect.getsource(module)
        assert "open_rgb_image(" in source
        assert 'Image.open(image_file).convert("RGB")' not in source
        assert "control = Image.open(control_path)" not in source
