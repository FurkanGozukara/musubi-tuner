"""Training images in sRGB.

The trainers read pixel values as sRGB, but phone photos often store another colour space (Display P3) and print files
CMYK, described by an embedded ICC profile that colour-managed viewers honour. Read as plain sRGB such a photo trains
duller, shifted colours (skin tones included) than the user sees. Converting with the embedded profile (LittleCMS
through PIL.ImageCms) gives the trainer the colours on the screen; kohya-ss/sd-scripts #2393 proposes the same.
Images without a profile, or already tagged sRGB, are used as they are.
"""

from __future__ import annotations

import io
import logging
import os
from functools import lru_cache
from typing import Union

from PIL import Image

logger = logging.getLogger(__name__)

try:
    from PIL import ImageCms
except ImportError:  # Pillow built without LittleCMS: the pixels are used as they are
    ImageCms = None


@lru_cache(maxsize=1)
def _srgb_profile():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def to_srgb(image: Image.Image, name: str = "") -> Image.Image:
    """``image`` converted to sRGB with its embedded ICC profile (alpha kept); unchanged without a profile, with an sRGB
    one, or when the conversion fails (a warning names the file)."""
    icc = image.info.get("icc_profile")
    if not icc or ImageCms is None:
        return image
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if image.mode in ("RGB", "RGBA") and "srgb" in (ImageCms.getProfileDescription(source) or "").lower():
            return image
        alpha = image.getchannel("A") if image.mode in ("RGBA", "LA", "PA") else None
        if image.mode in ("RGB", "CMYK"):
            pixels = image
        elif image.mode in ("L", "LA"):
            pixels = image.getchannel("L")  # a grey profile
        else:  # RGBA (alpha put back below), palette and other modes
            pixels = image.convert("RGB")
        converted = ImageCms.profileToProfile(pixels, source, _srgb_profile(), outputMode="RGB")
        if alpha is not None:
            converted.putalpha(alpha)
        return converted
    except Exception as exc:  # noqa: BLE001 - a damaged or unusual profile must not stop training
        logger.warning(f"Could not convert {name or 'an image'} from its colour profile to sRGB; using its pixels as they are: {exc}")
        return image


def open_rgb_image(path: Union[str, os.PathLike]) -> Image.Image:
    """The image at ``path`` in sRGB, as RGB or (with alpha) RGBA."""
    image = to_srgb(Image.open(path), os.fspath(path))
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    return image
