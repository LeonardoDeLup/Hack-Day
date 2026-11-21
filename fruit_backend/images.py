"""Bounded, metadata-free preparation of image bytes for vision inference."""

import hashlib
import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import InvalidImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_EDGE = 1280
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    width: int
    height: int
    original_sha256: str
    prepared_sha256: str


def prepare_image(image_bytes: bytes) -> PreparedImage:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise InvalidImageError("Provide nonempty image bytes.")
    if len(image_bytes) > MAX_BYTES:
        raise InvalidImageError("Image exceeds the 10 MiB upload limit.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise InvalidImageError("Use a JPEG, PNG, or WebP image.")
                if probe.width * probe.height > MAX_PIXELS:
                    raise InvalidImageError("Image exceeds the 25 megapixel limit.")
                if getattr(probe, "is_animated", False):
                    raise InvalidImageError("Animated images are not supported; upload a still image.")
                probe.verify()

            with Image.open(io.BytesIO(image_bytes)) as source:
                source.load()
                upright = ImageOps.exif_transpose(source)
                if upright.mode in {"RGBA", "LA"} or "transparency" in upright.info:
                    rgba = upright.convert("RGBA")
                    rgb = Image.new("RGB", rgba.size, "white")
                    rgb.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    rgb = upright.convert("RGB")
                rgb.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
                # A fresh image ensures EXIF/ICC/text metadata is not forwarded.
                clean = Image.new("RGB", rgb.size)
                clean.paste(rgb)
                output = io.BytesIO()
                clean.save(output, format="JPEG", quality=90)
                data = output.getvalue()
                width, height = clean.size
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImageError("Cannot decode this image. Upload an intact JPEG, PNG, or WebP.") from exc

    return PreparedImage(
        data=data, width=width, height=height,
        original_sha256=hashlib.sha256(image_bytes).hexdigest(),
        prepared_sha256=hashlib.sha256(data).hexdigest(),
    )
