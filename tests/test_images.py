import io

import pytest
from PIL import Image

from fruit_backend.errors import InvalidImageError
from fruit_backend.images import MAX_BYTES, prepare_image


def encoded(image, format="PNG", **kwargs):
    output = io.BytesIO()
    image.save(output, format=format, **kwargs)
    return output.getvalue()


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP"])
def test_supported_inputs_are_metadata_free_jpeg(format):
    prepared = prepare_image(encoded(Image.new("RGB", (30, 20), "red"), format))
    with Image.open(io.BytesIO(prepared.data)) as result:
        assert result.format == "JPEG"
        assert result.mode == "RGB"
        assert result.size == (30, 20)  # no upscaling
        assert not result.getexif()


def test_resizing_preserves_aspect_ratio():
    result = prepare_image(encoded(Image.new("RGB", (3000, 1500))))
    assert (result.width, result.height) == (1280, 640)


def test_exif_rotation_and_removal():
    source = Image.new("RGB", (80, 40), "red")
    exif = Image.Exif()
    exif[274] = 6
    exif[270] = "Private description"
    result = prepare_image(encoded(source, "JPEG", exif=exif))
    assert (result.width, result.height) == (40, 80)
    assert b"Private description" not in result.data


def test_transparency_is_composited_on_white():
    prepared = prepare_image(encoded(Image.new("RGBA", (20, 20), (255, 0, 0, 0))))
    with Image.open(io.BytesIO(prepared.data)) as result:
        assert result.getpixel((10, 10)) == (255, 255, 255)


@pytest.mark.parametrize("data", [b"", b"not a photo", b"x" * (MAX_BYTES + 1)])
def test_invalid_bytes_are_rejected(data):
    with pytest.raises(InvalidImageError):
        prepare_image(data)


def test_extension_does_not_bypass_format_check():
    with pytest.raises(InvalidImageError, match="JPEG, PNG, or WebP"):
        prepare_image(encoded(Image.new("RGB", (20, 20)), "BMP"))


def test_pixel_limit_is_checked_before_decode():
    data = encoded(Image.new("1", (5001, 5000)))
    with pytest.raises(InvalidImageError, match="megapixel"):
        prepare_image(data)


def test_animated_png_is_rejected():
    first = Image.new("RGB", (20, 20), "red")
    data = encoded(first, save_all=True, append_images=[Image.new("RGB", (20, 20), "blue")], duration=100)
    with pytest.raises(InvalidImageError, match="Animated"):
        prepare_image(data)


def test_truncated_image_is_rejected(image_bytes):
    with pytest.raises(InvalidImageError):
        prepare_image(image_bytes[:40])


def test_hashes_are_stable_and_tied_to_content(image_bytes):
    first = prepare_image(image_bytes)
    assert first == prepare_image(image_bytes)
    changed = prepare_image(encoded(Image.new("RGB", (80, 40), "green")))
    assert changed.original_sha256 != first.original_sha256
    assert changed.prepared_sha256 != first.prepared_sha256
