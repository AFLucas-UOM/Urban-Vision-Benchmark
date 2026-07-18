import json

from PIL import Image
import numpy as np

from mtsd_detection.augmentation import augment_image, mild_motion_blur

IDENTITY_OPS = {
    "brightness": {"min": 1, "max": 1}, "contrast": {"min": 1, "max": 1},
    "color": {"min": 1, "max": 1}, "gaussian_blur": {"p": 0, "sigma_max": 0},
    "gaussian_noise": {"p": 0, "sigma_max": 0},
    "jpeg_compression": {"p": 0, "quality_min": 90, "quality_max": 90},
    "gamma": {"min": 1, "max": 1},
}


def _exif_source(tmp_path, orientation):
    """A 40x20 JPEG whose top-left 8x8 block is red, tagged with the given EXIF orientation."""
    source = tmp_path / f"exif{orientation}.jpg"
    image = Image.new("RGB", (40, 20), (0, 0, 255))
    for x in range(8):
        for y in range(8):
            image.putpixel((x, y), (255, 0, 0))
    exif = Image.Exif()
    exif[274] = orientation  # 274 = Orientation
    image.save(source, quality=95, exif=exif)
    return source


def test_augmented_copy_applies_exif_orientation_6(tmp_path):
    # QA labels are stored in the EXIF-applied (displayed) coordinate space, so a
    # raw 40x20 image tagged Orientation=6 (90 degrees CW on display) must come out
    # as 20x40 pixels; without exif_transpose the copy would be rotated vs its label.
    source = _exif_source(tmp_path, orientation=6)
    destination = tmp_path / "exif6_aug1.png"  # png: lossless, no EXIF of its own
    info = augment_image(source, destination, "exif6.jpg", 1, 42,
                         {"ops": IDENTITY_OPS})
    assert info["generated_filename"] == destination.name
    with Image.open(destination) as out:
        assert out.size == (20, 40)
        # 90 CW rotation sends the raw top-left marker to the displayed top-right.
        r, g, b = out.getpixel((17, 2))
        assert r > 150 and b < 100, f"expected red marker at top-right, got {(r, g, b)}"
        r, g, b = out.getpixel((2, 37))
        assert b > 150 and r < 100, f"expected blue background at bottom-left, got {(r, g, b)}"


def test_augmented_copy_untagged_image_unchanged_dimensions(tmp_path):
    source = tmp_path / "plain.jpg"
    Image.new("RGB", (40, 20), (0, 0, 255)).save(source, quality=95)
    destination = tmp_path / "plain_aug1.png"
    augment_image(source, destination, "plain.jpg", 1, 42, {"ops": IDENTITY_OPS})
    with Image.open(destination) as out:
        assert out.size == (40, 20)


def test_mild_motion_blur_preserves_geometry_and_spreads_horizontally():
    image = Image.new("RGB", (21, 11), "black")
    image.putpixel((10, 5), (255, 255, 255))
    output = mild_motion_blur(image, kernel_size=5)
    values = np.asarray(output)
    assert output.size == image.size
    assert values[5, 8, 0] > 0
    assert values[5, 12, 0] > 0
    assert values[3, 10, 0] == 0


def test_motion_blur_rejects_even_kernel():
    image = Image.new("RGB", (10, 10), "black")
    try:
        mild_motion_blur(image, kernel_size=4)
    except ValueError as exc:
        assert "odd integer" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_aug3_uses_motion_blur_recipe(tmp_path):
    source = tmp_path / "source.png"
    image = Image.new("RGB", (21, 11), "black")
    image.putpixel((10, 5), (255, 255, 255))
    image.save(source)
    destination = tmp_path / "source_aug3.png"
    ops = {**IDENTITY_OPS, "motion_blur": {
        "copy_index": 3, "direction": "horizontal",
        "kernel_size": 9, "blur_weight": 0.85,
    }}
    info = augment_image(source, destination, source.name, 3, 42, {"ops": ops})
    applied = json.loads(info["ops_applied"])
    assert set(applied) == {"motion_blur"}
    assert applied["motion_blur"]["kernel_size"] == 9
    with Image.open(destination) as output:
        values = np.asarray(output)
    assert values[5, 6, 0] > 0
    assert values[5, 14, 0] > 0
