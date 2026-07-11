from PIL import Image

from mtsd_detection.augmentation import augment_image

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
