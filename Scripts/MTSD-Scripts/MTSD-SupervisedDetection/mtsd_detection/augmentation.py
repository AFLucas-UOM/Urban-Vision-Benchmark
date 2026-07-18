from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .utils import sha256_file


def mild_motion_blur(image: Image.Image, kernel_size: int = 5,
                     blur_weight: float = 0.85) -> Image.Image:
    """Apply a short horizontal exposure trail without changing geometry."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer of at least 3")
    if not 0 <= blur_weight <= 1:
        raise ValueError("blur_weight must be between 0 and 1")
    import numpy as np
    values = np.asarray(image.convert("RGB"), dtype=np.float32)
    radius = kernel_size // 2
    padded = np.pad(values, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    blurred = sum(padded[:, offset:offset + values.shape[1]]
                  for offset in range(kernel_size)) / kernel_size
    result = np.clip(blur_weight * blurred + (1 - blur_weight) * values,
                     0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def augment_image(source: Path, destination: Path, out_name: str, copy_index: int,
                  global_seed: int, config: dict[str, Any]) -> dict[str, Any]:
    seed_material = f"{global_seed}:{out_name}:{copy_index}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    ops = config["ops"]
    applied: dict[str, Any] = {}
    with Image.open(source) as raw:
        # QA bboxes live in the EXIF-applied (displayed) coordinate space, and the
        # augmented copy is saved without EXIF metadata; transposing the pixels here
        # keeps the generated image aligned with its copied label file.
        image = ImageOps.exif_transpose(raw).convert("RGB")
        motion = ops.get("motion_blur", {})
        if motion and copy_index == int(motion.get("copy_index", 3)):
            kernel_size = int(motion.get("kernel_size", 5))
            blur_weight = float(motion.get("blur_weight", 0.85))
            image = mild_motion_blur(image, kernel_size, blur_weight)
            applied["motion_blur"] = {
                "direction": "horizontal", "kernel_size": kernel_size,
                "blur_weight": blur_weight,
            }
        else:
            for name, enhancer in (("brightness", ImageEnhance.Brightness),
                                   ("contrast", ImageEnhance.Contrast), ("color", ImageEnhance.Color)):
                factor = rng.uniform(float(ops[name]["min"]), float(ops[name]["max"]))
                image = enhancer(image).enhance(factor); applied[name] = round(factor, 6)
            blur = ops["gaussian_blur"]
            if rng.random() < float(blur["p"]):
                sigma = rng.uniform(0, float(blur["sigma_max"]))
                image = image.filter(ImageFilter.GaussianBlur(sigma)); applied["gaussian_blur"] = round(sigma, 6)
            noise = ops["gaussian_noise"]
            if rng.random() < float(noise["p"]):
                import numpy as np
                sigma = rng.uniform(0, float(noise["sigma_max"]))
                generator = np.random.default_rng(seed ^ 0xA5A5A5A5)
                values = np.asarray(image).astype(np.float32) / 255.0
                values = np.clip(values + generator.normal(0, sigma, values.shape), 0, 1)
                image = Image.fromarray((values * 255).astype("uint8")); applied["gaussian_noise"] = round(sigma, 6)
            gamma = rng.uniform(float(ops["gamma"]["min"]), float(ops["gamma"]["max"]))
            image = image.point([round(255 * ((i / 255) ** gamma)) for i in range(256)] * 3)
            applied["gamma"] = round(gamma, 6)
            jpeg = ops["jpeg_compression"]
            if rng.random() < float(jpeg["p"]):
                quality = rng.randint(int(jpeg["quality_min"]), int(jpeg["quality_max"]))
                buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0); image = Image.open(buffer).convert("RGB"); applied["jpeg_compression"] = quality
        save_kwargs: dict[str, Any] = {}
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 92
        image.save(destination, **save_kwargs)
    return {"copy_index": copy_index, "ops_applied": json.dumps(applied, sort_keys=True),
            "seed_material": seed_material, "generated_filename": destination.name,
            "generated_image_hash": sha256_file(destination)}
