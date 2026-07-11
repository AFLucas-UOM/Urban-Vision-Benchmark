from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .utils import sha256_file


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
        save_kwargs: dict[str, Any] = {}
        if rng.random() < float(jpeg["p"]):
            quality = rng.randint(int(jpeg["quality_min"]), int(jpeg["quality_max"]))
            buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0); image = Image.open(buffer).convert("RGB"); applied["jpeg_compression"] = quality
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 92
        image.save(destination, **save_kwargs)
    return {"copy_index": copy_index, "ops_applied": json.dumps(applied, sort_keys=True),
            "seed_material": seed_material, "generated_filename": destination.name,
            "generated_image_hash": sha256_file(destination)}

