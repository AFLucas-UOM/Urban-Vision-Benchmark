"""Dataset, transforms, class weights, and split utilities over the crop manifest.

Missing labels are encoded as -1 in the target tensor; the loss masks them out.
Crops are letterboxed to a square (aspect-preserving, grey fill) before resizing
so that sign shape is not distorted, which matters for the shape head.
"""

import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import logging

log = logging.getLogger("mtsd_attr")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MISSING_LABEL = -1


class PadToSquare:
    """Pad a PIL image to a square canvas with the image centred.

    Preserves aspect ratio so circles stay circles after the subsequent resize.
    """

    def __init__(self, fill=(114, 114, 114)):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), self.fill)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


def build_transforms(image_size, train, aug_cfg):
    """Build the preprocessing pipeline for one split.

    Args:
        image_size: Target square side in pixels.
        train: If True, include augmentation from aug_cfg.
        aug_cfg: The training.augmentation config block.

    Returns:
        A torchvision Compose.
    """
    steps = [PadToSquare()]
    if train:
        if aug_cfg.get("rotation_degrees"):
            steps.append(transforms.RandomRotation(
                aug_cfg["rotation_degrees"], fill=(114, 114, 114)))
        scale = tuple(aug_cfg.get("random_resized_crop_scale", (0.8, 1.0)))
        steps.append(transforms.RandomResizedCrop(
            image_size, scale=scale, ratio=(0.9, 1.11), antialias=True))
        if aug_cfg.get("horizontal_flip"):
            steps.append(transforms.RandomHorizontalFlip())
        if aug_cfg.get("color_jitter"):
            steps.append(transforms.ColorJitter(*aug_cfg["color_jitter"]))
    else:
        steps.append(transforms.Resize((image_size, image_size), antialias=True))
    steps += [transforms.ToTensor(),
              transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class AttributeCropDataset(Dataset):
    """Sign-crop dataset yielding (image_tensor, target_vector, crop_id).

    The target vector holds one class index per attribute, in the order the
    attributes appear in the config, with MISSING_LABEL (-1) where the crop has
    no label for that attribute.
    """

    def __init__(self, records, attributes, subproject_root, transform):
        self.records = records
        self.attributes = list(attributes)
        self.class_to_idx = {
            attr: {c: i for i, c in enumerate(spec["classes"])}
            for attr, spec in attributes.items()
        }
        self.root = Path(subproject_root)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        path = self.root / rec["crop_path"].replace("\\", "/")
        with Image.open(path) as img:
            tensor = self.transform(img.convert("RGB"))
        target = torch.full((len(self.attributes),), MISSING_LABEL, dtype=torch.long)
        for i, attr in enumerate(self.attributes):
            value = rec["labels"].get(attr)
            if value is not None:
                target[i] = self.class_to_idx[attr][value]
        return tensor, target, rec["crop_id"]


def load_split_records(manifest_path):
    """Load the manifest and return its records grouped by split.

    Returns:
        (records_by_split dict, manifest dict)
    """
    with open(manifest_path, encoding="utf-8-sig") as f:
        manifest = json.load(f)
    by_split = {}
    for rec in manifest["records"]:
        by_split.setdefault(rec["split"], []).append(rec)
    return by_split, manifest


def subsample_records(records, max_samples, seed):
    """Deterministically subsample records for smoke-test runs."""
    if len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    return rng.sample(records, max_samples)


def compute_class_weights(records, attributes):
    """Compute inverse-frequency class weights per attribute from given records.

    Weight for class c is N_labelled / (n_classes * count_c); classes absent
    from the records get weight 0 (they cannot be learned) with a warning.

    Returns:
        dict attr -> FloatTensor of per-class weights.
    """
    weights = {}
    for attr, spec in attributes.items():
        classes = spec["classes"]
        counts = Counter(
            rec["labels"][attr] for rec in records
            if rec["labels"].get(attr) is not None
        )
        total = sum(counts.values())
        w = []
        for c in classes:
            n = counts.get(c, 0)
            if n == 0:
                log.warning("Attribute %s: class %r has no training samples; "
                            "its loss weight is set to 0", attr, c)
                w.append(0.0)
            else:
                w.append(total / (len(classes) * n))
        weights[attr] = torch.tensor(w, dtype=torch.float32)
    return weights
