"""Size-ablation report: grouping by size/adaptation, distinct output names."""

import csv
import json

import pytest

from mtsd_attr.evaluate import write_comparison_report, write_size_ablation_report

ATTRS = ["view_angle", "mounting"]


def _metrics_payload(variant, meta, score, adaptation, params=(1000, 10)):
    attributes = {}
    for attr, classes in (("view_angle", ["Front", "Back", "Side"]),
                          ("mounting", ["Pole-Mounted", "Wall-Mounted"])):
        attributes[attr] = {
            "n": 10, "accuracy": score, "macro_f1": score,
            "classes": classes,
            "per_class_f1": {c: score for c in classes},
            "support": {c: 5 for c in classes},
            "confusion_matrix": [[1] * len(classes)] * len(classes),
        }
    return {
        "run_id": f"{variant}-20260713-000000",
        "variant": variant,
        "split": "test",
        "smoke_test": False,
        "mean_macro_f1": score,
        "run_info": {
            "adaptation": adaptation,
            "best_epoch": 3,
            "stopped_epoch": 5,
            "stop_reason": "early_stopping",
            "val_mean_macro_f1": score,
            "parameters": {"total": params[0], "trainable": params[1],
                           "trainable_pct": 1.0},
            "variant_meta": meta,
            "backbone_meta": {"loaded_backend": "test", "loaded_version": "x"},
            "training_meta": {"physical_batch_size": 16,
                              "gradient_accumulation_steps": 2,
                              "effective_batch_size": 32},
            "train_duration_s": 123.4,
            "test_eval_duration_s": 1.0,
            "test_images_per_s": 10.0,
        },
        "attributes": attributes,
    }


def _meta(variant, family, arch, size, adaptation):
    return {"variant": variant, "family": family, "architecture": arch,
            "model_size": size, "adaptation": adaptation, "model_id": "id",
            "resolution": 224, "display_name": f"{family} {arch} {adaptation}",
            "legacy": False}


@pytest.fixture
def fake_cfg(tmp_path):
    variants = {
        "dinov3_vitb_frozen": ("DINOv3", "ViT-B/16", "base", "frozen", 0.80),
        "dinov3_vitl_frozen": ("DINOv3", "ViT-L/16", "large", "frozen", 0.85),
        "dinov3_vitb_lora": ("DINOv3", "ViT-B/16", "base", "lora", 0.88),
        "convnext_base_finetune": ("ConvNeXt", "ConvNeXt-Base", "base",
                                   "finetune", 0.86),
        "lingbot_vitb_frozen": ("LingBot-Vision", "ViT-B/16", "base",
                                "frozen", 0.83),
    }
    models = {}
    for variant, (family, arch, size, adaptation, _) in variants.items():
        models[variant] = {"backbone": "x", "family": family,
                           "architecture": arch, "model_size": size,
                           "adaptation": adaptation, "image_size": 224}
    cfg = {
        "paths": {"metrics_dir": tmp_path / "metrics",
                  "reports_dir": tmp_path / "reports"},
        "attributes": {"view_angle": {"classes": ["Front", "Back", "Side"]},
                       "mounting": {"classes": ["Pole-Mounted",
                                                "Wall-Mounted"]}},
        "models": models,
        "run_all": {"variants": list(models),
                    "profiles": {"size_ablation_all": list(variants)}},
    }
    for variant, (family, arch, size, adaptation, score) in variants.items():
        folder = tmp_path / "metrics" / variant
        folder.mkdir(parents=True)
        payload = _metrics_payload(
            variant, _meta(variant, family, arch, size, adaptation), score,
            adaptation)
        (folder / "test_metrics.json").write_text(json.dumps(payload),
                                                  encoding="utf-8")
    return cfg


def test_size_ablation_report_grouping(fake_cfg):
    md_path = write_size_ablation_report(fake_cfg)
    assert md_path.name == "size_ablation.md"
    text = md_path.read_text(encoding="utf-8")

    # Grouped sections exist and are labelled by adaptation.
    assert "## 1. Frozen linear probes, by family (adaptation: frozen)" in text
    assert "## 2. LoRA adaptation, by family (adaptation: lora)" in text
    assert "## 3. ConvNeXt full fine-tuning (adaptation: finetune)" in text
    assert "## 4. All frozen representation models" in text

    # The frozen DINOv3 section contains both sizes; sizes are explicit.
    frozen_dino = text.split("### DINOv3 (frozen)")[1].split("###")[0]
    assert "dinov3_vitb_frozen" in frozen_dino
    assert "dinov3_vitl_frozen" in frozen_dino
    assert "ViT-L/16" in frozen_dino
    # No adaptation mixing: the LoRA run is not in the frozen section.
    assert "dinov3_vitb_lora" not in frozen_dino
    lora_section = text.split("## 2.")[1].split("## 3.")[0]
    assert "dinov3_vitb_lora" in lora_section
    assert "dinov3_vitb_frozen" not in lora_section
    finetune_section = text.split("## 3.")[1].split("## 4.")[0]
    assert "convnext_base_finetune" in finetune_section

    # All-frozen ranking is sorted by test score (ViT-L 0.85 first).
    ranking = text.split("## 4.")[1]
    assert ranking.index("dinov3_vitl_frozen") \
        < ranking.index("lingbot_vitb_frozen") \
        < ranking.index("dinov3_vitb_frozen")

    csv_path = md_path.parent / "size_ablation.csv"
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    by_variant = {r["variant"]: r for r in rows}
    row = by_variant["dinov3_vitl_frozen"]
    assert row["family"] == "DINOv3" and row["model_size"] == "large"
    assert row["view_angle_macro_f1"] and row["mounting_accuracy"]
    assert row["train_duration_s"] == "123.4"
    assert row["loaded_backend"] == "test"


def test_size_ablation_does_not_touch_historical_report_names(fake_cfg):
    md_path = write_size_ablation_report(fake_cfg)
    reports = {p.name for p in fake_cfg["paths"]["reports_dir"].iterdir()}
    assert reports == {"size_ablation.md", "size_ablation.csv"}
    assert "None" not in md_path.read_text("utf-8")
    assert "None" not in (md_path.parent / "size_ablation.csv").read_text("utf-8")


def test_comparison_report_metadata_fallback_for_old_payloads(fake_cfg):
    # Strip variant_meta from one payload to simulate a historical run.
    folder = fake_cfg["paths"]["metrics_dir"] / "dinov3_vitb_frozen"
    payload = json.loads((folder / "test_metrics.json").read_text("utf-8"))
    payload["run_info"].pop("variant_meta")
    payload["run_info"].pop("backbone_meta")
    (folder / "test_metrics.json").write_text(json.dumps(payload), "utf-8")

    md_path = write_comparison_report(fake_cfg)
    text = md_path.read_text(encoding="utf-8")
    # Family/size columns fall back to the config entry's metadata.
    assert "| dinov3_vitb_frozen | DINOv3 | ViT-B/16 | base | frozen |" in text
    csv_text = (md_path.parent / "comparison.csv").read_text("utf-8")
    assert "family" in csv_text.splitlines()[0]
    assert "None" not in text
    assert "None" not in csv_text


def test_reports_render_new_metrics_and_neutral_timing(fake_cfg):
    for folder in (fake_cfg["paths"]["metrics_dir"] / variant
                   for variant in fake_cfg["models"]):
        payload = json.loads((folder / "test_metrics.json").read_text("utf-8"))
        for metric in payload["attributes"].values():
            metric["macro_precision"] = metric["macro_f1"] - 0.01
            metric["macro_recall"] = metric["macro_f1"] - 0.02
            metric["per_class_precision"] = dict(metric["per_class_f1"])
            metric["per_class_recall"] = dict(metric["per_class_f1"])
            metric["macro_f1_ci"] = {"level": 0.95, "lower": 0.70,
                                      "upper": 0.90, "n_bootstrap": 20}
        payload["mean_macro_precision"] = payload["mean_macro_f1"] - 0.01
        payload["mean_macro_recall"] = payload["mean_macro_f1"] - 0.02
        payload["mean_macro_f1_ci"] = {"level": 0.95, "lower": 0.70,
                                        "upper": 0.90, "n_bootstrap": 20}
        payload["evaluation_split"] = "test"
        payload["evaluation"] = {
            "n_images": 100,
            "timing_repeats": 3,
            "timing_warmup_batches": 5,
            "end_to_end_duration_s": 2.0,
            "end_to_end_images_per_s": 50.0,
            "model_forward_duration_s": 1.2,
            "model_forward_ms_per_image": 12.0,
        }
        payload["run_info"].update({
            "eval_end_to_end_duration_s": 2.0,
            "eval_end_to_end_images_per_s": 50.0,
            "eval_model_forward_duration_s": 1.2,
            "eval_model_forward_ms_per_image": 12.0,
        })
        (folder / "test_metrics.json").write_text(
            json.dumps(payload), encoding="utf-8")

    comparison = write_comparison_report(fake_cfg)
    size_ablation = write_size_ablation_report(fake_cfg)
    comparison_csv = comparison.parent / "comparison.csv"
    size_csv = size_ablation.parent / "size_ablation.csv"
    comparison_text = comparison.read_text("utf-8")
    size_text = size_ablation.read_text("utf-8")
    comparison_csv_text = comparison_csv.read_text("utf-8")
    size_csv_text = size_csv.read_text("utf-8")
    assert "test_end_to_end_images_per_s" in comparison_csv_text
    assert "Forward ms/image" in comparison_text
    assert "End-to-end im/s" in size_text
    assert "test_end_to_end_images_per_s" in size_csv_text
    assert "None" not in comparison_text
    assert "None" not in size_text
    assert "None" not in comparison_csv_text
    assert "None" not in size_csv_text
