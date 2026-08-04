"""Metric, bootstrap, and per-class output coverage."""

import csv
import json

import numpy as np
import pytest
import torch

import mtsd_attr.evaluate as evaluate_module
from mtsd_attr.evaluate import (bootstrap_macro_f1, bootstrap_macro_f1_shared,
                                evaluate_model, reevaluate_from_checkpoint,
                                save_metrics_bundle)
from mtsd_attr.dataset import MISSING_LABEL


class _FixedModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, images):
        logits = torch.full((len(images), 2), -10.0)
        for row, prediction in zip(logits, self.predictions):
            row[prediction] = 10.0
        return {"attribute": logits}


class _MultiHeadModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, images):
        outputs = {}
        for attr, predictions in self.predictions.items():
            logits = torch.full((len(images), 2), -10.0)
            for row, prediction in zip(logits, predictions):
                row[prediction] = 10.0
            outputs[attr] = logits
        return outputs


def _loader():
    images = torch.zeros(4, 3, 2, 2)
    targets = torch.tensor([[0], [1], [1], [0]])
    return [(images, targets, ["a", "b", "c", "d"])]


def _multihead_loader(targets):
    images = torch.zeros(len(targets), 3, 2, 2)
    return [(images, torch.tensor(targets), list(range(len(targets))))]


def test_evaluate_model_reports_macro_precision_recall_and_ci():
    attributes = {"attribute": {"classes": ["zero", "one"]}}
    metrics, timing = evaluate_model(
        _FixedModel([0, 0, 1, 0]), _loader(), attributes,
        torch.device("cpu"), bootstrap_samples=100, bootstrap_seed=7,
        return_timing=True)
    metric = metrics["attribute"]

    assert metric["macro_precision"] == pytest.approx(0.8333333333333333)
    assert metric["macro_recall"] == pytest.approx(0.75)
    assert metric["per_class_precision"]["zero"] == pytest.approx(2 / 3)
    assert metric["per_class_recall"]["one"] == pytest.approx(0.5)
    assert metric["per_class_f1"]["one"] == pytest.approx(2 / 3)
    assert metric["macro_f1_ci"]["n_bootstrap"] == 100
    assert metric["macro_f1_ci"]["level"] == 0.95
    assert timing["n_images"] == 4
    assert timing["ms_per_image"] is not None
    assert timing["mean_macro_f1_ci"]["n_bootstrap"] == 100


def test_bootstrap_macro_f1_is_reproducible():
    args = ([0, 1, 1, 0], [0, 0, 1, 0], [0, 1])
    first = bootstrap_macro_f1(*args, n_bootstrap=100, seed=13)
    second = bootstrap_macro_f1(*args, n_bootstrap=100, seed=13)
    assert first == second
    assert 0.0 <= first["lower"] <= first["upper"] <= 1.0


def test_per_class_csv_contains_precision_recall_and_f1(tmp_path):
    attributes = {"attribute": {"classes": ["zero", "one"]}}
    metrics = evaluate_model(
        _FixedModel([0, 0, 1, 0]), _loader(), attributes,
        torch.device("cpu"), bootstrap_samples=0)
    save_metrics_bundle(tmp_path, "demo", "test", metrics, "run-1")
    with open(tmp_path / "demo" / "test_per_class_f1.csv", newline="",
              encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0].keys() >= {"precision", "recall", "f1", "support"}
    assert rows[0]["precision"] == "0.6667"


def test_shared_bootstrap_uses_one_crop_resample_for_all_heads():
    y_true = {"head_a": [0, 0, 1, 1], "head_b": [0, 0, 1, 1]}
    y_pred = {"head_a": [0, 1, 1, 1], "head_b": [0, 0, 0, 1]}
    labels = {"head_a": [0, 1], "head_b": [0, 1]}
    per_attr, mean_samples = bootstrap_macro_f1_shared(
        y_true, y_pred, labels, n_bootstrap=40, seed=19)

    rng = np.random.default_rng(19)
    expected_mean = []
    for _ in range(40):
        indices = rng.integers(0, 4, size=4)
        scores = []
        for attr in ("head_a", "head_b"):
            true = np.asarray(y_true[attr])[indices]
            pred = np.asarray(y_pred[attr])[indices]
            cm = evaluate_module.confusion_matrix(true, pred, labels=[0, 1])
            scores.append(evaluate_module._macro_f1_from_confusion(cm))
        expected_mean.append(np.mean(scores))
    assert np.allclose(mean_samples, expected_mean)

    # Independent streams produce a different statistic, proving this test is
    # sensitive to accidental per-head resampling.
    independent_a = bootstrap_macro_f1(y_true["head_a"], y_pred["head_a"],
                                       [0, 1], 40, 19)
    independent_b = bootstrap_macro_f1(y_true["head_b"], y_pred["head_b"],
                                       [0, 1], 40, 20)
    assert not np.isclose(independent_a["lower"], independent_b["lower"])
    assert len(per_attr["head_a"]) == len(per_attr["head_b"]) == 40


def test_shared_bootstrap_masks_missing_labels_per_head():
    y_true = {
        "head_a": [0, MISSING_LABEL, 1, MISSING_LABEL],
        "head_b": [MISSING_LABEL, 0, 1, 1],
        "empty": [MISSING_LABEL] * 4,
    }
    y_pred = {
        "head_a": [0, 0, 1, 1],
        "head_b": [0, 0, 0, 1],
        "empty": [0, 0, 0, 0],
    }
    labels = {attr: [0, 1] for attr in y_true}
    per_attr, mean_samples = bootstrap_macro_f1_shared(
        y_true, y_pred, labels, n_bootstrap=30, seed=5)
    assert np.isfinite(per_attr["head_a"]).any()
    assert np.isfinite(per_attr["head_b"]).any()
    assert not np.isfinite(per_attr["empty"]).any()
    assert np.isfinite(mean_samples).all()


def test_bootstrap_keeps_zero_support_classes_in_macro_f1():
    metrics, _ = evaluate_model(
        _FixedModel([0, 0, 0, 0]), _loader(),
        {"attribute": {"classes": ["zero", "one", "three"]}},
        torch.device("cpu"), bootstrap_samples=20, timing_repeats=1,
        return_timing=True)
    metric = metrics["attribute"]
    assert metric["support"] == {"zero": 2, "one": 2, "three": 0}
    assert metric["macro_f1"] == pytest.approx(2 / 9)
    assert metric["macro_f1_ci"]["level"] == 0.95


def test_timing_metadata_uses_configured_warmup_and_median(monkeypatch):
    calls = {"warmup": 0, "end": 0, "forward": 0}
    end_durations = iter([3.0, 1.0, 2.0])
    forward_durations = iter([6.0, 2.0, 4.0])

    def fake_warmup(*args):
        calls["warmup"] += 1

    def fake_end(*args):
        calls["end"] += 1
        return 4, next(end_durations)

    def fake_forward(*args):
        calls["forward"] += 1
        return 4, next(forward_durations)

    monkeypatch.setattr(evaluate_module, "_timing_warmup", fake_warmup)
    monkeypatch.setattr(evaluate_module, "_time_end_to_end", fake_end)
    monkeypatch.setattr(evaluate_module, "_time_model_forward", fake_forward)
    result = evaluate_module._measure_timing(
        object(), object(), [], torch.device("cpu"), False,
        timing_warmup_batches=5, timing_repeats=3, n_images=4)
    assert result["timing_warmup_batches"] == 5
    assert result["timing_repeats"] == 3
    assert result["end_to_end_duration_s"] == 2.0
    assert result["model_forward_duration_s"] == 4.0
    assert calls == {"warmup": 1, "end": 3, "forward": 3}


def test_cpu_timing_returns_new_metadata():
    _, timing = evaluate_model(
        _FixedModel([0, 0, 1, 0]), _loader(),
        {"attribute": {"classes": ["zero", "one"]}},
        torch.device("cpu"), bootstrap_samples=0,
        timing_warmup_batches=1, timing_repeats=3, return_timing=True)
    assert timing["n_images"] == 4
    assert timing["timing_repeats"] == 3
    assert timing["timing_warmup_batches"] == 1
    assert timing["end_to_end_duration_s"] >= 0
    assert timing["end_to_end_images_per_s"] >= 0
    assert timing["model_forward_duration_s"] >= 0
    assert timing["model_forward_ms_per_image"] >= 0


def test_reevaluation_is_split_aware_and_test_only_bootstraps(tmp_path, monkeypatch):
    attributes = {"attribute": {"classes": ["zero", "one"]}}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": [{"split": "val"},
                                                  {"split": "test"}]}),
                        encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints" / "demo"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "best.pt"
    torch.save({"model_cfg": {"backbone": "fake"}, "probe": "linear",
                "model_state": {}, "attributes": attributes,
                "parameters": {"total": 1, "trainable": 1}}, checkpoint)

    class FakeBackbone:
        image_size = 2
        backbone_meta = {}

    class FakeDataset:
        def __init__(self, records, *args):
            self.records = records

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return (torch.zeros(3, 2, 2), torch.tensor([index % 2]), index)

    monkeypatch.setattr("mtsd_attr.backbones.build_backbone",
                        lambda cfg: FakeBackbone())
    monkeypatch.setattr("mtsd_attr.train_common._checkpoint_adaptation",
                        lambda ckpt, cfg: "frozen")
    monkeypatch.setattr("mtsd_attr.train_common._load_checkpoint_state",
                        lambda *args: None)
    monkeypatch.setattr("mtsd_attr.dataset.AttributeCropDataset", FakeDataset)
    monkeypatch.setattr("mtsd_attr.dataset.build_transforms", lambda *args: None)
    monkeypatch.setattr("mtsd_attr.multihead_model.MultiHeadClassifier",
                        lambda *args: _FixedModel([0, 0, 1, 0]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cfg = {
        "paths": {"checkpoints_dir": tmp_path / "checkpoints",
                  "metrics_dir": tmp_path / "metrics",
                  "manifest": manifest,
                  "subproject_root": tmp_path},
        "training": {"batch_size": 2, "augmentation": {}},
        "attributes": attributes,
        "evaluation": {"bootstrap_samples": 10, "bootstrap_seed": 3,
                        "timing_warmup_batches": 0, "timing_repeats": 1},
        "seed": 3,
    }
    reevaluate_from_checkpoint(cfg, "demo", split="val")
    val_payload = json.loads(
        (tmp_path / "metrics" / "demo" / "val_metrics.json").read_text())
    assert val_payload["evaluation_split"] == "val"
    assert val_payload["mean_macro_f1_ci"] is None
    assert not any(key.startswith("test_")
                   for key in val_payload["run_info"])

    reevaluate_from_checkpoint(cfg, "demo", split="test")
    test_payload = json.loads(
        (tmp_path / "metrics" / "demo" / "test_metrics.json").read_text())
    assert test_payload["evaluation_split"] == "test"
    assert test_payload["mean_macro_f1_ci"]["n_bootstrap"] == 10
    assert "test_eval_duration_s" in test_payload["run_info"]
