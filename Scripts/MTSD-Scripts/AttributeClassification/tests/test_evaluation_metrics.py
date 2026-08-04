"""Metric, bootstrap, and per-class output coverage."""

import csv

import pytest
import torch

from mtsd_attr.evaluate import (bootstrap_macro_f1, evaluate_model,
                                save_metrics_bundle)


class _FixedModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def eval(self):
        return self

    def __call__(self, images):
        logits = torch.full((len(images), 2), -10.0)
        for row, prediction in zip(logits, self.predictions):
            row[prediction] = 10.0
        return {"attribute": logits}


def _loader():
    images = torch.zeros(4, 3, 2, 2)
    targets = torch.tensor([[0], [1], [1], [0]])
    return [(images, targets, ["a", "b", "c", "d"])]


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
