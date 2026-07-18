from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from mtsd_detection.wandb_utils import (
    _final_metrics,
    attach_rfdetr_logger,
    attach_yolo_logger,
    summarize_yolo_results,
)


class FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)


def test_yolo_logger_emits_mdwd_metric_surface(tmp_path):
    class FakeModel:
        def __init__(self):
            self.callbacks = {}

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

    results = tmp_path / "results.csv"
    results.write_text(
        "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
        "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
        "val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2\n"
        "0,12.5,.10,.20,.30,.8,.6,.7,.5,.11,.21,.31,.001,.002,.003\n",
        encoding="utf-8",
    )
    model, run = FakeModel(), FakeRun()
    assert attach_yolo_logger(model, run, log_interval_steps=1)
    trainer = SimpleNamespace(
        epoch=0, save_dir=tmp_path, loss_items=[.1, .2, .3],
        optimizer=SimpleNamespace(param_groups=[{"lr": .001}, {"lr": .002}, {"lr": .003}]),
        metrics={},
    )
    model.callbacks["on_train_batch_end"](trainer)
    model.callbacks["on_fit_epoch_end"](trainer)
    payload = run.logged[-1]
    assert payload["train/box_loss"] == pytest.approx(.1)
    assert payload["metrics/mAP50-95"] == pytest.approx(.5)
    assert payload["metrics/F1"] == pytest.approx(2 * .8 * .6 / 1.4)
    assert payload["lr/pg2"] == pytest.approx(.003)
    assert payload["runtime/epoch_seconds"] == pytest.approx(12.5)
    summary = summarize_yolo_results(tmp_path)
    assert summary["mAP50-95"] == pytest.approx(.5) and summary["completed_epochs"] == 1


def test_rfdetr_logger_uses_same_comparable_names_without_fake_dfl():
    model = SimpleNamespace(callbacks=defaultdict(list))
    run = FakeRun()
    assert attach_rfdetr_logger(model, run, log_interval_steps=10)
    coco = [.45, .70, .52, 0, 0, 0, 0, 0, .60, 0, 0, 0]
    model.callbacks["on_fit_epoch_end"][0]({
        "epoch": 2,
        "train_loss": 1.2,
        "train_loss_ce": .2,
        "train_loss_bbox": .3,
        "train_loss_giou": .4,
        "train_lr": .001,
        "test_loss": 1.1,
        "ema_test_coco_eval_bbox": coco,
        "ema_test_results_json": {"precision": .8, "recall": .5},
        "epoch_time": "0:01:30",
    })
    payload = run.logged[-1]
    assert payload["epoch"] == 3
    assert payload["train/cls_loss"] == pytest.approx(.2)
    assert payload["train/box_loss"] == pytest.approx(.3)
    assert payload["train/giou_loss"] == pytest.approx(.4)
    assert "train/dfl_loss" not in payload
    assert payload["metrics/mAP50-95"] == pytest.approx(.45)
    assert payload["metrics/mAP50"] == pytest.approx(.70)
    assert payload["metrics/F1"] == pytest.approx(2 * .8 * .5 / 1.3)
    assert payload["runtime/epoch_seconds"] == pytest.approx(90)


def test_final_payload_contains_common_test_and_native_metrics():
    payload = _final_metrics({
        "training_seconds": 120,
        "training_summary": {"completed_epochs": 4, "mAP50-95": .4},
        "native_metrics": {"val": {"metrics/mAP50(B)": .6}},
        "unified_test_metrics": {
            "map50_95": .35, "map50": .55, "map75": .30, "ar100": .50,
            "precision": .7, "recall": .6, "f1": .646,
        },
    })
    assert payload["epoch"] == 4
    assert payload["test/mAP50-95"] == pytest.approx(.35)
    assert payload["test/precision"] == pytest.approx(.7)
    assert payload["native/val/metrics/mAP50(B)"] == pytest.approx(.6)
    assert payload["runtime/train_seconds"] == pytest.approx(120)
