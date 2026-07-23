import json
import sys
from pathlib import Path
from types import SimpleNamespace

from mtsd_detection.model_registry import registry
from mtsd_detection import train_rfdetr, train_yolo


TRAINING = {"epochs": 100, "image_size": 640, "effective_batch": 32, "optimizer": "AdamW",
            "lr0": .001, "lrf": .01, "weight_decay": .0005, "warmup_epochs": 3,
            "patience": 10, "seed": 42, "deterministic": True, "device": "cpu"}


def test_mocked_yolo_vertical_run_preserves_protocol(tmp_path, monkeypatch):
    class FakeYOLO:
        def __init__(self, checkpoint): self.checkpoint = checkpoint
        def train(self, **kwargs):
            run = Path(kwargs["project"]) / kwargs["name"]
            (run / "weights").mkdir(parents=True); (run / "weights/best.pt").write_bytes(b"best")
            return object()
        def val(self, **kwargs): return SimpleNamespace(results_dict={"metrics/mAP50(B)": .5})
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    checkpoint = tmp_path / "yolo12n.pt"; checkpoint.write_bytes(b"source")
    data = tmp_path / "data.yaml"; data.write_text("names: [A]\n", encoding="utf-8")
    run_dir = tmp_path / "runs/run"
    result = train_yolo.train(registry()["yolo12n"], checkpoint, data, run_dir, TRAINING, smoke=True)
    assert Path(result["checkpoint_best"]).is_file()
    assert result["native_metrics"]["test"]["metrics/mAP50(B)"] == .5
    assert result["train_args"]["mosaic"] == 0 and result["train_args"]["fliplr"] == 0
    assert result["train_args"]["plots"] is True
    assert result["train_args"]["batch"] == 32 and result["train_args"]["nbs"] == 64
    assert result["batch_plan"]["gradient_accumulation_steps"] == 2


def test_yolo_resolution_batch_plan_preserves_optimizer_batch():
    training = {**TRAINING, "ultralytics_optimizer_batch": 64,
                "ultralytics_physical_batch_by_image_size": {640: 32, 960: 16, 1280: 8}}
    assert train_yolo.yolo_batch_plan(training, 640) == {
        "physical_batch": 32, "optimizer_effective_batch": 64, "gradient_accumulation_steps": 2}
    assert train_yolo.yolo_batch_plan(training, 960) == {
        "physical_batch": 16, "optimizer_effective_batch": 64, "gradient_accumulation_steps": 4}
    assert train_yolo.yolo_batch_plan(training, 1280) == {
        "physical_batch": 8, "optimizer_effective_batch": 64, "gradient_accumulation_steps": 8}


def test_yolo_online_augmentation_metric_push_preset():
    args = train_yolo.yolo_online_augmentation_args({
        **TRAINING,
        "yolo_online_augmentation": "traffic_metric_push",
    })
    assert args["mosaic"] > 0
    assert args["scale"] > 0
    assert args["flipud"] == 0
    assert args["close_mosaic"] == 10


def test_mocked_rfdetr_vertical_run_records_native_metrics(tmp_path, monkeypatch):
    class FakeRFDETR:
        constructed_kwargs = None
        def __init__(self, **kwargs):
            type(self).constructed_kwargs = kwargs
        def train(self, **kwargs):
            output = Path(kwargs["output_dir"])
            (output / "checkpoint_best_ema.pth").write_bytes(b"best")
            (output / "results.json").write_text(json.dumps({"map": .4}), encoding="utf-8")
    monkeypatch.setattr(train_rfdetr, "_rfdetr_class", lambda spec: FakeRFDETR)
    dataset = tmp_path / "coco"
    for split in ("train", "valid", "test"):
        path = dataset / split / "_annotations.coco.json"; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"categories": [{"id": 0, "name": "A", "supercategory": "x"}],
                                    "images": [], "annotations": []}), encoding="utf-8")
    checkpoint = tmp_path / "rf.pth"; checkpoint.write_bytes(b"source")
    result = train_rfdetr.train(registry()["rfdetr-n"], checkpoint, dataset, tmp_path / "run", TRAINING, smoke=True)
    assert Path(result["checkpoint_best"]).name == "checkpoint_best_ema.pth"
    assert result["native_metrics"]["test"] == {"map": .4}
    assert FakeRFDETR.constructed_kwargs["resolution"] == 384
    assert result["model_args"]["resolution"] == 384
    assert result["train_args"]["multi_scale"] is True
    assert result["train_args"]["wandb"] is False
    assert result["train_args"]["run_test"] is True


def test_rfdetr_uses_native_scale_resolutions():
    specs = registry()
    assert train_rfdetr.rfdetr_resolution(specs["rfdetr-n"], TRAINING) == 384
    assert train_rfdetr.rfdetr_resolution(specs["rfdetr-s"], TRAINING) == 512
    assert train_rfdetr.rfdetr_resolution(specs["rfdetr-m"], TRAINING) == 576
