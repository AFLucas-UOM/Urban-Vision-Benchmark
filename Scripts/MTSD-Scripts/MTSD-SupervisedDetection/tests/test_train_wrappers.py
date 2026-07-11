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


def test_mocked_rfdetr_vertical_run_records_native_metrics(tmp_path, monkeypatch):
    class FakeRFDETR:
        def __init__(self, **kwargs): self.kwargs = kwargs
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
    assert result["train_args"]["multi_scale"] is True
