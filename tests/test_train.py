"""End-to-end training checks on tiny synthetic data. CPU only, a couple of minutes."""

import json

import torch
from PIL import Image

from yolov1.config import DataConfig, ModelConfig, TrainConfig
from yolov1.evaluate import main as evaluate_main
from yolov1.predict import Detector
from yolov1.predict import main as predict_main
from yolov1.train import build_parser, configs_from_args, load_model, lr_lambda, train

OUT = "run"


def tiny(tmp_path, **train_kw):
    data = DataConfig(dataset="shapes", image_size=64, grid_size=4, train_size=16, val_size=4, num_workers=0)
    model = ModelConfig(arch="small", width=0.5, hidden=256, dropout=0.0)
    train_cfg = TrainConfig(
        batch_size=4,
        lr=1e-3,
        warmup_epochs=5,
        log_every=0,
        plot_every=0,
        device="cpu",
        output_dir=str(tmp_path / OUT),
        **train_kw,
    )
    return data, model, train_cfg


def test_overfit_a_few_samples(tmp_path):
    """The classic sanity check: a detector that cannot memorise 4 images has a bug in data, loss or decoding."""
    best = train(*tiny(tmp_path, epochs=60, overfit=4))
    history = json.loads((tmp_path / OUT / "history.json").read_text())
    assert history[-1]["train_loss"] < 0.1 * history[0]["train_loss"]
    assert best["map"] > 0.9


def test_outputs_resume_evaluate_and_predict(tmp_path):
    data, model, cfg = tiny(tmp_path, epochs=2)
    train(data, model, cfg)
    run = tmp_path / OUT
    for name in ("best.pt", "last.pt", "history.json", "history.png", "predictions.png", "best_metrics.json"):
        assert (run / name).exists(), name

    cfg.epochs = 3
    train(data, model, cfg, resume=str(run / "last.pt"))
    assert [h["epoch"] for h in json.loads((run / "history.json").read_text())] == [1, 2, 3]

    net, info, configs = load_model(run / "last.pt")
    assert info.num_classes == 3 and info.image_size == 64 and configs["model"]["width"] == 0.5
    assert net(torch.randn(1, 3, 64, 64)).shape == (1, 4, 4, 13)

    evaluate_main(["--checkpoint", str(run / "best.pt"), "--report", str(tmp_path / "report.json"), "--device", "cpu"])
    report = json.loads((tmp_path / "report.json").read_text())
    assert 0.0 <= report["map"] <= 1.0 and set(report["ap"]) <= {"circle", "square", "triangle"}

    Image.new("RGB", (50, 40), (200, 200, 200)).save(tmp_path / "photo.jpg")
    predict_main(["--checkpoint", str(run / "best.pt"), "--images", str(tmp_path / "photo.jpg"),
                  "--output-dir", str(tmp_path / "out"), "--device", "cpu"])  # fmt: skip
    assert Image.open(tmp_path / "out" / "photo_pred.jpg").size == (50, 40)
    detections = json.loads((tmp_path / "out" / "detections.json").read_text())
    assert list(detections) == [str(tmp_path / "photo.jpg")]
    detector = Detector(run / "best.pt", conf_threshold=0.0)
    assert detector.detect(Image.open(tmp_path / "photo.jpg")).shape[1] == 6


def test_cli_flags_reach_configs():
    args = build_parser().parse_args(
        ["--dataset", "yolo", "--classes", "a", "b", "--width", "0.5", "--no-amp", "--seed", "7", "--lambda-coord", "3"]
    )
    data, model, cfg = configs_from_args(args)
    assert data.dataset == "yolo" and data.classes == ["a", "b"] and data.seed == 7
    assert model.width == 0.5 and cfg.amp is False and cfg.seed == 7 and cfg.lambda_coord == 3.0


def test_lr_schedule():
    assert lr_lambda(0, 10, 100) == 0.1
    assert lr_lambda(10, 10, 100) == 1.0
    assert abs(lr_lambda(100, 10, 100) - 0.01) < 1e-9
