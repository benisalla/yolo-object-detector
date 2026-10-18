"""The `yolov1-evaluate` command: mAP of a checkpoint on the validation split of a dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from yolov1.config import DataConfig, TrainConfig
from yolov1.data import build_datasets
from yolov1.loss import YoloLoss
from yolov1.train import build_dataloaders, evaluate, load_model, resolve_device
from yolov1.viz import plot_predictions


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv1 checkpoint (mAP@IoU on the validation split).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default=None, help="dataset folder (defaults to the one used for training)")
    parser.add_argument("--conf-threshold", type=float, default=0.1)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--map-iou", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--report", default=None, help="write the metrics to this json file")
    parser.add_argument("--predictions", default=None, help="write a prediction grid to this png file")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model, info, configs = load_model(args.checkpoint, device)
    data_cfg = DataConfig(**configs["data"])
    if args.root:
        data_cfg.root = args.root
    cfg = TrainConfig(
        batch_size=args.batch_size, conf_threshold=args.conf_threshold, nms_iou=args.nms_iou, map_iou=args.map_iou
    )
    datasets = build_datasets(data_cfg)
    _, val_loader = build_dataloaders(datasets, cfg, data_cfg, device)
    loss_fn = YoloLoss(info.grid_size, info.num_boxes, info.num_classes)
    with torch.no_grad():
        result = evaluate(model, val_loader, loss_fn, info, cfg, device)
    print(
        f"validation images: {len(datasets.val)}   loss {result['loss']:.3f}   mAP@{args.map_iou} {result['map']:.4f}"
    )
    for name, ap in sorted(result["ap"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:24} AP {ap:.4f}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        report = {"map": result["map"], "map_iou": args.map_iou, "loss": result["loss"], "ap": result["ap"]}
        Path(args.report).write_text(json.dumps(report, indent=2))
    if args.predictions:
        ex = result["examples"]
        plot_predictions(ex["images"], ex["predictions"], ex["targets"], info.class_names, args.predictions)


if __name__ == "__main__":
    main()
