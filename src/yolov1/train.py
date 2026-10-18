"""Training loop and the `yolov1-train` command.

Writes to --output-dir:  best.pt (highest validation mAP), last.pt (for --resume), history.json,
history.png (curves) and predictions.png (validation images with ground truth and predictions).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from yolov1.boxes import decode_predictions, non_max_suppression
from yolov1.config import DataConfig, ModelConfig, TrainConfig
from yolov1.data import Datasets, build_datasets, collate_fn
from yolov1.loss import YoloLoss
from yolov1.metrics import mean_average_precision
from yolov1.model import YoloV1
from yolov1.viz import plot_history, plot_predictions

log = logging.getLogger("yolov1")
BEST_CHECKPOINT, LAST_CHECKPOINT = "best.pt", "last.pt"


@dataclass
class DatasetInfo:
    class_names: list[str]
    image_size: int
    grid_size: int
    num_boxes: int

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to 1% of the peak."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def build_model(cfg: ModelConfig, info: DatasetInfo) -> YoloV1:
    return YoloV1(cfg, info.num_classes, info.image_size, info.grid_size, info.num_boxes)


def build_dataloaders(datasets: Datasets, cfg: TrainConfig, data_cfg: DataConfig, device: torch.device):
    kw = dict(
        batch_size=cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=data_cfg.num_workers > 0,
    )
    train_loader = DataLoader(datasets.train, shuffle=True, drop_last=len(datasets.train) > cfg.batch_size, **kw)
    val_loader = DataLoader(datasets.val, shuffle=False, **kw)
    return train_loader, val_loader


def postprocess(output: torch.Tensor, info: DatasetInfo, conf_threshold: float, nms_iou: float) -> list[torch.Tensor]:
    """Network output -> list of NMS-filtered detections (K, 6) per image, on the CPU."""
    return [
        non_max_suppression(d.cpu(), nms_iou, conf_threshold)
        for d in decode_predictions(output, info.num_boxes, info.num_classes)
    ]


@torch.no_grad()
def evaluate(model, loader, loss_fn, info: DatasetInfo, cfg: TrainConfig, device, keep_examples: int = 8) -> dict:
    model.eval()
    total_loss, n_batches = 0.0, 0
    predictions, targets = [], []
    examples = {"images": [], "predictions": [], "targets": []}
    for pixels, grid, boxes in loader:
        pixels, grid = pixels.to(device, non_blocking=True), grid.to(device, non_blocking=True)
        output = model(pixels)
        total_loss += loss_fn(output, grid).item()
        n_batches += 1
        detections = postprocess(output, info, cfg.conf_threshold, cfg.nms_iou)
        predictions += detections
        targets += [b.cpu() for b in boxes]
        if len(examples["images"]) < keep_examples:
            take = keep_examples - len(examples["images"])
            examples["images"] += list(pixels[:take].cpu())
            examples["predictions"] += detections[:take]
            examples["targets"] += [b.cpu() for b in boxes[:take]]
    model.train()
    result = mean_average_precision(predictions, targets, info.num_classes, cfg.map_iou)
    return {
        "loss": total_loss / max(1, n_batches),
        "map": result["map"],
        "ap": {info.class_names[c]: ap for c, ap in result["ap"].items()},
        "examples": examples,
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_map: float, configs: dict, info) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_map": best_map,
            "configs": configs,
            "info": dataclasses.asdict(info),
        },
        path,
    )


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[YoloV1, DatasetInfo, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    info = DatasetInfo(**ckpt["info"])
    model = build_model(ModelConfig(**ckpt["configs"]["model"]), info)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), info, ckpt["configs"]


def train(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    resume: str | None = None,
    callback: Callable[[dict], None] | None = None,
) -> dict:
    """Train and return the best validation metrics. `callback(event)` is called after every epoch."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = {
        "data": dataclasses.asdict(data_cfg),
        "model": dataclasses.asdict(model_cfg),
        "train": dataclasses.asdict(cfg),
    }

    datasets = build_datasets(data_cfg, overfit=cfg.overfit)
    info = DatasetInfo(datasets.class_names, data_cfg.image_size, data_cfg.grid_size, data_cfg.num_boxes)
    train_loader, val_loader = build_dataloaders(datasets, cfg, data_cfg, device)
    model = build_model(model_cfg, info).to(device)
    loss_fn = YoloLoss(info.grid_size, info.num_boxes, info.num_classes, cfg.lambda_coord, cfg.lambda_noobj)
    log.info(
        "%s: %d classes, train %d, val %d | model %s x%.2f, %.1fM parameters | %s",
        data_cfg.dataset, info.num_classes, len(datasets.train), len(datasets.val),
        model_cfg.arch, model_cfg.width, model.num_parameters() / 1e6, device,
    )  # fmt: skip

    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr
    )
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = max(1, int(cfg.warmup_epochs * len(train_loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, warmup_steps, total_steps))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    start_epoch, best_map = 1, -1.0
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_map = ckpt["epoch"] + 1, ckpt["best_map"]
        history_path = Path(resume).with_name("history.json")
        if history_path.exists():
            history = json.loads(history_path.read_text())
        log.info("resumed from %s at epoch %d (best mAP %.3f)", resume, ckpt["epoch"], best_map)

    best = {}
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0, running, components, n = time.time(), 0.0, {"coord": 0.0, "obj": 0.0, "noobj": 0.0, "class": 0.0}, 0
        for step, (pixels, grid, _) in enumerate(train_loader, 1):
            pixels, grid = pixels.to(device, non_blocking=True), grid.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                output = model(pixels)
            loss = loss_fn(output, grid)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            for k in components:
                components[k] += loss_fn.last_components[k]
            n += 1
            if cfg.log_every and step % cfg.log_every == 0:
                log.info("epoch %d step %d/%d loss %.3f", epoch, step, len(train_loader), running / n)

        val = evaluate(model, val_loader, loss_fn, info, cfg, device)
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, n),
            **{f"train_{k}": v / max(1, n) for k, v in components.items()},
            "val_loss": val["loss"],
            "val_map": val["map"],
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        history.append(record)
        log.info(
            "epoch %d/%d  train loss %.3f  val loss %.3f  val mAP@%.1f %.3f  (%.0fs)",
            epoch, cfg.epochs, record["train_loss"], val["loss"], cfg.map_iou, val["map"], record["seconds"],
        )  # fmt: skip
        if val["map"] > best_map:
            best_map = val["map"]
            best = {"epoch": epoch, "map": val["map"], "ap": val["ap"], "val_loss": val["loss"]}
            save_checkpoint(out / BEST_CHECKPOINT, model, optimizer, scheduler, epoch, best_map, configs, info)
        save_checkpoint(out / LAST_CHECKPOINT, model, optimizer, scheduler, epoch, best_map, configs, info)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if epoch == cfg.epochs or (cfg.plot_every and epoch % cfg.plot_every == 0):
            plot_history(history, out / "history.png")
            ex = val["examples"]
            if ex["images"]:
                plot_predictions(
                    ex["images"], ex["predictions"], ex["targets"], info.class_names, out / "predictions.png"
                )
        if callback:
            callback({**record, "ap": val["ap"], "best_map": best_map})

    (out / "best_metrics.json").write_text(json.dumps(best, indent=2))
    log.info("best validation mAP@%.1f %.3f at epoch %s", cfg.map_iou, best_map, best.get("epoch"))
    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YOLOv1.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for group_name, config_cls in (("data", DataConfig), ("model", ModelConfig), ("training", TrainConfig)):
        group = parser.add_argument_group(group_name)
        for f in fields(config_cls):
            flag = "--" + f.name.replace("_", "-")
            if group_name == "training" and f.name == "seed":
                continue  # one --seed flag (data group) feeds both configs
            if isinstance(f.default, bool):
                group.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
            elif f.name == "classes":
                group.add_argument(flag, nargs="+", default=None, help="class names (yolo dataset)")
            else:
                group.add_argument(flag, type=type(f.default), default=f.default)
    parser.add_argument("--resume", default=None, help="continue from a last.pt checkpoint")
    return parser


def configs_from_args(args: argparse.Namespace) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    def pick(config_cls):
        return config_cls(**{f.name: getattr(args, f.name) for f in fields(config_cls) if hasattr(args, f.name)})

    data_cfg, model_cfg, train_cfg = pick(DataConfig), pick(ModelConfig), pick(TrainConfig)
    train_cfg.seed = data_cfg.seed
    return data_cfg, model_cfg, train_cfg


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    train(*configs_from_args(args), resume=args.resume)


if __name__ == "__main__":
    main()
