"""Datasets. Every source yields (PIL image, boxes (n, 5) [class, cx, cy, w, h] relative to the image);
`DetectionDataset` applies the transforms and encodes the YOLO target grid."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset

from yolov1.boxes import encode_targets
from yolov1.config import DataConfig
from yolov1.data.transforms import Compose, denormalize, eval_transforms, train_transforms

__all__ = ["DetectionDataset", "Datasets", "build_datasets", "collate_fn", "denormalize"]


class DetectionDataset(Dataset):
    """(image tensor (3, H, W), target grid (S, S, C + 5B), boxes (n, 5)) for every sample of `source`."""

    def __init__(self, source: Dataset, transforms: Compose, grid_size: int, num_boxes: int, num_classes: int):
        self.source, self.transforms = source, transforms
        self.grid_size, self.num_boxes, self.num_classes = grid_size, num_boxes, num_classes

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        image, boxes = self.source[index]
        pixels, boxes = self.transforms(image, boxes)
        target = encode_targets(boxes, self.grid_size, self.num_boxes, self.num_classes)
        return pixels, target, boxes


def collate_fn(batch):
    pixels, targets, boxes = zip(*batch, strict=True)
    return torch.stack(pixels), torch.stack(targets), list(boxes)


@dataclass
class Datasets:
    train: DetectionDataset
    val: DetectionDataset
    class_names: list[str]


def build_sources(cfg: DataConfig):
    """Return (train_source, val_source, class_names) for cfg.dataset."""
    if cfg.dataset == "shapes":
        from yolov1.data.shapes import ShapesDetection

        kw = dict(image_size=max(cfg.image_size, 64), max_objects=cfg.max_objects, seed=cfg.seed)
        train = ShapesDetection(cfg.train_size, split="train", **kw)
        val = ShapesDetection(cfg.val_size, split="val", **kw)
    elif cfg.dataset == "mnist":
        from yolov1.data.mnist import MnistDetection

        kw = dict(root=cfg.root, image_size=cfg.image_size, max_objects=cfg.max_objects, seed=cfg.seed)
        train = MnistDetection(size=cfg.train_size, split="train", **kw)
        val = MnistDetection(size=cfg.val_size, split="val", **kw)
    elif cfg.dataset == "voc":
        from yolov1.data.voc import VocDetection

        train = VocDetection(cfg.root, cfg.voc_year, "train")
        val = VocDetection(cfg.root, cfg.voc_year, "val")
    elif cfg.dataset == "yolo":
        from yolov1.data.yolo_txt import build_yolo_txt

        train, val = build_yolo_txt(cfg.root, cfg.images_dir, cfg.labels_dir, cfg.classes, cfg.val_fraction, cfg.seed)
    else:
        raise ValueError(f"unknown dataset {cfg.dataset!r} (shapes | mnist | voc | yolo)")
    return train, val, list(train.class_names)


def build_datasets(cfg: DataConfig, overfit: int = 0) -> Datasets:
    train_src, val_src, class_names = build_sources(cfg)
    if overfit:
        train_src = Subset(train_src, list(range(min(overfit, len(train_src)))))
        val_src = train_src
    C = len(class_names)
    train = DetectionDataset(
        train_src, train_transforms(cfg.image_size, cfg.augment and not overfit), cfg.grid_size, cfg.num_boxes, C
    )
    val = DetectionDataset(val_src, eval_transforms(cfg.image_size), cfg.grid_size, cfg.num_boxes, C)
    return Datasets(train, val, class_names)
