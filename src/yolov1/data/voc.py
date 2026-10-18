"""PASCAL VOC detection (the benchmark of the YOLOv1 paper), downloaded by torchvision (~2 GB for 2012)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision import datasets

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]  # fmt: skip


class VocDetection(Dataset):
    def __init__(self, root: str, year: str = "2012", split: str = "train", download: bool = True):
        self.voc = datasets.VOCDetection(root, year=year, image_set=split, download=download)
        self.class_names = CLASSES

    def __len__(self) -> int:
        return len(self.voc)

    def __getitem__(self, index: int):
        image, annotation = self.voc[index]
        image = image.convert("RGB")
        w, h = image.size
        objects = annotation["annotation"]["object"]
        if isinstance(objects, dict):
            objects = [objects]
        rows = []
        for obj in objects:
            if obj.get("difficult", "0") == "1":
                continue
            b = obj["bndbox"]
            x1, y1, x2, y2 = (float(b[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            rows.append([CLASSES.index(obj["name"]), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
        return image, torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)
