"""Any dataset in the YOLO text format (the Mapillary traffic-sign export used in 2023, Roboflow exports...).

    root/
      images/            name.jpg (or .png, .jpeg)       root/images/train, root/images/val
      labels/            name.txt: "class cx cy w h"     root/labels/train, root/labels/val
      classes.txt        optional, one class name per line

With `images/train` + `images/val` folders the split is used as is. Otherwise the files are split
by a seeded shuffle into disjoint train / validation sets (the 2023 notebook's "test" and
"validation" CSVs were subsets of the training CSV; see legacy/README.md).
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def read_label_file(path: Path) -> torch.Tensor:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(parts[0])] + [float(v) for v in parts[1:5]])
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


class YoloTxtDetection(Dataset):
    def __init__(self, images: list[Path], labels_dir: Path, class_names: list[str]):
        self.images, self.labels_dir, self.class_names = images, labels_dir, class_names

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        boxes = read_label_file(self.labels_dir / f"{path.stem}.txt")
        boxes[:, 1:] = boxes[:, 1:].clamp(0, 1)
        return image, boxes


def load_class_names(root: Path, classes: list[str] | None, labels_dirs: list[Path]) -> list[str]:
    if classes:
        return list(classes)
    for name in ("classes.txt", "classes.names", "obj.names"):
        if (root / name).exists():
            return [line.strip() for line in (root / name).read_text().splitlines() if line.strip()]
    highest = -1
    for folder in labels_dirs:
        for txt in folder.glob("*.txt"):
            rows = read_label_file(txt)
            if rows.shape[0]:
                highest = max(highest, int(rows[:, 0].max()))
    return [str(i) for i in range(highest + 1)]


def build_yolo_txt(
    root: str | Path,
    images_dir: str = "images",
    labels_dir: str = "labels",
    classes: list[str] | None = None,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[YoloTxtDetection, YoloTxtDetection]:
    root = Path(root)
    img_root, lbl_root = root / images_dir, root / labels_dir
    if (img_root / "train").is_dir() and (img_root / "val").is_dir():
        names = load_class_names(root, classes, [lbl_root / "train", lbl_root / "val"])
        return (
            YoloTxtDetection(list_images(img_root / "train"), lbl_root / "train", names),
            YoloTxtDetection(list_images(img_root / "val"), lbl_root / "val", names),
        )
    images = list_images(img_root)
    if not images:
        raise FileNotFoundError(f"no images found in {img_root}")
    random.Random(seed).shuffle(images)
    n_val = max(1, int(round(val_fraction * len(images)))) if len(images) > 1 else 0
    names = load_class_names(root, classes, [lbl_root])
    return YoloTxtDetection(images[n_val:], lbl_root, names), YoloTxtDetection(images[:n_val], lbl_root, names)
