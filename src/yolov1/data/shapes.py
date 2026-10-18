"""Synthetic detection data: colored circles, squares and triangles on a plain background.

No download, fully deterministic (image i is always the same for a given seed), with exact boxes:
the ideal first dataset to prove the detector learns.
"""

from __future__ import annotations

import math
import random

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

CLASSES = ["circle", "square", "triangle"]


def _draw_shape(draw: ImageDraw.ImageDraw, cls: int, cx: float, cy: float, r: float, color) -> None:
    if cls == 0:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif cls == 1:
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)]
        draw.polygon(pts, fill=color)


class ShapesDetection(Dataset):
    """Returns (PIL image, boxes (n, 5) [class, cx, cy, w, h] relative to the image)."""

    def __init__(self, size: int, image_size: int = 128, max_objects: int = 3, seed: int = 0, split: str = "train"):
        self.size, self.image_size, self.max_objects = size, image_size, max_objects
        self.seed = seed * 1_000_003 + {"train": 0, "val": 1, "test": 2}[split] * 100_003
        self.class_names = CLASSES

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + index)
        s = self.image_size
        bg = tuple(rng.randint(150, 255) for _ in range(3))
        image = Image.new("RGB", (s, s), bg)
        draw = ImageDraw.Draw(image)
        boxes = []
        for _ in range(rng.randint(1, self.max_objects)):
            cls = rng.randrange(len(CLASSES))
            r = rng.uniform(0.08, 0.2) * s
            cx, cy = rng.uniform(r, s - r), rng.uniform(r, s - r)
            if any(
                abs(cx - bx * s) < (r + bw * s / 2) and abs(cy - by * s) < (r + bh * s / 2)
                for _, bx, by, bw, bh in boxes
            ):
                continue  # skip overlapping shapes: one object per cell is YOLOv1's assumption anyway
            color = tuple(rng.randint(0, 120) for _ in range(3))
            _draw_shape(draw, cls, cx, cy, r, color)
            if cls == 2:  # the triangle's tight box
                top, bottom = cy - r, cy + r * math.sin(math.pi / 6)
                half_w = r * math.cos(math.pi / 6)
                boxes.append([cls, cx / s, (top + bottom) / 2 / s, 2 * half_w / s, (bottom - top) / s])
            else:
                boxes.append([cls, cx / s, cy / s, 2 * r / s, 2 * r / s])
        return image, torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5)
