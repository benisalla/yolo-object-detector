"""MNIST detection: 1-3 handwritten digits pasted at random positions and sizes on a dark canvas.

Ten classes with real (noisy, varied) pixels but exact boxes. MNIST is downloaded by torchvision
(about 12 MB) the first time.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

CLASSES = [str(d) for d in range(10)]


class MnistDetection(Dataset):
    def __init__(
        self,
        root: str,
        size: int,
        image_size: int = 112,
        max_objects: int = 3,
        seed: int = 0,
        split: str = "train",
        download: bool = True,
    ):
        self.mnist = datasets.MNIST(root, train=split == "train", download=download)
        self.size, self.image_size, self.max_objects = size, image_size, max_objects
        self.seed = seed * 1_000_003 + {"train": 0, "val": 1, "test": 2}[split] * 100_003
        self.class_names = CLASSES

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + index)
        s = self.image_size
        canvas = np.zeros((s, s), dtype=np.uint8)
        boxes = []
        for _ in range(rng.randint(1, self.max_objects)):
            digit, label = self.mnist[rng.randrange(len(self.mnist))]
            side = int(rng.uniform(0.25, 0.45) * s)
            digit = np.array(digit.resize((side, side), Image.BILINEAR))
            ys, xs = np.nonzero(digit > 50)
            if len(xs) == 0:
                continue
            x0, y0 = rng.randint(0, s - side), rng.randint(0, s - side)
            bx1, by1, bx2, by2 = x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1
            if any(bx1 < ox2 and bx2 > ox1 and by1 < oy2 and by2 > oy1 for ox1, oy1, ox2, oy2 in (b[5:] for b in boxes)):
                continue  # keep digits apart
            region = canvas[y0 : y0 + side, x0 : x0 + side]
            np.maximum(region, digit, out=region)
            boxes.append(
                [label, (bx1 + bx2) / 2 / s, (by1 + by2) / 2 / s, (bx2 - bx1) / s, (by2 - by1) / s, bx1, by1, bx2, by2]
            )
        image = Image.fromarray(canvas).convert("RGB")
        return image, torch.tensor([b[:5] for b in boxes], dtype=torch.float32).reshape(-1, 5)
