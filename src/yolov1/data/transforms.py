"""Image + box transforms. Every transform takes (PIL image, boxes (n, 5) [class, cx, cy, w, h]) and
returns the same pair, with box coordinates still relative to the (possibly cropped) image."""

from __future__ import annotations

import random

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes


class Resize:
    """Square resize (no aspect ratio preservation, like the paper). Relative boxes do not change."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image, boxes):
        return image.resize((self.size, self.size), Image.BILINEAR), boxes


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, boxes):
        if random.random() < self.p:
            image = TF.hflip(image)
            boxes = boxes.clone()
            boxes[:, 1] = 1 - boxes[:, 1]
        return image, boxes


class RandomScaleTranslate:
    """Random crop / zoom-out of up to `amount` of the image size (paper: 20%), boxes clipped to the view.
    Boxes that keep less than `min_visible` of their area are dropped."""

    def __init__(self, amount: float = 0.2, min_visible: float = 0.3):
        self.amount, self.min_visible = amount, min_visible

    def __call__(self, image, boxes):
        w, h = image.size
        scale = 1 + random.uniform(-self.amount, self.amount)
        new_w, new_h = w * scale, h * scale
        dx = random.uniform(-self.amount, self.amount) * w
        dy = random.uniform(-self.amount, self.amount) * h
        # view window in the original image, in pixels
        x0, y0 = (w - new_w) / 2 + dx, (h - new_h) / 2 + dy
        image = image.transform(
            (int(round(new_w)), int(round(new_h))),
            Image.AFFINE,
            (1, 0, x0, 0, 1, y0),  # output (u, v) samples input (u + x0, v + y0)
            Image.BILINEAR,
            fillcolor=(124, 116, 104),
        )
        if boxes.shape[0] == 0:
            return image, boxes
        b = boxes.clone()
        x1 = (b[:, 1] - b[:, 3] / 2) * w - x0
        y1 = (b[:, 2] - b[:, 4] / 2) * h - y0
        x2 = (b[:, 1] + b[:, 3] / 2) * w - x0
        y2 = (b[:, 2] + b[:, 4] / 2) * h - y0
        area = (x2 - x1) * (y2 - y1)
        cx1, cy1 = x1.clamp(0, new_w), y1.clamp(0, new_h)
        cx2, cy2 = x2.clamp(0, new_w), y2.clamp(0, new_h)
        visible = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)
        keep = visible > self.min_visible * area
        b = torch.stack(
            [b[:, 0], (cx1 + cx2) / 2 / new_w, (cy1 + cy2) / 2 / new_h, (cx2 - cx1) / new_w, (cy2 - cy1) / new_h], 1
        )
        return image, b[keep]


class ColorJitter:
    """Brightness / saturation jitter (paper: up to a factor 1.5), boxes untouched."""

    def __init__(self, factor: float = 0.5):
        self.jitter = T.ColorJitter(brightness=factor, saturation=factor, hue=0.02)

    def __call__(self, image, boxes):
        return self.jitter(image), boxes


class ToTensor:
    def __call__(self, image, boxes):
        return TF.normalize(TF.to_tensor(image), MEAN, STD), boxes


def train_transforms(image_size: int, augment: bool = True) -> Compose:
    steps = []
    if augment:
        steps += [RandomScaleTranslate(), RandomHorizontalFlip(), ColorJitter()]
    return Compose(steps + [Resize(image_size), ToTensor()])


def eval_transforms(image_size: int) -> Compose:
    return Compose([Resize(image_size), ToTensor()])


def denormalize(pixels: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=pixels.device).view(3, 1, 1)
    std = torch.tensor(STD, device=pixels.device).view(3, 1, 1)
    return (pixels * std + mean).clamp(0, 1)
