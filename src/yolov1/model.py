"""The YOLOv1 network (Redmon et al., 2016, https://arxiv.org/abs/1506.02640).

A convolutional backbone (24 conv layers in the paper, "DarkNet") reduces the image by a factor
64, then two fully-connected layers output an S x S x (C + 5B) tensor: for each of the S x S grid
cells, C class probabilities and B boxes (confidence, cx, cy, w, h).

Each architecture is a list:  (out_channels, kernel, stride, padding) is a conv-bn-leaky block,
"M" is a 2x2 max-pool and [block, block, n] repeats the two blocks n times.
"""

from __future__ import annotations

import torch
from torch import nn

from yolov1.config import ModelConfig

Block = tuple[int, int, int, int]

# The paper network: 24 conv layers, 448 -> 7 (stride 64), 1024 output channels.
YOLOV1: list = [
    (64, 7, 2, 3),
    "M",
    (192, 3, 1, 1),
    "M",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "M",
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "M",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]

# A small backbone for CPU runs and small images: 6 conv layers, stride 16 (112 -> 7).
SMALL: list = [
    (32, 3, 1, 1),
    "M",
    (64, 3, 1, 1),
    "M",
    (128, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    (256, 3, 1, 1),
]

ARCHITECTURES = {"yolov1": YOLOV1, "small": SMALL}


class ConvBlock(nn.Module):
    """conv (no bias) -> batch norm -> leaky ReLU(0.1). The paper has no batch norm; it stabilises training."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def _scaled(channels: int, width: float) -> int:
    return max(8, int(round(channels * width / 8)) * 8)


def build_backbone(arch: list, in_channels: int, width: float) -> tuple[nn.Sequential, int, int]:
    """Return (layers, out_channels, stride) for an architecture list."""
    layers: list[nn.Module] = []
    stride = 1

    def add(block: Block) -> None:
        nonlocal in_channels, stride
        out, k, s, p = block
        out = _scaled(out, width)
        layers.append(ConvBlock(in_channels, out, k, s, p))
        in_channels, stride = out, stride * s

    for item in arch:
        if isinstance(item, tuple):
            add(item)
        elif item == "M":
            layers.append(nn.MaxPool2d(2, 2))
            stride *= 2
        elif isinstance(item, list):
            first, second, repeats = item
            for _ in range(repeats):
                add(first)
                add(second)
        else:
            raise ValueError(f"unknown architecture item {item!r}")
    return nn.Sequential(*layers), in_channels, stride


class YoloV1(nn.Module):
    """YOLOv1 detector. forward(images) -> (N, S, S, C + 5B)."""

    def __init__(
        self,
        cfg: ModelConfig,
        num_classes: int,
        image_size: int = 448,
        grid_size: int = 7,
        num_boxes: int = 2,
        in_channels: int = 3,
    ):
        super().__init__()
        if cfg.arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {sorted(ARCHITECTURES)}, got {cfg.arch!r}")
        self.num_classes, self.grid_size, self.num_boxes = num_classes, grid_size, num_boxes
        self.image_size = image_size
        self.backbone, channels, self.stride = build_backbone(ARCHITECTURES[cfg.arch], in_channels, cfg.width)
        if image_size % self.stride:
            raise ValueError(f"image_size {image_size} must be a multiple of the stride {self.stride}")
        feature = image_size // self.stride
        self.cell_values = num_classes + 5 * num_boxes
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * feature * feature, cfg.hidden),
            nn.Dropout(cfg.dropout),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(cfg.hidden, grid_size * grid_size * self.cell_values),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.backbone(x))
        return out.view(-1, self.grid_size, self.grid_size, self.cell_values)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
