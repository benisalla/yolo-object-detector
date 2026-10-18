"""Drawing boxes on images, prediction grids and training curves."""

from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib
import torch
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from yolov1.data.transforms import denormalize  # noqa: E402


def class_color(index: int, total: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((index / max(total, 1)) % 1.0, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    class_names: list[str],
    with_scores: bool = True,
    width: int = 2,
) -> Image.Image:
    """Draw [class, score, cx, cy, w, h] (or [class, cx, cy, w, h] when with_scores=False) rows on a copy."""
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    W, H = image.size
    font = ImageFont.load_default()
    for row in boxes.tolist():
        if with_scores:
            cls, score, cx, cy, w, h = row
            label = f"{class_names[int(cls)]} {score:.2f}"
        else:
            cls, cx, cy, w, h = row
            label = class_names[int(cls)]
        color = class_color(int(cls), len(class_names))
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H  # center -> top-left corner
        x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        ty = y1 - th - 2 if y1 - th - 2 > 0 else y1
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=color)
        draw.text((x1 + 2, ty), label, fill="black", font=font)
    return image


def tensor_to_pil(pixels: torch.Tensor) -> Image.Image:
    return Image.fromarray((denormalize(pixels).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))


def plot_predictions(
    images: list[torch.Tensor],
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    class_names: list[str],
    path: str | Path,
    max_images: int = 8,
) -> None:
    n = min(max_images, len(images))
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False)
    for i in range(n):
        pil = tensor_to_pil(images[i])
        axes[0, i].imshow(draw_boxes(pil, targets[i], class_names, with_scores=False))
        axes[1, i].imshow(draw_boxes(pil, predictions[i], class_names))
        axes[0, i].set_title("ground truth", fontsize=9)
        axes[1, i].set_title(f"prediction ({len(predictions[i])} boxes)", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_history(history: list[dict], path: str | Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set_title("YOLO loss")
    axes[0].legend()
    axes[1].plot(epochs, [h["val_map"] for h in history], "o-", color="tab:green")
    axes[1].set_title("validation mAP@0.5")
    axes[1].set_ylim(0, 1)
    for key in ("coord", "obj", "noobj", "class"):
        axes[2].plot(epochs, [h["train_" + key] for h in history], label=key)
    axes[2].set_title("train loss components")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
