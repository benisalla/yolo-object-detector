"""Box utilities: IoU, target encoding, prediction decoding and non-maximum suppression.

Box formats (all coordinates relative to the image, in [0, 1]):
  cxcywh  (center x, center y, width, height)  used by labels, targets and predictions
  xyxy    (x_min, y_min, x_max, y_max)          used for IoU and drawing

Detections are rows  [class, score, cx, cy, w, h];  ground truth rows are  [class, cx, cy, w, h].
"""

from __future__ import annotations

import torch


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def box_iou(a: torch.Tensor, b: torch.Tensor, box_format: str = "cxcywh", eps: float = 1e-6) -> torch.Tensor:
    """Element-wise IoU of two broadcastable (..., 4) tensors -> (...)."""
    if box_format == "cxcywh":
        a, b = cxcywh_to_xyxy(a), cxcywh_to_xyxy(b)
    elif box_format != "xyxy":
        raise ValueError(f"unknown box_format {box_format!r}")
    lt = torch.max(a[..., :2], b[..., :2])
    rb = torch.min(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    return inter / (area_a + area_b - inter + eps)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor, box_format: str = "cxcywh") -> torch.Tensor:
    """IoU between every box of a (N, 4) and every box of b (M, 4) -> (N, M)."""
    return box_iou(a[:, None, :], b[None, :, :], box_format)


def encode_targets(boxes: torch.Tensor, grid_size: int, num_boxes: int, num_classes: int) -> torch.Tensor:
    """Ground truth boxes (n, 5) [class, cx, cy, w, h] -> target grid (S, S, C + 5B).

    The cell containing the box center gets: one-hot class, confidence 1 in box slot 0 and the box
    as (x offset in the cell, y offset in the cell, w * S, h * S), as in the paper. If several
    objects fall in one cell only the first is kept: YOLOv1 predicts one object per cell.
    """
    S, B, C = grid_size, num_boxes, num_classes
    target = torch.zeros(S, S, C + 5 * B)
    for cls, cx, cy, w, h in boxes.tolist():
        col, row = min(int(cx * S), S - 1), min(int(cy * S), S - 1)  # cx = 1.0 belongs to the last cell
        if target[row, col, C] == 1:
            continue
        target[row, col, int(cls)] = 1
        target[row, col, C] = 1
        target[row, col, C + 1 : C + 5] = torch.tensor([cx * S - col, cy * S - row, w * S, h * S])
    return target


def decode_predictions(pred: torch.Tensor, num_boxes: int, num_classes: int) -> list[torch.Tensor]:
    """Network output (N, S, S, C + 5B) -> one (S*S*B, 6) tensor of [class, score, cx, cy, w, h] per image.

    score = box confidence * best class probability (the paper's class-specific confidence), with
    everything clamped to valid ranges because the network outputs are unbounded.
    """
    N, S, _, _ = pred.shape
    B, C = num_boxes, num_classes
    pred = pred.detach().float()
    class_prob, class_idx = pred[..., :C].clamp(0, 1).max(-1)  # (N, S, S)
    boxes = pred[..., C:].reshape(N, S, S, B, 5)
    conf = boxes[..., 0].clamp(0, 1)  # (N, S, S, B)
    cols = torch.arange(S, device=pred.device, dtype=pred.dtype)
    cx = (boxes[..., 1] + cols.view(1, 1, S, 1)) / S
    cy = (boxes[..., 2] + cols.view(1, S, 1, 1)) / S
    wh = boxes[..., 3:5].clamp(min=0) / S
    score = conf * class_prob.unsqueeze(-1)
    cls = class_idx.unsqueeze(-1).expand_as(conf).to(pred.dtype)
    rows = torch.cat([cls.unsqueeze(-1), score.unsqueeze(-1), cx.unsqueeze(-1), cy.unsqueeze(-1), wh], -1)
    return [rows[i].reshape(-1, 6) for i in range(N)]


def non_max_suppression(detections: torch.Tensor, iou_threshold: float, score_threshold: float) -> torch.Tensor:
    """Greedy per-class NMS on (K, 6) [class, score, cx, cy, w, h] rows -> kept rows sorted by score."""
    detections = detections[detections[:, 1] > score_threshold]
    if detections.numel() == 0:
        return detections.reshape(0, 6)
    detections = detections[detections[:, 1].argsort(descending=True)]
    keep = []
    while detections.shape[0]:
        best, rest = detections[0], detections[1:]
        keep.append(best)
        if rest.shape[0] == 0:
            break
        same_class = rest[:, 0] == best[0]
        ious = box_iou(best[None, 2:6], rest[:, 2:6])
        detections = rest[~(same_class & (ious > iou_threshold))]
    return torch.stack(keep)
