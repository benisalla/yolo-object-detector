"""Mean average precision (mAP) at a fixed IoU threshold, VOC style with all-point interpolation."""

from __future__ import annotations

import torch

from yolov1.boxes import pairwise_iou


def average_precision(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """Area under the precision envelope of a (recall, precision) curve, both sorted by score."""
    r = torch.cat([torch.zeros(1), recall, torch.ones(1)])
    p = torch.cat([torch.zeros(1), precision, torch.zeros(1)])
    for i in range(p.numel() - 2, -1, -1):  # make precision monotonically decreasing
        p[i] = torch.max(p[i], p[i + 1])
    changes = torch.nonzero(r[1:] != r[:-1]).flatten()
    return float(((r[changes + 1] - r[changes]) * p[changes + 1]).sum())


def mean_average_precision(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict:
    """predictions[i]: (K_i, 6) [class, score, cx, cy, w, h]; targets[i]: (M_i, 5) [class, cx, cy, w, h].

    Returns {"map": float, "ap": {class_id: float}} over the classes that have at least one ground
    truth box. Each ground truth box can be matched once; extra detections are false positives.
    """
    assert len(predictions) == len(targets), "one prediction tensor per image"
    aps: dict[int, float] = {}
    for c in range(num_classes):
        gts = {i: t[t[:, 0] == c, 1:5] for i, t in enumerate(targets) if (t[:, 0] == c).any()}
        n_gt = sum(g.shape[0] for g in gts.values())
        if n_gt == 0:
            continue
        dets = [(i, row) for i, p in enumerate(predictions) for row in p[p[:, 0] == c]]
        if not dets:
            aps[c] = 0.0
            continue
        dets.sort(key=lambda d: -float(d[1][1]))
        matched = {i: torch.zeros(g.shape[0], dtype=torch.bool) for i, g in gts.items()}
        tp = torch.zeros(len(dets))
        for k, (img, row) in enumerate(dets):
            if img not in gts:
                continue
            ious = pairwise_iou(row[None, 2:6], gts[img])[0]
            best = int(ious.argmax())
            if ious[best] >= iou_threshold and not matched[img][best]:
                matched[img][best] = True
                tp[k] = 1
        tp_cum = tp.cumsum(0)
        fp_cum = (1 - tp).cumsum(0)
        aps[c] = average_precision(tp_cum / n_gt, tp_cum / (tp_cum + fp_cum))
    return {"map": sum(aps.values()) / len(aps) if aps else 0.0, "ap": aps}
