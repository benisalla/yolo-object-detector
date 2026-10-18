"""The YOLOv1 loss (paper, eq. 3): sum-squared error over cells and boxes.

For each cell with an object, the predictor whose box has the highest IoU with the ground truth is
"responsible" for it. The loss is

  lambda_coord * sum_resp [(x - x̂)² + (y - ŷ)² + (√w - √ŵ)² + (√h - √ĥ)²]      coordinates
  +              sum_resp (1 - Ĉ)²                                              confidence, responsible boxes
  + lambda_noobj * sum_not_resp Ĉ²                                              confidence, every other box
  +              sum_obj_cells sum_c (p(c) - p̂(c))²                             class probabilities

all divided by the batch size, so the learning rate does not depend on the batch size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from yolov1.boxes import box_iou


class YoloLoss(nn.Module):
    def __init__(
        self,
        grid_size: int,
        num_boxes: int,
        num_classes: int,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ):
        super().__init__()
        self.S, self.B, self.C = grid_size, num_boxes, num_classes
        self.lambda_coord, self.lambda_noobj = lambda_coord, lambda_noobj
        self.last_components: dict[str, float] = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        N, S, B, C = pred.shape[0], self.S, self.B, self.C
        pred = pred.float().reshape(N, S, S, C + 5 * B)
        target = target.float()
        obj = target[..., C]  # (N, S, S) 1 where the cell holds an object
        tgt_box = target[..., C + 1 : C + 5]  # (N, S, S, 4) cell-relative x, y and grid-relative w, h
        pred_boxes = pred[..., C:].reshape(N, S, S, B, 5)
        pred_conf, pred_xy, pred_wh = pred_boxes[..., 0], pred_boxes[..., 1:3], pred_boxes[..., 3:5]

        # Which of the B boxes is responsible: highest IoU with the ground truth box of its cell.
        with torch.no_grad():
            ious = box_iou(pred_boxes[..., 1:5], tgt_box.unsqueeze(3))  # (N, S, S, B)
            responsible = F.one_hot(ious.argmax(-1), B).float() * obj.unsqueeze(-1)  # (N, S, S, B)

        tgt_xy, tgt_wh = tgt_box[..., :2].unsqueeze(3), tgt_box[..., 2:4].unsqueeze(3)
        xy_err = ((pred_xy - tgt_xy) ** 2).sum(-1)
        sqrt_pred_wh = torch.sign(pred_wh) * torch.sqrt(pred_wh.abs() + 1e-6)
        wh_err = ((sqrt_pred_wh - torch.sqrt(tgt_wh)) ** 2).sum(-1)
        coord_loss = (responsible * (xy_err + wh_err)).sum()
        obj_loss = (responsible * (pred_conf - 1) ** 2).sum()
        noobj_loss = ((1 - responsible) * pred_conf**2).sum()
        class_loss = (obj.unsqueeze(-1) * (pred[..., :C] - target[..., :C]) ** 2).sum()

        total = (self.lambda_coord * coord_loss + obj_loss + self.lambda_noobj * noobj_loss + class_loss) / N
        self.last_components = {
            "coord": coord_loss.item() / N,
            "obj": obj_loss.item() / N,
            "noobj": noobj_loss.item() / N,
            "class": class_loss.item() / N,
        }
        return total
