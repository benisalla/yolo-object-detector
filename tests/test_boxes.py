"""IoU, target encoding / decoding and NMS, checked by hand."""

import pytest
import torch

from yolov1.boxes import (
    box_iou,
    cxcywh_to_xyxy,
    decode_predictions,
    encode_targets,
    non_max_suppression,
    pairwise_iou,
    xyxy_to_cxcywh,
)


def test_format_conversion_roundtrip():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.1, 0.9, 0.2, 0.2]])
    assert torch.allclose(cxcywh_to_xyxy(boxes), torch.tensor([[0.4, 0.3, 0.6, 0.7], [0.0, 0.8, 0.2, 1.0]]))
    assert torch.allclose(xyxy_to_cxcywh(cxcywh_to_xyxy(boxes)), boxes)


def test_iou_by_hand():
    a = torch.tensor([0.5, 0.5, 0.4, 0.4])  # [0.3, 0.7]²
    b = torch.tensor([0.6, 0.6, 0.4, 0.4])  # [0.4, 0.8]²: intersection 0.3², union 2 * 0.16 - 0.09
    assert box_iou(a, b).item() == pytest.approx(0.09 / 0.23, abs=1e-5)
    assert box_iou(a, a).item() == pytest.approx(1.0, abs=1e-5)
    assert box_iou(a, torch.tensor([0.9, 0.9, 0.1, 0.1])).item() == 0.0
    xyxy = box_iou(cxcywh_to_xyxy(a), cxcywh_to_xyxy(b), box_format="xyxy")
    assert xyxy.item() == pytest.approx(0.09 / 0.23, abs=1e-5)
    assert pairwise_iou(torch.stack([a, b]), torch.stack([a, b])).shape == (2, 2)


def test_encode_puts_the_box_in_the_right_cell():
    S, B, C = 7, 2, 3
    target = encode_targets(torch.tensor([[2, 0.5, 0.2, 0.3, 0.1]]), S, B, C)
    assert target.shape == (S, S, C + 5 * B)
    row, col = int(0.2 * S), int(0.5 * S)
    assert target[row, col, 2] == 1 and target[row, col, C] == 1
    expected = torch.tensor([0.5 * S - col, 0.2 * S - row, 0.3 * S, 0.1 * S])
    assert torch.allclose(target[row, col, C + 1 : C + 5], expected)
    assert target[..., C].sum() == 1


def test_encode_edge_coordinates_do_not_overflow():
    target = encode_targets(torch.tensor([[0, 1.0, 1.0, 0.1, 0.1]]), 7, 2, 1)
    assert target[6, 6, 1] == 1


def test_encode_keeps_one_object_per_cell():
    boxes = torch.tensor([[0, 0.5, 0.5, 0.1, 0.1], [1, 0.52, 0.52, 0.1, 0.1]])
    target = encode_targets(boxes, 7, 2, 2)
    assert target[3, 3, 0] == 1 and target[3, 3, 1] == 0


def test_decode_inverts_encode():
    S, B, C = 7, 2, 3
    boxes = torch.tensor([[2, 0.5, 0.2, 0.3, 0.1], [0, 0.9, 0.8, 0.1, 0.2]])
    target = encode_targets(boxes, S, B, C)
    decoded = decode_predictions(target.unsqueeze(0), B, C)[0]
    assert decoded.shape == (S * S * B, 6)
    found = decoded[decoded[:, 1] > 0.5]
    assert found.shape[0] == 2
    for cls, cx, cy, w, h in boxes.tolist():
        row = found[found[:, 0] == cls][0]
        assert torch.allclose(row[2:], torch.tensor([cx, cy, w, h]), atol=1e-5)


def test_decode_clamps_unbounded_outputs():
    pred = torch.full((1, 2, 2, 3 + 10), -5.0)
    decoded = decode_predictions(pred, 2, 3)[0]
    assert (decoded[:, 1] == 0).all() and (decoded[:, 4:] >= 0).all()


def test_nms_removes_duplicates_of_the_same_class_only():
    dets = torch.tensor(
        [
            [0, 0.9, 0.5, 0.5, 0.2, 0.2],
            [0, 0.8, 0.51, 0.5, 0.2, 0.2],  # same object, lower score -> dropped
            [1, 0.7, 0.51, 0.5, 0.2, 0.2],  # other class -> kept
            [0, 0.6, 0.1, 0.1, 0.1, 0.1],  # far away -> kept
            [0, 0.05, 0.9, 0.9, 0.1, 0.1],  # below the score threshold
        ]
    )
    kept = non_max_suppression(dets, iou_threshold=0.5, score_threshold=0.1)
    assert torch.allclose(kept[:, 1], torch.tensor([0.9, 0.7, 0.6]))
    assert non_max_suppression(dets, 0.5, 0.95).shape == (0, 6)
