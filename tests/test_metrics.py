import pytest
import torch

from yolov1.metrics import average_precision, mean_average_precision

GT = [torch.tensor([[0, 0.5, 0.5, 0.2, 0.2], [1, 0.2, 0.2, 0.1, 0.1]]), torch.tensor([[0, 0.7, 0.7, 0.2, 0.2]])]


def test_perfect_detections():
    preds = [torch.cat([g[:, :1], torch.ones(len(g), 1), g[:, 1:]], 1) for g in GT]
    result = mean_average_precision(preds, GT, num_classes=3)
    assert result["map"] == pytest.approx(1.0) and set(result["ap"]) == {0, 1}  # class 2 has no ground truth


def test_no_detections_and_empty_images():
    assert mean_average_precision([torch.zeros(0, 6), torch.zeros(0, 6)], GT, 3)["map"] == 0.0
    assert mean_average_precision([torch.zeros(0, 6)], [torch.zeros(0, 5)], 3)["map"] == 0.0


def test_average_precision_by_hand():
    # class 0, two ground truth boxes. Detections by score: hit, miss, hit
    # -> precision 1, 1/2, 2/3 and recall 0.5, 0.5, 1
    preds = [
        torch.tensor([[0, 0.9, 0.5, 0.5, 0.2, 0.2], [0, 0.8, 0.1, 0.9, 0.1, 0.1]]),
        torch.tensor([[0, 0.7, 0.7, 0.7, 0.2, 0.2]]),
    ]
    result = mean_average_precision(preds, [GT[0][:1], GT[1]], num_classes=1)
    assert result["ap"][0] == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
    assert average_precision(torch.tensor([0.5, 0.5, 1.0]), torch.tensor([1.0, 0.5, 2 / 3])) == pytest.approx(5 / 6)


def test_duplicate_detection_is_a_false_positive():
    preds = [torch.tensor([[0, 0.9, 0.5, 0.5, 0.2, 0.2], [0, 0.8, 0.51, 0.5, 0.2, 0.2]]), torch.zeros(0, 6)]
    result = mean_average_precision(preds, [GT[0][:1], torch.zeros(0, 5)], num_classes=1)
    assert result["ap"][0] == pytest.approx(1.0)  # recall 1 is reached before the duplicate: precision envelope = 1
    preds = [torch.tensor([[0, 0.9, 0.51, 0.5, 0.2, 0.2], [0, 0.95, 0.9, 0.9, 0.05, 0.05]]), torch.zeros(0, 6)]
    result = mean_average_precision(preds, [GT[0][:1], torch.zeros(0, 5)], num_classes=1)
    assert result["ap"][0] == pytest.approx(0.5)  # the top-scored box is a false positive


def test_iou_threshold_matters():
    preds = [torch.tensor([[0, 0.9, 0.58, 0.5, 0.2, 0.2]]), torch.zeros(0, 6)]  # IoU with the gt ≈ 0.43
    targets = [GT[0][:1], torch.zeros(0, 5)]
    assert mean_average_precision(preds, targets, 1, iou_threshold=0.5)["map"] == 0.0
    assert mean_average_precision(preds, targets, 1, iou_threshold=0.4)["map"] == pytest.approx(1.0)
