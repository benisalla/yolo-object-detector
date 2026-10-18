import torch

from yolov1.boxes import encode_targets
from yolov1.loss import YoloLoss

S, B, C = 4, 2, 3


def perfect_prediction(target: torch.Tensor) -> torch.Tensor:
    """A prediction that reproduces the target in box slot 0 and leaves every other box at 0 confidence."""
    return target.clone()


def test_perfect_prediction_has_zero_loss():
    target = encode_targets(torch.tensor([[1, 0.3, 0.6, 0.2, 0.3]]), S, B, C).unsqueeze(0)
    loss_fn = YoloLoss(S, B, C)
    assert loss_fn(perfect_prediction(target), target).item() < 1e-4
    assert all(v < 1e-4 for v in loss_fn.last_components.values())


def test_components_and_batch_mean():
    target = encode_targets(torch.tensor([[1, 0.3, 0.6, 0.2, 0.3]]), S, B, C).unsqueeze(0)
    loss_fn = YoloLoss(S, B, C, lambda_coord=5.0, lambda_noobj=0.5)
    pred = torch.zeros_like(target)
    loss = loss_fn(pred, target)
    comp = loss_fn.last_components
    # zeros everywhere: coord error = x² + y² + (√w)² + (√h)², confidence error 1, class error 1, no-object error 0
    x, y, w, h = target[0, 2, 1, C + 1 : C + 5].tolist()
    assert abs(comp["coord"] - (x * x + y * y + w + h)) < 1e-4
    assert abs(comp["obj"] - 1.0) < 1e-6 and abs(comp["class"] - 1.0) < 1e-6 and comp["noobj"] == 0
    assert abs(loss.item() - (5.0 * comp["coord"] + 1.0 + 1.0)) < 1e-4
    doubled = loss_fn(torch.cat([pred, pred]), torch.cat([target, target]))
    assert abs(doubled.item() - loss.item()) < 1e-5  # mean over the batch


def test_noobj_penalises_every_non_responsible_box():
    target = torch.zeros(1, S, S, C + 5 * B)
    pred = torch.zeros_like(target)
    pred[..., C] = 1.0  # confidence 1 in box 0 of every empty cell
    pred[..., C + 5] = 1.0  # and in box 1
    loss_fn = YoloLoss(S, B, C, lambda_noobj=0.5)
    assert abs(loss_fn(pred, target).item() - 0.5 * 2 * S * S) < 1e-5


def test_responsible_box_is_the_best_iou():
    target = encode_targets(torch.tensor([[0, 0.3, 0.6, 0.2, 0.3]]), S, B, C).unsqueeze(0)
    pred = torch.zeros_like(target)
    row, col = int(0.6 * S), int(0.3 * S)
    pred[0, row, col, C + 5] = 1.0  # box 1 has the right confidence...
    pred[0, row, col, C + 6 : C + 10] = target[0, row, col, C + 1 : C + 5]  # ...and the right coordinates
    pred[0, row, col, :C] = target[0, row, col, :C]
    loss_fn = YoloLoss(S, B, C)
    assert loss_fn(pred, target).item() < 1e-4  # box 1 was chosen, so no coordinate or confidence error


def test_gradient_step_reduces_loss():
    target = encode_targets(torch.tensor([[2, 0.7, 0.2, 0.3, 0.2]]), S, B, C).unsqueeze(0)
    pred = torch.randn_like(target, requires_grad=True)
    loss_fn = YoloLoss(S, B, C)
    before = loss_fn(pred, target)
    before.backward()
    with torch.no_grad():
        pred -= 0.05 * pred.grad
    assert loss_fn(pred, target).item() < before.item()
