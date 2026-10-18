import pytest
import torch
from torch import nn

from yolov1.config import ModelConfig
from yolov1.model import YoloV1


def test_paper_architecture():
    model = YoloV1(ModelConfig(), num_classes=20, image_size=448, grid_size=7, num_boxes=2)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(convs) == 24 and len(linears) == 2 and model.stride == 64
    assert linears[0].in_features == 1024 * 7 * 7 and linears[0].out_features == 4096
    assert linears[1].out_features == 7 * 7 * (20 + 2 * 5)


def test_small_architecture_output_and_gradients():
    model = YoloV1(ModelConfig(arch="small", width=0.5, hidden=64), num_classes=3, image_size=64, grid_size=4)
    assert model.stride == 16
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 4, 4, 3 + 10)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_width_multiplier_and_validation():
    small = YoloV1(ModelConfig(arch="small", width=0.25, hidden=32), 3, 64, 4).num_parameters()
    big = YoloV1(ModelConfig(arch="small", width=1.0, hidden=32), 3, 64, 4).num_parameters()
    assert big > 4 * small
    with pytest.raises(ValueError):
        YoloV1(ModelConfig(arch="small"), 3, image_size=100, grid_size=4)  # 100 is not a multiple of 16
    with pytest.raises(ValueError):
        YoloV1(ModelConfig(arch="resnet"), 3)


def test_batch_items_are_independent_in_eval_mode():
    model = YoloV1(ModelConfig(arch="small", width=0.25, hidden=32), 3, 64, 4).eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        full, single = model(x), model(x[:1])
    assert torch.allclose(full[:1], single, atol=1e-5)
