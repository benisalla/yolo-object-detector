import random

import numpy as np
import torch
from PIL import Image

from yolov1.config import DataConfig
from yolov1.data import build_datasets, collate_fn
from yolov1.data.shapes import ShapesDetection
from yolov1.data.transforms import RandomHorizontalFlip, RandomScaleTranslate, Resize, eval_transforms
from yolov1.data.yolo_txt import build_yolo_txt


def test_shapes_are_deterministic_and_valid():
    a, b = ShapesDetection(10, image_size=64, seed=3), ShapesDetection(10, image_size=64, seed=3)
    for i in range(10):
        img, boxes = a[i]
        assert img.size == (64, 64) and 1 <= len(boxes) <= 3
        assert torch.equal(boxes, b[i][1])
        xyxy_ok = (boxes[:, 1] - boxes[:, 3] / 2 >= -1e-6).all() and (boxes[:, 2] + boxes[:, 4] / 2 <= 1 + 1e-6).all()
        assert xyxy_ok and (boxes[:, 0] < 3).all()
    assert not torch.equal(a[0][1], ShapesDetection(10, image_size=64, seed=3, split="val")[0][1])


def test_flip_mirrors_boxes():
    img, boxes = ShapesDetection(1, image_size=64)[0]
    random.seed(0)
    flipped, fboxes = RandomHorizontalFlip(p=1.0)(img, boxes)
    assert torch.allclose(fboxes[:, 1], 1 - boxes[:, 1]) and torch.equal(fboxes[:, 2:], boxes[:, 2:])
    assert flipped.size == img.size


def test_scale_translate_keeps_boxes_on_the_object():
    """After a random crop, the box must still frame the pixels painted in the shape's colour."""
    random.seed(1)
    img, boxes = ShapesDetection(1, image_size=96, max_objects=1)[0]
    rgb = np.array(img).reshape(-1, 3)
    shape_rgb = rgb[rgb.sum(1).argmin()]  # the (flat) shape colour: darker than the background and the padding
    for _ in range(5):
        out, ob = RandomScaleTranslate(amount=0.2, min_visible=0.0)(img, boxes)
        out, ob = Resize(96)(out, ob)
        assert ob.shape[0] == 1
        painted = torch.from_numpy((np.abs(np.array(out).astype(int) - shape_rgb) <= 3).all(-1))
        cx, cy, w, h = ob[0, 1:].tolist()
        x1, y1 = int((cx - w / 2) * 96), int((cy - h / 2) * 96)
        x2, y2 = int((cx + w / 2) * 96 + 0.999), int((cy + h / 2) * 96 + 0.999)
        inside = torch.zeros(96, 96, dtype=torch.bool)
        inside[max(y1, 0) : y2, max(x1, 0) : x2] = True
        assert painted[inside].sum() > 0.5 * painted.sum()  # most shape pixels are inside the box...
        assert painted[~inside].sum() <= 0.02 * painted.sum()  # ...and (almost) none outside


def test_build_datasets_and_collate():
    cfg = DataConfig(dataset="shapes", image_size=64, grid_size=4, train_size=6, val_size=3, num_workers=0)
    ds = build_datasets(cfg)
    assert ds.class_names == ["circle", "square", "triangle"] and len(ds.train) == 6 and len(ds.val) == 3
    pixels, grid, boxes = collate_fn([ds.val[i] for i in range(3)])
    assert pixels.shape == (3, 3, 64, 64) and grid.shape == (3, 4, 4, 13) and len(boxes) == 3
    assert grid[..., 3].sum() == sum(len(b) for b in boxes)  # one target per box (no two shapes share a cell)
    overfit = build_datasets(cfg, overfit=2)
    assert len(overfit.train) == 2 and len(overfit.val) == 2


def test_yolo_txt_folder_split_is_disjoint(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for i in range(10):
        Image.new("RGB", (32, 24), (i * 20, 0, 0)).save(tmp_path / "images" / f"img{i}.jpg")
        (tmp_path / "labels" / f"img{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.25 0.5\n1 0.1 0.1 0.1 0.1\n")
    (tmp_path / "classes.txt").write_text("sign\nlight\n")
    train, val = build_yolo_txt(tmp_path, val_fraction=0.2, seed=0)
    assert train.class_names == ["sign", "light"] and len(train) == 8 and len(val) == 2
    assert not {p.name for p in train.images} & {p.name for p in val.images}
    img, boxes = train[0]
    assert img.size == (32, 24) and boxes.shape == (2, 5)
    pixels, grid, _ = eval_transforms(64)(img, boxes) + (None,)  # pixel tensor + untouched boxes
    assert pixels.shape == (3, 64, 64)
    train2, _ = build_yolo_txt(tmp_path, classes=["a", "b", "c"], seed=0)
    assert train2.class_names == ["a", "b", "c"]
