# Testing

```bash
pip install -e '.[dev]'
pytest            # CPU only, no network, under a minute
ruff check src tests
```

| file | what it checks |
|---|---|
| `tests/test_boxes.py` | IoU and format conversions by hand, target encoding puts the box in the right cell with the right offsets, `cx = 1.0` does not overflow the grid, one object per cell, decode inverts encode, unbounded outputs are clamped, NMS drops same-class duplicates only and respects the score threshold |
| `tests/test_model.py` | the `yolov1` architecture has 24 conv + 2 FC layers, stride 64 and the paper's layer sizes; the `small` one has stride 16 and the right output shape; every parameter gets a finite gradient; width multiplier; invalid image sizes / names raise; batch items are independent in eval mode |
| `tests/test_loss.py` | a perfect prediction has zero loss, each loss component computed by hand, mean over the batch, the no-object term hits every non-responsible box, the responsible box is the best-IoU one, a gradient step lowers the loss |
| `tests/test_metrics.py` | perfect detections give mAP 1 and skip classes without ground truth, no detections give 0, AP computed by hand on a 3-detection example, a duplicate detection is a false positive, the IoU threshold is honoured |
| `tests/test_data.py` | shapes are deterministic, valid and differ between splits; flips mirror the boxes; after a random scale / translate the box still frames the shape's pixels; `build_datasets` + `collate_fn` shapes and target count; a YOLO-format folder written to disk is split into disjoint train / validation sets and class names are read from `classes.txt` or `--classes` |
| `tests/test_train.py` | **overfit sanity check** (a small network memorises 4 images: loss falls by more than 10× and mAP@0.5 goes above 0.9), all output files are written, `--resume` continues the history, checkpoint reload, `yolov1-evaluate` and `yolov1-predict` work end to end, CLI flags reach the configs, learning-rate schedule values |

`test_overfit_a_few_samples` is the slowest test (about 20 s) and the most important one: a
detector that cannot memorise four images has a bug in its data, loss or decoding.

## Checks on real data

The unit tests cannot show that the model learns real images. That is what the simple-data runs
in the [README](../README.md#results) are for: `shapes` (synthetic) and `mnist` (handwritten
digits). Run them again after changing the model, loss or data pipeline:

```bash
yolov1-train --dataset shapes --image-size 112 --arch small --hidden 512 \
             --train-size 2000 --val-size 200 --epochs 20 --batch-size 32 --lr 1e-3 \
             --output-dir checkpoints/shapes
yolov1-train --dataset mnist --root data --image-size 112 --arch small --hidden 512 --dropout 0.1 \
             --no-augment --train-size 3000 --val-size 300 --epochs 30 --batch-size 32 --lr 1e-3 \
             --output-dir checkpoints/mnist
yolov1-evaluate --checkpoint checkpoints/mnist/best.pt
```
