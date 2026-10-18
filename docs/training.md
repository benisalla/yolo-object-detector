# Training

```bash
pip install -e .
yolov1-train --help
```

Every flag of `yolov1-train` maps to a field of the three dataclasses in
[`src/yolov1/config.py`](../src/yolov1/config.py): `DataConfig`, `ModelConfig`, `TrainConfig`.

## Datasets

| `--dataset` | classes | images | download | notes |
|---|---|---|---|---|
| `shapes` | 3: circle, square, triangle | generated on the fly, deterministic | none | 1 to `--max-objects` shapes per image; exact boxes |
| `mnist` | 10 digits | MNIST digits pasted on a 112 × 112 canvas | 12 MB (torchvision) | real handwriting, exact boxes |
| `voc` | 20 | PASCAL VOC `--voc-year 2012` (or 2007) | ~2 GB (torchvision) | the benchmark of the paper |
| `yolo` | from `classes.txt` or `--classes` | any folder in the YOLO text format | your data | Mapillary traffic signs (the 2023 project), Roboflow exports... |

### YOLO text format (`--dataset yolo --root <folder>`)

```
<root>/
  images/            name.jpg            or  images/train/, images/val/
  labels/            name.txt            or  labels/train/, labels/val/
  classes.txt        one class name per line (optional)
```

Each line of a label file is `class cx cy w h`, relative to the image size. With `train/` and
`val/` sub-folders the split is used as is; otherwise `--val-fraction` of the images are held out
by a seeded shuffle (`--seed`). `--images-dir` / `--labels-dir` change the folder names (for
example the Kaggle Mapillary export with its `txts (YOLO)` folder).

### Augmentation (training split only, `--no-augment` to disable)

Random scale / translation of up to 20% of the image, horizontal flip and brightness /
saturation jitter, as in the paper. Boxes are moved and clipped with the image; a box that keeps
less than 30% of its area is dropped.

## Recipes

```bash
# CPU, minutes: synthetic shapes with the small network (112 px -> 7 x 7 grid)
yolov1-train --dataset shapes --image-size 112 --arch small --hidden 512 \
             --train-size 2000 --val-size 200 --epochs 20 --batch-size 32 --lr 1e-3 \
             --output-dir checkpoints/shapes

# CPU, under an hour: MNIST digits, 10 classes
yolov1-train --dataset mnist --root data --image-size 112 --arch small --hidden 512 \
             --train-size 3000 --val-size 300 --epochs 20 --batch-size 32 --lr 1e-3 \
             --output-dir checkpoints/mnist

# GPU: the paper network on PASCAL VOC 2012 (use the notebook on Kaggle / Colab)
yolov1-train --dataset voc --root data --image-size 448 --arch yolov1 \
             --epochs 135 --batch-size 32 --lr 1e-4 --warmup-epochs 3 --output-dir checkpoints/voc

# GPU: your own YOLO-format data (e.g. Mapillary traffic signs)
yolov1-train --dataset yolo --root data/mapillary --image-size 448 --arch yolov1 \
             --epochs 100 --batch-size 32 --lr 1e-4 --output-dir checkpoints/mapillary
```

`--image-size` must be a multiple of the architecture stride (64 for `yolov1`, 16 for `small`);
`--grid-size` is 7 by default and does not have to equal `image_size / stride`.

Sanity check before a long run: `--overfit 8` trains and validates on 8 training images with no
augmentation. The validation mAP should reach 1.0 in a few dozen epochs; if not, something is
wrong with the data.

## Outputs

Everything goes to `--output-dir`:

| file | content |
|---|---|
| `best.pt` | weights + optimizer + configs + class names at the epoch with the highest validation mAP |
| `last.pt` | same, last epoch (`--resume checkpoints/run/last.pt` continues the run) |
| `history.json` | per-epoch train loss and its four components, validation loss, mAP, learning rate, seconds |
| `history.png` | the curves (redrawn every `--plot-every` epochs) |
| `predictions.png` | 8 validation images: ground truth on top, predictions below |
| `best_metrics.json` | best epoch, mAP and per-class AP |

Evaluation uses `--conf-threshold` (0.1), `--nms-iou` (0.5) and `--map-iou` (0.5). The low
confidence threshold is deliberate: mAP integrates over the score, so cutting at a high threshold
only lowers the recall. For display and deployment use a higher threshold (`yolov1-predict`
defaults to 0.3).

## Evaluation and inference

```bash
# mAP of a checkpoint on the validation split, with a per-class table and a prediction grid
yolov1-evaluate --checkpoint checkpoints/shapes/best.pt --report report.json --predictions preds.png

# image files -> outputs/<name>_pred.jpg + outputs/detections.json
yolov1-predict --checkpoint checkpoints/mnist/best.pt --images a.png b.png --conf-threshold 0.3

# live webcam (pip install -e '.[camera]')
yolov1-predict --checkpoint checkpoints/mapillary/best.pt --camera 0
```

In Python:

```python
from PIL import Image
from yolov1.predict import Detector

detector = Detector("checkpoints/mnist/best.pt", conf_threshold=0.3)
boxes = detector.detect(Image.open("digits.png"))  # (K, 6) rows: class, score, cx, cy, w, h
print(detector.to_records(boxes, *Image.open("digits.png").size))
```

## Kaggle / Colab

[`notebooks/train_kaggle_colab.ipynb`](../notebooks/train_kaggle_colab.ipynb) embeds the
package (so it needs no GitHub access), lets you pick the dataset, trains with a live dashboard
and zips the result. It is generated from `src/` by:

```bash
pip install -e '.[dev]'
python notebooks/build_notebook.py . notebooks/train_kaggle_colab.ipynb
```

Regenerate it after changing anything in `src/yolov1`.
