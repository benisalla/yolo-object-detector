# legacy/

The two 2023 notebooks this project started from, kept for reference only. The maintained code is
the PyTorch package in [`src/yolov1`](../src/yolov1).

| file | what it was |
|---|---|
| `yolo-v1-local-final.ipynb` | the local notebook: Mapillary traffic signs in YOLO text format, training, webcam demo |
| `yolo-v1-kaggle-colab.ipynb` | the same code prepared for Kaggle / Colab (dataset unzip + CSV creation cells) |
| `README_original.md` | the original README |

## What changed, and why

Problems found in the notebooks, and how the package handles them:

| Notebook | Effect | Now |
|---|---|---|
| `test_size = [train_size[0], 0.2 * N]`, `val_size = [test_size[0], 0.2 * N]` when writing the CSV splits | the "test" and "validation" CSVs were the first 20% of the *training* CSV, so every reported mAP was measured on training images | `yolo` datasets are split by a seeded shuffle into disjoint train / validation sets (or use `images/train`, `images/val` folders); the unit tests check the split is disjoint |
| `mean_average_precision(..., n_class=20)` left at the default ("i should change this later") with 76 classes | classes 20 to 75 were never counted in the mAP | `num_classes` always comes from the dataset |
| mAP at IoU 0.2, computed on the test loader but printed as "Train mAP" | numbers were not comparable with anything | mAP@0.5 (VOC style, all-point interpolation), on the validation split, reported as such |
| the test `DataLoader` had `drop_last=True` | up to `batch_size - 1` test images were silently skipped | no `drop_last` for evaluation |
| `i, j = int(S * y), int(S * x)` | an object centred on the right or bottom edge (`x == 1.0`) raised `IndexError` | the cell index is clamped to `S - 1` |
| label parsing `int(x)` on strings like `"3.0"` | `ValueError` on labels written with a decimal class id | every value is parsed as float, class ids cast afterwards |
| `Image.open(path)` without `.convert("RGB")` | grayscale / RGBA images crashed the batch collate | always converted to RGB |
| `W, H, _ = frame.shape` in the webcam demo | OpenCV frames are `(height, width, channels)`: width and height were swapped | `PIL.Image.size` is `(width, height)`; one `draw_boxes` for files, grids and the camera |
| `cv2.rectangle(frame, (x, y), (x + w, y + h))` with `x, y` the box **centre** | every drawn box was shifted by half its width and height | centre → corners before drawing |
| `for idx in range(len(bboxes)): bboxes = non_max_suppression(bboxes[idx], ...)` | the loop variable overwrote the list it iterated; it only worked for a batch of one | NMS per image in `postprocess()` |
| decoding kept one of the two boxes per cell and used the raw objectness as the score | the second predictor was never used at test time; confidence ignored the class probability | every box is decoded with the paper's class-specific score `confidence × P(class)`, then per-class NMS |
| the loss hard-coded two boxes and applied the no-object term only in empty cells | `B` was a parameter that did nothing; the non-responsible box of an object cell was never pushed to confidence 0 (paper eq. 3 does) | `YoloLoss` works for any `B` and follows the paper: every non-responsible predictor gets the λ_noobj term |
| loss summed over the batch, `lr = 2e-5` with Adam | the learning rate depended on the batch size | loss divided by the batch size; AdamW with warmup + cosine schedule |
| `nn.Dropout(0.0)` and a 496-unit hidden layer | the paper uses 4096 units and dropout 0.5 | `--hidden` and `--dropout` flags, paper values by default |
| training was a single notebook cell with the `train_fn(...)` call commented out and checkpoint saving commented out | no validation-based model selection, no history | `yolov1-train` saves `best.pt` (highest validation mAP), `last.pt`, `history.json` and curves, and can `--resume` |
| no tests | nothing caught the problems above | 31 pytest checks, including an overfit sanity check |
