"""Build notebooks/train_kaggle_colab.ipynb: a self-contained training notebook.

Every module of src/yolov1 is embedded verbatim (as %%writefile cells), so the notebook runs the
exact package code and needs no GitHub access. Regenerate after changing src/yolov1:

    python notebooks/build_notebook.py . notebooks/train_kaggle_colab.ipynb
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
MODULES = [
    "__init__", "config", "boxes", "model", "loss", "metrics", "viz",
    "data/transforms", "data/shapes", "data/mnist", "data/voc", "data/yolo_txt", "data/__init__",
    "train", "evaluate", "predict",
]  # fmt: skip

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------
md("""
# YOLOv1 object detector: training notebook (Kaggle & Colab)

Train the YOLOv1 detector with a **free Kaggle or Colab GPU** on synthetic shapes, MNIST
digits, PASCAL VOC or any dataset in the YOLO text format (for example the Mapillary
traffic-sign export). The notebook is **self-contained**: section 1 writes the `yolov1`
package (identical to the repo's `src/`), and training goes through the same `train()`
function as the `yolov1-train` command.

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings → Accelerator → GPU T4 x2* (or P100) | *Runtime → Change runtime type → T4 GPU* |
| Internet | *Settings → Internet → On* (to download data) | on by default |
| Checkpoints kept in | `/kaggle/working` (saved with "Save Version") | Google Drive if `USE_GOOGLE_DRIVE = True` |

**Sections:** 1. The package · 2. Settings · 3. Data · 4. Model · 5. Training with a live
dashboard · 6. Results · 7. Download
""")

# ---------------------------------------------------------------------------
md("""
## 1. The package

The cells below write `src/yolov1/*.py`, copied from the repo at build time. Read them in this
order: `config` → `boxes` (target encoding / decoding, NMS) → `model` → `loss` → `metrics` →
`data` → `train`.
""")
code("""
import os

os.makedirs("src/yolov1/data", exist_ok=True)
""")
for module in MODULES:
    source = (ROOT / "src" / "yolov1" / f"{module}.py").read_text()
    code(f"%%writefile src/yolov1/{module}.py\n{source}")
code("""
import sys

sys.path.insert(0, "src")
for name in [m for m in sys.modules if m.startswith("yolov1")]:
    del sys.modules[name]  # pick up the files just written if this cell is re-run
import yolov1.train as T

print("yolov1 ready")
""")

# ---------------------------------------------------------------------------
md("""
## 2. Settings

| `DATASET` | classes | images | download | what for |
|---|---|---|---|---|
| `"shapes"` | 3 | generated, exact boxes | none | prove the detector learns (minutes) |
| `"mnist"` | 10 | MNIST digits pasted on a canvas | 12 MB | real pixels, exact boxes |
| `"voc"` | 20 | PASCAL VOC 2012 | ~2 GB | the benchmark of the paper (hours) |
| `"yolo"` | yours | any YOLO text-format folder | your data | e.g. Mapillary traffic signs (set `YOLO_ROOT`) |

`SMOKE_TEST = True` runs a small version to check the whole notebook in a few minutes. Keep it
`False` for real training.
""")
code("""
import dataclasses
from pathlib import Path

import torch

from yolov1.config import DataConfig, ModelConfig, TrainConfig

SMOKE_TEST = False          # True = small quick check of the whole notebook
DATASET = "voc"             # "shapes" | "mnist" | "voc" | "yolo"
USE_GOOGLE_DRIVE = False    # Colab only: keep checkpoints on Drive (survive disconnects)
RESUME = True               # continue from last.pt in OUTPUT_DIR if it exists

# "yolo" dataset: a folder with images/ + labels/ (+ classes.txt), or train/ val/ sub-folders.
# Kaggle: add the dataset to the notebook and point YOLO_ROOT at /kaggle/input/<dataset>.
YOLO_ROOT = "/kaggle/input/traffic-signs-dataset-mapillary-and-dfg"
IMAGES_DIR, LABELS_DIR = "images", "txts (YOLO)"
YOLO_CLASSES = None         # list of class names, or None to read classes.txt / infer the ids

# --- where checkpoints go --------------------------------------------------------
ON_KAGGLE = Path("/kaggle/working").exists()
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ
if ON_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_DIR = Path(f"/content/drive/MyDrive/yolo-object-detector/{DATASET}")
elif ON_KAGGLE:
    OUTPUT_DIR = Path(f"/kaggle/working/checkpoints/{DATASET}")
else:
    OUTPUT_DIR = Path(f"checkpoints/{DATASET}")

# --- per-dataset presets -----------------------------------------------------------
PRESETS = {
    "shapes": dict(data=dict(image_size=112, train_size=4000, val_size=400),
                   model=dict(arch="small", hidden=512), epochs=25, batch_size=64, lr=1e-3),
    "mnist": dict(data=dict(image_size=112, train_size=20000, val_size=1000, root="data", augment=False),
                  model=dict(arch="small", hidden=512, dropout=0.3), epochs=15, batch_size=64, lr=1e-3),
    "voc": dict(data=dict(image_size=448, root="data"),
                model=dict(arch="yolov1", hidden=4096), epochs=135, batch_size=32, lr=1e-4),
    "yolo": dict(data=dict(image_size=448, root=YOLO_ROOT, images_dir=IMAGES_DIR, labels_dir=LABELS_DIR,
                           classes=YOLO_CLASSES),
                 model=dict(arch="yolov1", hidden=4096), epochs=100, batch_size=32, lr=1e-4),
}
preset = PRESETS[DATASET]
data_config = DataConfig(dataset=DATASET, num_workers=min(4, os.cpu_count() or 1), **preset["data"])
model_config = ModelConfig(**preset["model"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 1e9:.0f} GB)")
else:
    print("No GPU: enable one in the notebook settings (fine only for SMOKE_TEST / shapes).")

train_config = TrainConfig(
    output_dir=str(OUTPUT_DIR), epochs=preset["epochs"], batch_size=preset["batch_size"], lr=preset["lr"],
    warmup_epochs=2, device=device.type, log_every=25, plot_every=0,
)
if SMOKE_TEST:
    OUTPUT_DIR = OUTPUT_DIR.parent / "smoke-test"  # never mixed with a real run
    data_config = dataclasses.replace(data_config, train_size=64, val_size=16)
    model_config = dataclasses.replace(model_config, width=0.25, hidden=128)
    train_config = dataclasses.replace(
        train_config, output_dir=str(OUTPUT_DIR), epochs=3, batch_size=8, warmup_epochs=1, log_every=2,
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
resume = str(OUTPUT_DIR / T.LAST_CHECKPOINT) if RESUME and (OUTPUT_DIR / T.LAST_CHECKPOINT).exists() else None

for name, cfg in [("DataConfig", data_config), ("ModelConfig", model_config), ("TrainConfig", train_config)]:
    print(f"\\n{name}:")
    for key, value in dataclasses.asdict(cfg).items():
        print(f"  {key:16} {value}")
""")

# ---------------------------------------------------------------------------
md("""
## 3. Data

A look at a few training images with their ground-truth boxes (after augmentation). VOC and
MNIST are downloaded here the first time.
""")
code("""
import logging

import matplotlib.pyplot as plt

from yolov1.data import build_datasets
from yolov1.viz import draw_boxes, tensor_to_pil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
T.seed_everything(train_config.seed)
datasets = build_datasets(data_config)
class_names = datasets.class_names
print(f"{len(class_names)} classes | train {len(datasets.train):,} | val {len(datasets.val):,}")

fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
for ax, i in zip(axes, range(0, len(datasets.train), max(1, len(datasets.train) // 6))):
    pixels, _, boxes = datasets.train[i]
    ax.imshow(draw_boxes(tensor_to_pil(pixels), boxes, class_names, with_scores=False))
    ax.axis("off")
plt.show()
""")

# ---------------------------------------------------------------------------
md("## 4. Model")
code("""
info = T.DatasetInfo(class_names, data_config.image_size, data_config.grid_size, data_config.num_boxes)
model = T.build_model(model_config, info).to(device)
print(f"{model_config.arch} x{model_config.width}: {model.num_parameters() / 1e6:.1f}M parameters, "
      f"stride {model.stride}, output {info.grid_size}x{info.grid_size}x{info.num_classes + 5 * info.num_boxes}")
""")

# ---------------------------------------------------------------------------
md("""
## 5. Training with a live dashboard

Same loop as `yolov1-train`: AdamW, warmup + cosine schedule, mixed precision, gradient
clipping, the paper's loss. After every epoch it evaluates mAP@0.5 on the validation split and
saves `last.pt`, plus `best.pt` when the mAP improves.

**Interrupted, or hit the session limit?** Run the notebook again with `RESUME = True`: it
continues from `last.pt`.
""")
code("""
from IPython.display import clear_output

history = []


def draw_dashboard(event):
    history.append(event)
    clear_output(wait=True)
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_map, ax_parts) = plt.subplots(1, 3, figsize=(16, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], "o-", label="validation")
    ax_loss.set_title("YOLO loss")
    ax_map.plot(epochs, [h["val_map"] for h in history], "o-", color="tab:green")
    ax_map.set_title("validation mAP@0.5")
    ax_map.set_ylim(0, 1)
    for key in ("coord", "obj", "noobj", "class"):
        ax_parts.plot(epochs, [h["train_" + key] for h in history], label=key)
    ax_parts.set_title("train loss components")
    for ax in (ax_loss, ax_map, ax_parts):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    plt.show()
    print(f"epoch {event['epoch']}/{train_config.epochs} | train loss {event['train_loss']:.3f} | "
          f"val loss {event['val_loss']:.3f} | val mAP@0.5 {event['val_map']:.3f} "
          f"(best {event['best_map']:.3f}) | {event['seconds']:.0f}s")


best = T.train(data_config, model_config, train_config, resume=resume, callback=draw_dashboard)
print(f"Training finished. Best validation mAP@0.5 {best['map']:.3f} at epoch {best['epoch']}")
""")

# ---------------------------------------------------------------------------
md("""
## 6. Results

The best checkpoint on the validation split: mAP, per-class AP and a grid of predictions.
""")
code("""
from IPython.display import Image as IPImage, display

from yolov1 import evaluate

evaluate.main(["--checkpoint", str(OUTPUT_DIR / T.BEST_CHECKPOINT), "--report", str(OUTPUT_DIR / "report.json"),
               "--predictions", str(OUTPUT_DIR / "predictions.png"), "--device", device.type])
display(IPImage(str(OUTPUT_DIR / "predictions.png")))
display(IPImage(str(OUTPUT_DIR / "history.png")))
""")

# ---------------------------------------------------------------------------
md("""
## 7. Download the trained model

Zips `best.pt`, the report and the figures (`last.pt` is left out: it is only needed to
resume). Use it with the repo:

```bash
unzip yolov1-run.zip -d checkpoints/run
yolov1-predict --checkpoint checkpoints/run/best.pt --images street.jpg
yolov1-predict --checkpoint checkpoints/run/best.pt --camera 0
```
""")
code("""
import shutil
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "bundle"
    bundle.mkdir()
    keep = [T.BEST_CHECKPOINT, "report.json", "history.json", "history.png", "predictions.png", "best_metrics.json"]
    for name in keep:
        if (OUTPUT_DIR / name).exists():
            shutil.copy(OUTPUT_DIR / name, bundle / name)
    archive = shutil.make_archive(str(OUTPUT_DIR.parent / "yolov1-run"), "zip", bundle)
print("created", archive)

if ON_COLAB:
    from google.colab import files

    files.download(archive)
elif ON_KAGGLE:
    print("Kaggle: click 'Save Version', then download it from the notebook's Output tab.")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.validate(nb)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
