"""All settings, grouped in three dataclasses. The CLI flags of yolov1-train map 1:1 to these fields."""

from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset: str = "shapes"  # shapes | mnist | voc | yolo
    root: str = "data"  # download / dataset folder (mnist, voc, yolo)
    image_size: int = 448  # network input; must be divisible by the architecture stride (64 or 16)
    grid_size: int = 7  # S: the image is split in S x S cells
    num_boxes: int = 2  # B: boxes predicted per cell
    train_size: int = 2000  # shapes / mnist: generated training images
    val_size: int = 200  # shapes / mnist: generated validation images
    max_objects: int = 3  # shapes / mnist: objects per image (1..max_objects)
    val_fraction: float = 0.1  # yolo folder without a train/val split
    classes: list[str] | None = None  # yolo: class names (else classes.txt or "0", "1", ...)
    images_dir: str = "images"  # yolo: image folder, relative to root
    labels_dir: str = "labels"  # yolo: label folder, relative to root
    voc_year: str = "2012"  # voc: 2007 or 2012
    augment: bool = True  # flip, random scale/translate and color jitter on the training split
    num_workers: int = 2
    seed: int = 0


@dataclass
class ModelConfig:
    arch: str = "yolov1"  # yolov1 = the 24-conv paper network (stride 64) | small = 6 convs, stride 16
    width: float = 1.0  # channel multiplier (0.25 for quick CPU runs)
    hidden: int = 4096  # units of the first fully-connected layer (paper: 4096)
    dropout: float = 0.5  # paper: 0.5 after the first fully-connected layer


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 1
    lambda_coord: float = 5.0  # paper weights
    lambda_noobj: float = 0.5
    grad_clip: float = 10.0
    conf_threshold: float = 0.1  # evaluation: keep boxes with class score above this before NMS
    nms_iou: float = 0.5  # evaluation: NMS IoU threshold
    map_iou: float = 0.5  # evaluation: a prediction is a true positive above this IoU
    amp: bool = True  # mixed precision on CUDA
    device: str = "auto"
    output_dir: str = "checkpoints/run"
    overfit: int = 0  # >0: train and validate on this many training samples (sanity check)
    log_every: int = 50
    plot_every: int = 1  # redraw history.png / predictions.png every N epochs (0: only at the end)
    seed: int = 0
