"""The `yolov1-predict` command: detect objects in image files or a webcam stream with a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from yolov1.data.transforms import eval_transforms
from yolov1.train import load_model, postprocess
from yolov1.viz import draw_boxes


class Detector:
    def __init__(self, checkpoint: str | Path, device: str = "cpu", conf_threshold: float = 0.3, nms_iou: float = 0.5):
        self.model, self.info, self.configs = load_model(checkpoint, device)
        self.device, self.conf_threshold, self.nms_iou = torch.device(device), conf_threshold, nms_iou
        self.transforms = eval_transforms(self.info.image_size)
        self.class_names = self.info.class_names

    @torch.no_grad()
    def detect(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> (K, 6) [class, score, cx, cy, w, h], coordinates relative to the original image."""
        pixels, _ = self.transforms(image.convert("RGB"), torch.zeros(0, 5))
        output = self.model(pixels.unsqueeze(0).to(self.device))
        return postprocess(output, self.info, self.conf_threshold, self.nms_iou)[0]

    def to_records(self, detections: torch.Tensor, width: int, height: int) -> list[dict]:
        records = []
        for cls, score, cx, cy, w, h in detections.tolist():
            records.append(
                {
                    "class": self.class_names[int(cls)],
                    "score": round(score, 4),
                    "box_xyxy": [
                        round((cx - w / 2) * width, 1),
                        round((cy - h / 2) * height, 1),
                        round((cx + w / 2) * width, 1),
                        round((cy + h / 2) * height, 1),
                    ],
                }
            )
        return records


def run_camera(detector: Detector, camera: int = 0) -> None:
    import cv2  # optional dependency
    import numpy as np

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {camera}")
    print("press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        drawn = draw_boxes(image, detector.detect(image), detector.class_names)
        cv2.imshow("YOLOv1", cv2.cvtColor(np.array(drawn), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect objects with a trained YOLOv1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="*", default=[], help="image files")
    parser.add_argument("--camera", type=int, default=None, help="webcam index for live detection (needs opencv)")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    detector = Detector(args.checkpoint, args.device, args.conf_threshold, args.nms_iou)
    if args.camera is not None:
        run_camera(detector, args.camera)
        return
    if not args.images:
        parser.error("give --images or --camera")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in args.images:
        image = Image.open(path).convert("RGB")
        detections = detector.detect(image)
        draw_boxes(image, detections, detector.class_names).save(out / f"{Path(path).stem}_pred.jpg")
        results[path] = detector.to_records(detections, *image.size)
        found = ", ".join(f"{r['class']} {r['score']:.2f}" for r in results[path])
        print(f"{path}: {len(detections)} objects {found}")
    (out / "detections.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out}/detections.json and {len(results)} annotated images")


if __name__ == "__main__":
    main()
