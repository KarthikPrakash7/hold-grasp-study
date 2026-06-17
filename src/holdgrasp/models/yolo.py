from __future__ import annotations
from pathlib import Path
from holdgrasp.models.base import Detection, Detector


class YOLODetector(Detector):
    """YOLOv8/11 adapter. Requires pip install -e ".[gpu]" on inference machine."""

    def __init__(self, checkpoint: str | Path, conf_threshold: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(str(checkpoint))
        self.conf_threshold = conf_threshold

    def detect(self, image_path: Path, classes: list[str]) -> list[Detection]:
        results = self.model.predict(
            str(image_path), conf=self.conf_threshold, verbose=False
        )
        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                if label not in classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        box=(x1, y1, x2, y2),
                        label=label,
                        score=float(box.conf[0]),
                        image_id=Path(image_path).stem,
                    )
                )
        return detections
