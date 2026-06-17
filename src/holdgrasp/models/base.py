from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


HOLD_CLASSES: list[str] = ["jug", "crimp", "sloper", "pinch", "pocket"]


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    label: str
    score: float
    image_id: str = ""

    def to_dict(self) -> dict:
        return {
            "box": list(self.box),
            "label": self.label,
            "score": self.score,
            "image_id": self.image_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Detection:
        return cls(
            box=tuple(d["box"]),
            label=d["label"],
            score=d["score"],
            image_id=d.get("image_id", ""),
        )


def iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class Detector(ABC):
    @abstractmethod
    def detect(self, image_path: Path, classes: list[str]) -> list[Detection]:
        """Return detections for image at image_path, one per detected object."""
