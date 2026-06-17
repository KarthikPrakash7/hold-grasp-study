from __future__ import annotations
from pathlib import Path
from holdgrasp.models.base import Detection, Detector


class GroundingDINODetector(Detector):
    """Zero-shot detector via Grounding DINO (Apache 2.0). GPU required for inference."""

    MODEL_ID = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, box_threshold: float = 0.35, text_threshold: float = 0.25):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.MODEL_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, image_path: Path, classes: list[str]) -> list[Detection]:
        from PIL import Image
        import torch
        # GDINO text format: "class1. class2. class3."
        text = ". ".join(classes) + "."
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        detections: list[Detection] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            matched = _closest_class(label, classes)
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    label=matched,
                    score=float(score),
                    image_id=Path(image_path).stem,
                )
            )
        return detections


def _closest_class(raw_label: str, classes: list[str]) -> str:
    """Map GDINO output token to nearest class name by substring match."""
    raw = raw_label.lower()
    for cls in classes:
        if cls in raw or raw in cls:
            return cls
    return raw
