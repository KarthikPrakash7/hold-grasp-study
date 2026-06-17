from __future__ import annotations
import re
from pathlib import Path
from holdgrasp.models.base import Detection, Detector

_MODEL_ID = "nvidia/locate-anything-3b"
_DETECT_TEMPLATE = "Locate all {cls} climbing holds in this image. Return bounding boxes."
_GENERIC_PROMPT = "Locate all climbing holds in this image. Return bounding boxes."
_CLASSIFY_PROMPT = (
    "What type of climbing hold is this? "
    "Choose from: jug, crimp, sloper, pinch, pocket. Answer with one word only."
)
# Matches [x1, y1, x2, y2] float groups in model output text
_BOX_RE = re.compile(
    r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
)


def _load_model(device: str):
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch
    processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    return processor, model


def _generate(processor, model, image, prompt: str, device: str) -> str:
    import torch
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(output[0], skip_special_tokens=True)


def _parse_boxes(text: str, label: str, image_id: str) -> list[Detection]:
    return [
        Detection(
            box=(float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))),
            label=label,
            score=0.5,  # LocateAnything does not output confidence scores
            image_id=image_id,
        )
        for m in _BOX_RE.finditer(text)
    ]


def _parse_class(text: str, classes: list[str]) -> str:
    low = text.strip().lower()
    for cls in classes:
        if cls in low:
            return cls
    return "unknown"


class LocateAnythingZeroShot(Detector):
    """LA-zero: prompt each class individually. NVIDIA non-commercial license."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor, self.model = _load_model(device)

    def detect(self, image_path: Path, classes: list[str]) -> list[Detection]:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        detections: list[Detection] = []
        for cls in classes:
            prompt = _DETECT_TEMPLATE.format(cls=cls)
            text = _generate(self.processor, self.model, image, prompt, self.device)
            detections.extend(_parse_boxes(text, cls, Path(image_path).stem))
        return detections


class LocateAnythingTwoStage(Detector):
    """LA-two: detect all holds → classify each crop. NVIDIA non-commercial license."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor, self.model = _load_model(device)

    def detect(self, image_path: Path, classes: list[str]) -> list[Detection]:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        # Stage 1: locate all holds
        text = _generate(self.processor, self.model, image, _GENERIC_PROMPT, self.device)
        raw_boxes = [
            (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
            for m in _BOX_RE.finditer(text)
        ]
        # Stage 2: classify each crop
        detections: list[Detection] = []
        for box in raw_boxes:
            x1, y1, x2, y2 = box
            crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
            cls_text = _generate(self.processor, self.model, crop, _CLASSIFY_PROMPT, self.device)
            label = _parse_class(cls_text, classes)
            detections.append(
                Detection(box=box, label=label, score=0.5, image_id=Path(image_path).stem)
            )
        return detections
