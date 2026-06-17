from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw
from holdgrasp.models.base import Detection, HOLD_CLASSES

_CLASS_COLORS: dict[str, str] = {
    "jug": "#2ecc71",
    "crimp": "#e74c3c",
    "sloper": "#3498db",
    "pinch": "#f39c12",
    "pocket": "#9b59b6",
}
_DEFAULT_COLOR = "#ffffff"


def draw_detections(
    image_path: Path,
    detections: list[Detection],
    out_path: Path,
    gt_detections: list[Detection] | None = None,
) -> None:
    """Pred boxes: solid 3px. GT boxes: concentric thin rings (dashed-like effect)."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    if gt_detections:
        for d in gt_detections:
            color = _CLASS_COLORS.get(d.label, _DEFAULT_COLOR)
            x1, y1, x2, y2 = d.box
            for offset in (0, 3, 6):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=color,
                    width=1,
                )
    for d in detections:
        color = _CLASS_COLORS.get(d.label, _DEFAULT_COLOR)
        x1, y1, x2, y2 = d.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, y1 + 2), f"{d.label} {d.score:.2f}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def error_gallery(
    image_paths: dict[str, Path],
    gt_by_image: dict[str, list[Detection]],
    pred_by_image: dict[str, list[Detection]],
    result_dir: Path,
    model_id: str,
    top_n: int = 5,
) -> None:
    """Save the top_n images by total absolute count error as annotated PNGs."""
    from holdgrasp.eval.metrics import count_error_per_image
    errors = count_error_per_image(gt_by_image, pred_by_image, HOLD_CLASSES)
    total_error = {
        img_id: sum(abs(v) for v in cls_errors.values())
        for img_id, cls_errors in errors.items()
    }
    worst = sorted(total_error, key=lambda k: total_error[k], reverse=True)[:top_n]
    gallery_dir = result_dir / "gallery" / model_id
    for img_id in worst:
        if img_id not in image_paths:
            continue
        draw_detections(
            image_paths[img_id],
            pred_by_image.get(img_id, []),
            gallery_dir / f"{img_id}_error.png",
            gt_detections=gt_by_image.get(img_id, []),
        )
