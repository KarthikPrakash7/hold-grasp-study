from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from holdgrasp.models.base import Detection, HOLD_CLASSES
from holdgrasp.eval.matching import MatchResult


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    count_error: float = 0.0


def load_coco_gt(path: Path) -> dict[str, list[Detection]]:
    """Load COCO JSON. Converts [x,y,w,h] bbox to (x1,y1,x2,y2). Returns {image_stem: [Detection]}."""
    data = json.loads(Path(path).read_text())
    id_to_stem = {img["id"]: Path(img["file_name"]).stem for img in data["images"]}
    id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    result: dict[str, list[Detection]] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        stem = id_to_stem[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        det = Detection(
            box=(x, y, x + w, y + h),
            label=id_to_name[ann["category_id"]],
            score=1.0,
            image_id=stem,
        )
        result.setdefault(stem, []).append(det)
    return result


def compute_metrics(result: MatchResult, classes: list[str]) -> dict[str, ClassMetrics]:
    tp: dict[str, int] = {c: 0 for c in classes}
    fp: dict[str, int] = {c: 0 for c in classes}
    fn: dict[str, int] = {c: 0 for c in classes}

    for gt_d, pred_d in result.matched:
        if gt_d.label == pred_d.label:
            tp[pred_d.label] += 1
        else:
            fn[gt_d.label] += 1
            fp[pred_d.label] += 1

    for gt_d in result.unmatched_gt:
        fn[gt_d.label] += 1

    for pred_d in result.unmatched_pred:
        fp[pred_d.label] += 1

    metrics: dict[str, ClassMetrics] = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[c] = ClassMetrics(precision=prec, recall=rec, f1=f1)
    return metrics


def confusion_matrix(result: MatchResult, classes: list[str]) -> np.ndarray:
    """Shape (len(classes), len(classes)). Row = GT class, col = predicted class."""
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for gt_d, pred_d in result.matched:
        if gt_d.label in idx and pred_d.label in idx:
            cm[idx[gt_d.label], idx[pred_d.label]] += 1
    return cm


def count_error_per_image(
    gt_by_image: dict[str, list[Detection]],
    pred_by_image: dict[str, list[Detection]],
    classes: list[str],
) -> dict[str, dict[str, int]]:
    """Returns {image_id: {class: predicted_count - gt_count}}."""
    all_ids = set(gt_by_image) | set(pred_by_image)
    errors: dict[str, dict[str, int]] = {}
    for img_id in all_ids:
        gt_counts = {c: 0 for c in classes}
        pred_counts = {c: 0 for c in classes}
        for d in gt_by_image.get(img_id, []):
            if d.label in gt_counts:
                gt_counts[d.label] += 1
        for d in pred_by_image.get(img_id, []):
            if d.label in pred_counts:
                pred_counts[d.label] += 1
        errors[img_id] = {c: pred_counts[c] - gt_counts[c] for c in classes}
    return errors
