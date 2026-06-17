import json
import pytest
import numpy as np
from pathlib import Path
from holdgrasp.models.base import Detection, HOLD_CLASSES
from holdgrasp.eval.matching import MatchResult
from holdgrasp.eval.metrics import (
    ClassMetrics,
    compute_metrics,
    confusion_matrix,
    load_coco_gt,
    count_error_per_image,
)


def _d(label, score=1.0, box=(0.0, 0.0, 10.0, 10.0), image_id="img1"):
    return Detection(box=box, label=label, score=score, image_id=image_id)


def test_perfect_precision_recall():
    result = MatchResult(matched=[(_d("jug"), _d("jug"))])
    metrics = compute_metrics(result, HOLD_CLASSES)
    assert metrics["jug"].precision == pytest.approx(1.0)
    assert metrics["jug"].recall == pytest.approx(1.0)
    assert metrics["jug"].f1 == pytest.approx(1.0)


def test_false_positive_only():
    result = MatchResult(unmatched_pred=[_d("jug")])
    metrics = compute_metrics(result, HOLD_CLASSES)
    assert metrics["jug"].precision == pytest.approx(0.0)
    assert metrics["jug"].recall == pytest.approx(0.0)


def test_false_negative_only():
    result = MatchResult(unmatched_gt=[_d("jug")])
    metrics = compute_metrics(result, HOLD_CLASSES)
    assert metrics["jug"].recall == pytest.approx(0.0)
    assert metrics["jug"].precision == pytest.approx(0.0)


def test_class_mismatch_counts_fp_and_fn():
    # GT is jug, pred is crimp — matched by IoU but wrong label
    result = MatchResult(matched=[(_d("jug"), _d("crimp"))])
    metrics = compute_metrics(result, HOLD_CLASSES)
    assert metrics["jug"].recall == pytest.approx(0.0)    # jug FN
    assert metrics["crimp"].precision == pytest.approx(0.0)  # crimp FP


def test_confusion_matrix_shape():
    cm = confusion_matrix(MatchResult(), HOLD_CLASSES)
    assert cm.shape == (5, 5)


def test_confusion_matrix_correct_prediction():
    result = MatchResult(matched=[(_d("jug"), _d("jug"))])
    cm = confusion_matrix(result, HOLD_CLASSES)
    jug_idx = HOLD_CLASSES.index("jug")
    assert cm[jug_idx, jug_idx] == 1


def test_confusion_matrix_mismatch():
    result = MatchResult(matched=[(_d("jug"), _d("crimp"))])
    cm = confusion_matrix(result, HOLD_CLASSES)
    jug_idx = HOLD_CLASSES.index("jug")
    crimp_idx = HOLD_CLASSES.index("crimp")
    assert cm[jug_idx, crimp_idx] == 1


def test_load_coco_gt(tmp_path):
    coco = {
        "images": [{"id": 1, "file_name": "wall01.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "jug"}, {"id": 2, "name": "crimp"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 40, 60], "iscrowd": 0},
        ],
    }
    p = tmp_path / "test.json"
    p.write_text(json.dumps(coco))
    gt = load_coco_gt(p)
    assert "wall01" in gt
    assert len(gt["wall01"]) == 1
    d = gt["wall01"][0]
    assert d.label == "jug"
    assert d.box == (10.0, 20.0, 50.0, 80.0)  # [x,y,w,h] → (x1,y1,x2,y2)


def test_count_error_per_image():
    gt_by_image = {"img1": [_d("jug"), _d("jug")]}   # 2 jugs
    pred_by_image = {"img1": [_d("jug")]}              # 1 jug
    errors = count_error_per_image(gt_by_image, pred_by_image, HOLD_CLASSES)
    assert errors["img1"]["jug"] == -1   # pred(1) - gt(2)
    assert errors["img1"]["crimp"] == 0
