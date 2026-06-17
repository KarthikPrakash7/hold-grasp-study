import pytest
from holdgrasp.models.base import Detection
from holdgrasp.eval.matching import match_detections, MatchResult


def _d(x1, y1, x2, y2, label="jug", score=1.0, image_id="img"):
    return Detection(box=(x1, y1, x2, y2), label=label, score=score, image_id=image_id)


def test_perfect_match():
    gt = [_d(0, 0, 10, 10)]
    pred = [_d(0, 0, 10, 10)]
    result = match_detections(gt, pred)
    assert len(result.matched) == 1
    assert result.matched[0][0] is gt[0]
    assert result.matched[0][1] is pred[0]
    assert len(result.unmatched_gt) == 0
    assert len(result.unmatched_pred) == 0


def test_no_overlap():
    gt = [_d(0, 0, 10, 10)]
    pred = [_d(20, 20, 30, 30)]
    result = match_detections(gt, pred)
    assert len(result.matched) == 0
    assert len(result.unmatched_gt) == 1
    assert len(result.unmatched_pred) == 1


def test_below_threshold():
    # IoU ≈ 0.11 — below 0.5
    gt = [_d(0, 0, 10, 10)]
    pred = [_d(7, 0, 17, 10)]
    result = match_detections(gt, pred)
    assert len(result.matched) == 0


def test_greedy_best_score_first():
    gt = [_d(0, 0, 10, 10)]
    pred_high = _d(0, 0, 10, 10, score=0.9)
    pred_low = _d(1, 1, 11, 11, score=0.5)
    result = match_detections(gt, [pred_high, pred_low])
    assert result.matched[0][1] is pred_high
    assert len(result.unmatched_pred) == 1


def test_class_mismatch_still_matches():
    # Matching is class-agnostic; confusion matrix handles label errors
    gt = [_d(0, 0, 10, 10, label="jug")]
    pred = [_d(0, 0, 10, 10, label="crimp")]
    result = match_detections(gt, pred)
    assert len(result.matched) == 1


def test_many_to_one_blocked():
    # Second pred cannot steal already-matched gt box
    gt = [_d(0, 0, 10, 10)]
    pred = [_d(0, 0, 10, 10, score=0.9), _d(0, 0, 10, 10, score=0.8)]
    result = match_detections(gt, pred)
    assert len(result.matched) == 1
    assert len(result.unmatched_pred) == 1
