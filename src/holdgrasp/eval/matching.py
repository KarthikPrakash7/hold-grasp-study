from __future__ import annotations
from dataclasses import dataclass, field
from holdgrasp.models.base import Detection, iou


@dataclass
class MatchResult:
    matched: list[tuple[Detection, Detection]] = field(default_factory=list)
    unmatched_gt: list[Detection] = field(default_factory=list)
    unmatched_pred: list[Detection] = field(default_factory=list)


def match_detections(
    gt: list[Detection],
    pred: list[Detection],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy one-to-one match. Predictions processed in descending score order."""
    sorted_pred = sorted(pred, key=lambda d: d.score, reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    pairs: list[tuple[Detection, Detection]] = []

    for pi, p in enumerate(sorted_pred):
        best_iou = iou_threshold - 1e-9
        best_gi = -1
        for gi, g in enumerate(gt):
            if gi in matched_gt:
                continue
            score = iou(g.box, p.box)
            if score > best_iou:
                best_iou = score
                best_gi = gi
        if best_gi >= 0:
            pairs.append((gt[best_gi], p))
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    unmatched_gt = [g for i, g in enumerate(gt) if i not in matched_gt]
    unmatched_pred = [p for i, p in enumerate(sorted_pred) if i not in matched_pred]
    return MatchResult(matched=pairs, unmatched_gt=unmatched_gt, unmatched_pred=unmatched_pred)
