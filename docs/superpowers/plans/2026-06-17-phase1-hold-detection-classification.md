# Phase 1: Hold Detection + Classification Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end detection/classification pipeline benchmarking 4 models (LA-zero, LA-two, GDINO, YOLO-ft) on climbing hold taxonomy, with cached prediction JSON for Mac-side evaluation.

**Architecture:** Three-layer design — model adapters all implement a common `Detector` ABC; eval harness runs on cached prediction JSON (Mac/CPU); CLI glues detect → evaluate → compare. Inference runs on PC (GPU), eval on Mac (CPU).

**Tech Stack:** Python 3.10+, Click, Pillow, NumPy, PyYAML; GPU extras: PyTorch, Transformers 4.57.1, Ultralytics

## Global Constraints

- Python >= 3.10
- `transformers==4.57.1` (exact — LocateAnything compatibility)
- GPU deps install-gated: `pip install -e ".[gpu]"` — core Mac eval must work without torch
- Hold classes (exact, in order): `["jug", "crimp", "sloper", "pinch", "pocket"]`
- Bounding boxes everywhere in code: `(x1, y1, x2, y2)` pixel coords
- COCO JSON bbox format `[x, y, w, h]` → convert to `(x, y, x+w, y+h)` on load
- Prediction JSON filename convention: `{image_stem}_{model_id}.json`
- IoU match threshold: 0.5
- LocateAnything-3B: NVIDIA non-commercial license — note in README, not in code

---

## File Map

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Package config, optional deps, CLI entry point |
| `docs/taxonomy.md` | Class definitions + annotation guidance |
| `src/holdgrasp/__init__.py` | Package root |
| `src/holdgrasp/models/base.py` | `Detection` dataclass, `Detector` ABC, `HOLD_CLASSES`, `iou()` |
| `src/holdgrasp/models/yolo.py` | YOLOv8/11 adapter |
| `src/holdgrasp/models/grounding_dino.py` | Grounding DINO zero-shot adapter |
| `src/holdgrasp/models/locate_anything.py` | LA-zero + LA-two adapters |
| `src/holdgrasp/eval/matching.py` | Greedy IoU matcher, `MatchResult` |
| `src/holdgrasp/eval/metrics.py` | P/R/F1, count error, confusion matrix, COCO loader |
| `src/holdgrasp/eval/viz.py` | Box drawing, error gallery |
| `src/holdgrasp/cli.py` | CLI: detect, evaluate, compare |
| `src/holdgrasp/scripts/prepare_yolo_data.py` | COCO → YOLO format conversion |
| `data/climbing/train_yolo/data.yaml` | YOLO training data config |
| `experiments/configs/yolo_finetune.yaml` | YOLO training hyperparameters |
| `experiments/configs/gdino.yaml` | Grounding DINO inference config |
| `experiments/configs/la_zero.yaml` | LA-zero inference config |
| `experiments/configs/la_two.yaml` | LA-two inference config |
| `tests/test_base.py` | Detection + iou tests |
| `tests/test_matching.py` | IoU matcher tests |
| `tests/test_metrics.py` | Metrics + COCO loader tests |
| `tests/test_cli.py` | CLI integration tests |

---

### Task 1: Project Scaffold + Taxonomy Doc

**Files:**
- Create: `pyproject.toml`
- Create: `src/holdgrasp/__init__.py`
- Create: `src/holdgrasp/models/__init__.py`
- Create: `src/holdgrasp/eval/__init__.py`
- Create: `src/holdgrasp/planner/__init__.py`
- Create: `src/holdgrasp/transfer/__init__.py`
- Create: `src/holdgrasp/scripts/__init__.py`
- Create: `tests/__init__.py`
- Modify: `.gitignore`
- Create: `docs/taxonomy.md`
- Create: directory placeholders under `data/`, `experiments/`, `notebooks/`

**Interfaces:**
- Produces: installable `holdgrasp` package; `holdgrasp --help` works

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "holdgrasp"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "click>=8.0",
    "pillow>=10.0",
    "numpy>=1.24",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
gpu = [
    "torch>=2.0",
    "transformers==4.57.1",
    "ultralytics>=8.0",
    "opencv-python-headless>=4.8",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[project.scripts]
holdgrasp = "holdgrasp.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/holdgrasp"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty init files**

Create these files, all empty:
- `src/holdgrasp/__init__.py`
- `src/holdgrasp/models/__init__.py`
- `src/holdgrasp/eval/__init__.py`
- `src/holdgrasp/planner/__init__.py`
- `src/holdgrasp/transfer/__init__.py`
- `src/holdgrasp/scripts/__init__.py`
- `tests/__init__.py`

- [ ] **Step 3: Update .gitignore**

Add to existing `.gitignore`:
```
data/climbing/raw/
data/grasp/raw/
experiments/results/
*.pt
*.pth
__pycache__/
*.egg-info/
.pytest_cache/
dist/
.venv/
```

- [ ] **Step 4: Write docs/taxonomy.md**

```markdown
# Climbing Hold Taxonomy

Read before annotating in Roboflow. All annotators must use these definitions.

## Classes

### jug
Large incut hold. The whole hand fits inside the cavity. Easy to grip. Often at route start/top.
**Annotation:** Box tightly around the entire hold body including the incut lip.

### crimp
Small horizontal edge, typically 5–20 mm depth. Only fingertips contact it.
**Annotation:** Box around the edge surface only, not surrounding plastic.

### sloper
Smooth, rounded hold with no positive edge. Requires open-hand friction. Surface is convex outward.
**Annotation:** Box around the rounded surface.

### pinch
Hold designed to be squeezed between thumb (one side) and fingers (opposite side). Typically a knob or rib.
**Annotation:** Box around the full pinch body including both contact sides.

### pocket
Hold with one or more finger-depth holes. Fingers insert into the pocket.
**Annotation:** Box around the entire pocket body including the hole opening.

## Ambiguous Cases

- Hold could be jug or sloper: choose by grip type most climbers would use.
- Dual-texture holds: label by the dominant grip type.
- Tiny jugs that look like crimps: label by incut size (thumb-width incut → jug).
- Footholds too small to classify: skip.
```

- [ ] **Step 5: Create directory structure and install**

```bash
mkdir -p data/climbing/annotations data/climbing/raw
mkdir -p data/grasp/annotations data/grasp/raw
mkdir -p experiments/configs experiments/results
mkdir -p notebooks
touch data/climbing/annotations/.gitkeep
touch experiments/configs/.gitkeep experiments/results/.gitkeep
touch notebooks/.gitkeep
pip install -e ".[dev]"
```

- [ ] **Step 6: Verify install**

```bash
python -c "import holdgrasp; print('OK')"
```
Expected output: `OK`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/ docs/ data/ experiments/ notebooks/ .gitignore
git commit -m "chore: scaffold package layout, taxonomy doc, and directory structure"
```

---

### Task 2: Detection Dataclass + Detector ABC

**Files:**
- Create: `src/holdgrasp/models/base.py`
- Create: `tests/test_base.py`

**Interfaces:**
- Produces:
  - `HOLD_CLASSES: list[str] = ["jug", "crimp", "sloper", "pinch", "pocket"]`
  - `Detection(box: tuple[float,float,float,float], label: str, score: float, image_id: str = "") -> Detection`
  - `Detection.to_dict() -> dict`
  - `Detection.from_dict(d: dict) -> Detection` (classmethod)
  - `iou(a: tuple[float,float,float,float], b: tuple[float,float,float,float]) -> float`
  - `Detector` ABC with `detect(image_path: Path, classes: list[str]) -> list[Detection]`

- [ ] **Step 1: Write failing tests**

`tests/test_base.py`:
```python
import pytest
from holdgrasp.models.base import Detection, HOLD_CLASSES, iou


def test_hold_classes_exact():
    assert HOLD_CLASSES == ["jug", "crimp", "sloper", "pinch", "pocket"]


def test_detection_creation():
    d = Detection(box=(10.0, 20.0, 50.0, 80.0), label="jug", score=0.9)
    assert d.box == (10.0, 20.0, 50.0, 80.0)
    assert d.label == "jug"
    assert d.score == 0.9
    assert d.image_id == ""


def test_detection_roundtrip():
    d = Detection(box=(10.0, 20.0, 50.0, 80.0), label="crimp", score=0.75, image_id="wall01")
    assert Detection.from_dict(d.to_dict()) == d


def test_iou_identical_boxes():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (20.0, 20.0, 30.0, 30.0)
    assert iou(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    a = (0.0, 0.0, 10.0, 10.0)   # area = 100
    b = (5.0, 0.0, 15.0, 10.0)   # area = 100, overlap = 5×10 = 50
    # union = 200 - 50 = 150
    assert iou(a, b) == pytest.approx(50.0 / 150.0)
```

- [ ] **Step 2: Run — verify FAIL**

```bash
pytest tests/test_base.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'holdgrasp.models.base'`

- [ ] **Step 3: Write base.py**

`src/holdgrasp/models/base.py`:
```python
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
```

- [ ] **Step 4: Run — verify PASS**

```bash
pytest tests/test_base.py -v
```
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/holdgrasp/models/base.py tests/test_base.py
git commit -m "feat: Detection dataclass, Detector ABC, and iou function"
```

---

### Task 3: IoU Greedy Matcher

**Files:**
- Create: `src/holdgrasp/eval/matching.py`
- Create: `tests/test_matching.py`

**Interfaces:**
- Consumes: `Detection` from `holdgrasp.models.base`, `iou` from `holdgrasp.models.base`
- Produces:
  - `MatchResult` dataclass with fields:
    - `matched: list[tuple[Detection, Detection]]` — (gt, pred) pairs
    - `unmatched_gt: list[Detection]`
    - `unmatched_pred: list[Detection]`
  - `match_detections(gt: list[Detection], pred: list[Detection], iou_threshold: float = 0.5) -> MatchResult`

- [ ] **Step 1: Write failing tests**

`tests/test_matching.py`:
```python
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
```

- [ ] **Step 2: Run — verify FAIL**

```bash
pytest tests/test_matching.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'holdgrasp.eval.matching'`

- [ ] **Step 3: Write matching.py**

`src/holdgrasp/eval/matching.py`:
```python
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
```

- [ ] **Step 4: Run — verify PASS**

```bash
pytest tests/test_matching.py -v
```
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/holdgrasp/eval/matching.py tests/test_matching.py
git commit -m "feat: greedy class-agnostic IoU matcher"
```

---

### Task 4: Metrics + COCO Loader

**Files:**
- Create: `src/holdgrasp/eval/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Detection`, `HOLD_CLASSES` from `holdgrasp.models.base`; `MatchResult` from `holdgrasp.eval.matching`
- Produces:
  - `load_coco_gt(path: Path) -> dict[str, list[Detection]]` — keyed by image filename stem
  - `ClassMetrics(precision: float, recall: float, f1: float, count_error: float = 0.0)`
  - `compute_metrics(result: MatchResult, classes: list[str]) -> dict[str, ClassMetrics]`
  - `confusion_matrix(result: MatchResult, classes: list[str]) -> np.ndarray` — shape `(len(classes), len(classes))`, rows=GT class, cols=pred class
  - `count_error_per_image(gt_by_image: dict[str, list[Detection]], pred_by_image: dict[str, list[Detection]], classes: list[str]) -> dict[str, dict[str, int]]` — image_id → class → (predicted_count - gt_count)

- [ ] **Step 1: Write failing tests**

`tests/test_metrics.py`:
```python
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
```

- [ ] **Step 2: Run — verify FAIL**

```bash
pytest tests/test_metrics.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'holdgrasp.eval.metrics'`

- [ ] **Step 3: Write metrics.py**

`src/holdgrasp/eval/metrics.py`:
```python
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
```

- [ ] **Step 4: Run — verify PASS**

```bash
pytest tests/test_metrics.py -v
```
Expected: 9 tests PASS

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```
Expected: 15 tests PASS total

- [ ] **Step 6: Commit**

```bash
git add src/holdgrasp/eval/metrics.py tests/test_metrics.py
git commit -m "feat: COCO loader, P/R/F1, confusion matrix, count error per image"
```

---

### Task 5: Visualization Module

**Files:**
- Create: `src/holdgrasp/eval/viz.py`

**Interfaces:**
- Consumes: `Detection`, `HOLD_CLASSES` from `holdgrasp.models.base`; `count_error_per_image` from `holdgrasp.eval.metrics`
- Produces:
  - `draw_detections(image_path: Path, detections: list[Detection], out_path: Path, gt_detections: list[Detection] | None = None) -> None`
  - `error_gallery(image_paths: dict[str, Path], gt_by_image: dict, pred_by_image: dict, result_dir: Path, model_id: str, top_n: int = 5) -> None`

No unit tests for this module — PIL rendering output cannot be meaningfully asserted. Verify visually by running on a real image.

- [ ] **Step 1: Write viz.py**

`src/holdgrasp/eval/viz.py`:
```python
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
```

- [ ] **Step 2: Verify import**

```bash
python -c "from holdgrasp.eval.viz import draw_detections, error_gallery; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/holdgrasp/eval/viz.py
git commit -m "feat: visualization module for detection overlays and error gallery"
```

---

### Task 6: YOLO Adapter + Fine-Tuning Config

**Files:**
- Create: `src/holdgrasp/models/yolo.py`
- Create: `data/climbing/train_yolo/data.yaml`
- Create: `experiments/configs/yolo_finetune.yaml`

**Interfaces:**
- Consumes: `Detector` ABC from `holdgrasp.models.base`
- Produces: `YOLODetector(checkpoint: str | Path, conf_threshold: float = 0.25) -> Detector`

GPU deps (`ultralytics`) imported inside `__init__` only — `import holdgrasp` works on Mac.

- [ ] **Step 1: Write yolo.py**

`src/holdgrasp/models/yolo.py`:
```python
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
```

- [ ] **Step 2: Write data.yaml**

`data/climbing/train_yolo/data.yaml`:
```yaml
# Paths relative to this file. Roboflow YOLO export: copy images and labels here.
path: .
train: train/images
val: val/images
nc: 5
names:
  0: jug
  1: crimp
  2: sloper
  3: pinch
  4: pocket
```

- [ ] **Step 3: Write yolo_finetune.yaml**

`experiments/configs/yolo_finetune.yaml`:
```yaml
# Run on PC (GPU):  yolo train cfg=experiments/configs/yolo_finetune.yaml
model: yolo11n.pt
data: data/climbing/train_yolo/data.yaml
epochs: 100
imgsz: 640
batch: 16
project: experiments/results
name: yolo-ft
save: true
device: 0
workers: 4
patience: 20
```

- [ ] **Step 4: Verify CPU import**

```bash
python -c "from holdgrasp.models.yolo import YOLODetector; print('OK')"
```
Expected: `OK` (no ultralytics import at module level)

- [ ] **Step 5: Commit**

```bash
git add src/holdgrasp/models/yolo.py data/climbing/train_yolo/data.yaml experiments/configs/yolo_finetune.yaml
git commit -m "feat: YOLO adapter and fine-tuning config"
```

---

### Task 7: Grounding DINO Adapter

**Files:**
- Create: `src/holdgrasp/models/grounding_dino.py`
- Create: `experiments/configs/gdino.yaml`

**Interfaces:**
- Consumes: `Detector` ABC
- Produces: `GroundingDINODetector(box_threshold: float = 0.35, text_threshold: float = 0.25) -> Detector`

- [ ] **Step 1: Write grounding_dino.py**

`src/holdgrasp/models/grounding_dino.py`:
```python
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
```

- [ ] **Step 2: Write gdino.yaml**

`experiments/configs/gdino.yaml`:
```yaml
model: gdino
model_id: IDEA-Research/grounding-dino-tiny
box_threshold: 0.35
text_threshold: 0.25
classes:
  - jug
  - crimp
  - sloper
  - pinch
  - pocket
```

- [ ] **Step 3: Verify CPU import**

```bash
python -c "from holdgrasp.models.grounding_dino import GroundingDINODetector; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/holdgrasp/models/grounding_dino.py experiments/configs/gdino.yaml
git commit -m "feat: Grounding DINO zero-shot adapter"
```

---

### Task 8: LocateAnything Adapters

**Files:**
- Create: `src/holdgrasp/models/locate_anything.py`
- Create: `experiments/configs/la_zero.yaml`
- Create: `experiments/configs/la_two.yaml`

**Interfaces:**
- Consumes: `Detector` ABC
- Produces:
  - `LocateAnythingZeroShot(device: str = "cuda") -> Detector` — one prompt per class
  - `LocateAnythingTwoStage(device: str = "cuda") -> Detector` — detect-all then classify crop

**Before running inference on PC:** Verify the exact HuggingFace model ID and generation API from the model card. The adapter below follows the standard causal-LM + AutoProcessor pattern. If LocateAnything-3B uses a custom API, update `_generate()` accordingly. NVIDIA non-commercial license — portfolio/research use OK; note this in README.

- [ ] **Step 1: Write locate_anything.py**

`src/holdgrasp/models/locate_anything.py`:
```python
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
```

- [ ] **Step 2: Write configs**

`experiments/configs/la_zero.yaml`:
```yaml
model: la-zero
model_id: nvidia/locate-anything-3b
device: cuda
classes:
  - jug
  - crimp
  - sloper
  - pinch
  - pocket
```

`experiments/configs/la_two.yaml`:
```yaml
model: la-two
model_id: nvidia/locate-anything-3b
device: cuda
classes:
  - jug
  - crimp
  - sloper
  - pinch
  - pocket
```

- [ ] **Step 3: Verify CPU import**

```bash
python -c "from holdgrasp.models.locate_anything import LocateAnythingZeroShot, LocateAnythingTwoStage; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/holdgrasp/models/locate_anything.py experiments/configs/la_zero.yaml experiments/configs/la_two.yaml
git commit -m "feat: LocateAnything zero-shot and two-stage detection adapters"
```

---

### Task 9: CLI — detect, evaluate, compare

**Files:**
- Create: `src/holdgrasp/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: all `Detector` implementations; `load_coco_gt`, `match_detections`, `compute_metrics`, `confusion_matrix`, `count_error_per_image`; `Detection.to_dict` / `from_dict`
- Produces: `holdgrasp detect`, `holdgrasp evaluate`, `holdgrasp compare` commands

Prediction JSON schema saved per image per model at `{out_dir}/{image_stem}_{model_id}.json`:
```json
{
  "model": "la-zero",
  "image_id": "wall01",
  "image_path": "data/climbing/raw/wall01.jpg",
  "detections": [{"box": [10.0, 20.0, 50.0, 80.0], "label": "jug", "score": 0.9, "image_id": "wall01"}]
}
```

- [ ] **Step 1: Write failing tests**

`tests/test_cli.py`:
```python
import json
from pathlib import Path
from click.testing import CliRunner
from holdgrasp.cli import main


def test_detect_help():
    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_evaluate_help():
    runner = CliRunner()
    result = runner.invoke(main, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--predictions" in result.output
    assert "--annotations" in result.output


def test_compare_help():
    runner = CliRunner()
    result = runner.invoke(main, ["compare", "--help"])
    assert result.exit_code == 0
    assert "--results-dir" in result.output


def test_evaluate_synthetic(tmp_path):
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    pred_payload = {
        "model": "test-model",
        "image_id": "wall01",
        "image_path": str(tmp_path / "wall01.jpg"),
        "detections": [
            {"box": [0.0, 0.0, 10.0, 10.0], "label": "jug", "score": 0.9, "image_id": "wall01"}
        ],
    }
    (pred_dir / "wall01_test-model.json").write_text(json.dumps(pred_payload))
    coco = {
        "images": [{"id": 1, "file_name": "wall01.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "jug"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0}
        ],
    }
    ann_path = tmp_path / "test.json"
    ann_path.write_text(json.dumps(coco))
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["evaluate", "--predictions", str(pred_dir), "--annotations", str(ann_path)],
    )
    assert result.exit_code == 0, result.output
    assert "jug" in result.output
    assert "test-model" in result.output
```

- [ ] **Step 2: Run — verify FAIL**

```bash
pytest tests/test_cli.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'holdgrasp.cli'`

- [ ] **Step 3: Write cli.py**

`src/holdgrasp/cli.py`:
```python
from __future__ import annotations
import importlib
import json
from pathlib import Path
import click
from holdgrasp.models.base import HOLD_CLASSES, Detection
from holdgrasp.eval.matching import match_detections
from holdgrasp.eval.metrics import load_coco_gt, compute_metrics, confusion_matrix

_MODEL_REGISTRY: dict[str, str] = {
    "la-zero": "holdgrasp.models.locate_anything:LocateAnythingZeroShot",
    "la-two": "holdgrasp.models.locate_anything:LocateAnythingTwoStage",
    "gdino": "holdgrasp.models.grounding_dino:GroundingDINODetector",
    "yolo-ft": "holdgrasp.models.yolo:YOLODetector",
}


def _load_detector(model_id: str, checkpoint: str | None):
    module_path, cls_name = _MODEL_REGISTRY[model_id].split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    if model_id == "yolo-ft":
        if not checkpoint:
            raise click.ClickException("--checkpoint required for yolo-ft")
        return cls(checkpoint)
    return cls()


def _load_prediction_dir(pred_dir: Path) -> dict[str, dict[str, list[Detection]]]:
    """Returns {model_id: {image_id: list[Detection]}}."""
    by_model: dict[str, dict[str, list[Detection]]] = {}
    for f in sorted(pred_dir.glob("*.json")):
        data = json.loads(f.read_text())
        model_id = data["model"]
        image_id = data["image_id"]
        dets = [Detection.from_dict(d) for d in data["detections"]]
        by_model.setdefault(model_id, {})[image_id] = dets
    return by_model


@click.group()
def main():
    """holdgrasp: climbing hold detection + classification toolkit."""


@main.command()
@click.option("--model", required=True, type=click.Choice(list(_MODEL_REGISTRY)))
@click.option("--image", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option("--checkpoint", default=None, help="Path to YOLO .pt checkpoint")
def detect(model: str, image: Path, out: Path, checkpoint: str | None):
    """Run inference on IMAGE and save prediction JSON to OUT/."""
    detector = _load_detector(model, checkpoint)
    detections = detector.detect(image, HOLD_CLASSES)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "image_id": image.stem,
        "image_path": str(image),
        "detections": [d.to_dict() for d in detections],
    }
    out_file = out / f"{image.stem}_{model}.json"
    out_file.write_text(json.dumps(payload, indent=2))
    click.echo(f"Saved {len(detections)} detections → {out_file}")


@main.command()
@click.option("--predictions", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--annotations", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", default=None, type=click.Path(path_type=Path),
              help="Append metrics to this JSON file (creates if missing)")
def evaluate(predictions: Path, annotations: Path, out: Path | None):
    """Evaluate cached predictions against COCO ground-truth annotations."""
    gt_by_image = load_coco_gt(annotations)
    by_model = _load_prediction_dir(predictions)

    all_results: dict = {}
    for model_id, pred_by_image in sorted(by_model.items()):
        all_gt: list[Detection] = []
        all_pred: list[Detection] = []
        for image_id in gt_by_image:
            all_gt.extend(gt_by_image[image_id])
            all_pred.extend(pred_by_image.get(image_id, []))

        match_result = match_detections(all_gt, all_pred)
        metrics = compute_metrics(match_result, HOLD_CLASSES)
        cm = confusion_matrix(match_result, HOLD_CLASSES)
        macro_f1 = sum(metrics[c].f1 for c in HOLD_CLASSES) / len(HOLD_CLASSES)

        click.echo(f"\n=== {model_id} ===")
        click.echo(f"{'Class':<10} {'P':>6} {'R':>6} {'F1':>6}")
        click.echo("-" * 32)
        for cls in HOLD_CLASSES:
            m = metrics[cls]
            click.echo(f"{cls:<10} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f}")
        click.echo(f"{'macro-F1':<10} {'':>6} {'':>6} {macro_f1:>6.3f}")
        click.echo("\nConfusion matrix (rows=GT, cols=pred):")
        click.echo("       " + "  ".join(f"{c[:5]:>5}" for c in HOLD_CLASSES))
        for i, cls in enumerate(HOLD_CLASSES):
            row = "  ".join(f"{cm[i, j]:>5}" for j in range(len(HOLD_CLASSES)))
            click.echo(f"{cls[:5]:>5}  {row}")

        all_results[model_id] = {
            "metrics": {
                c: {"precision": metrics[c].precision, "recall": metrics[c].recall, "f1": metrics[c].f1}
                for c in HOLD_CLASSES
            },
            "macro_f1": macro_f1,
            "confusion_matrix": cm.tolist(),
        }

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(out.read_text()) if out.exists() else {}
        existing.update(all_results)
        out.write_text(json.dumps(existing, indent=2))
        click.echo(f"\nMetrics saved → {out}")


@main.command()
@click.option("--results-dir", required=True, type=click.Path(exists=True, path_type=Path))
def compare(results_dir: Path):
    """Print comparison table for all models in results-dir/metrics.json."""
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        raise click.ClickException(
            f"No metrics.json in {results_dir}. Run: holdgrasp evaluate --out {results_dir}/metrics.json"
        )
    data = json.loads(metrics_file.read_text())
    header = f"{'Model':<14} " + " ".join(f"{c[:5]:>7}" for c in HOLD_CLASSES) + f" {'macroF1':>8}"
    click.echo(f"\n{header}")
    click.echo("-" * len(header))
    for model_id, result in sorted(data.items()):
        f1s = " ".join(f"{result['metrics'][c]['f1']:>7.3f}" for c in HOLD_CLASSES)
        click.echo(f"{model_id:<14} {f1s} {result['macro_f1']:>8.3f}")
```

- [ ] **Step 4: Run tests — verify PASS**

```bash
pytest tests/test_cli.py -v
```
Expected: 4 tests PASS

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```
Expected: all 19 tests PASS

- [ ] **Step 6: Smoke test**

```bash
holdgrasp --help
holdgrasp detect --help
holdgrasp evaluate --help
holdgrasp compare --help
```
Expected: help text for all four commands with correct options.

- [ ] **Step 7: Commit**

```bash
git add src/holdgrasp/cli.py tests/test_cli.py
git commit -m "feat: CLI with detect, evaluate, and compare commands"
```

---

### Task 10: YOLO Fine-Tuning Data Prep

**Files:**
- Create: `src/holdgrasp/scripts/prepare_yolo_data.py`

**Interfaces:**
- Consumes: COCO JSON from Roboflow export at `data/climbing/annotations/{train,val}.json`; raw images at `data/climbing/raw/`
- Produces: YOLO-format files at `data/climbing/train_yolo/{train,val}/{images,labels}/`

Runs on Mac (CPU). After running this, copy `data/climbing/train_yolo/` to PC and run:
```bash
yolo train cfg=experiments/configs/yolo_finetune.yaml
```

- [ ] **Step 1: Write prepare_yolo_data.py**

`src/holdgrasp/scripts/prepare_yolo_data.py`:
```python
"""
Convert Roboflow COCO JSON export to YOLO label format.

Usage:
    python -m holdgrasp.scripts.prepare_yolo_data \
        --train-ann data/climbing/annotations/train.json \
        --val-ann   data/climbing/annotations/val.json \
        --image-dir data/climbing/raw \
        --out-dir   data/climbing/train_yolo
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import click
from holdgrasp.models.base import HOLD_CLASSES

_CLASS_TO_IDX = {c: i for i, c in enumerate(HOLD_CLASSES)}


@click.command()
@click.option("--train-ann", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--val-ann", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--image-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(path_type=Path))
def main(train_ann: Path, val_ann: Path, image_dir: Path, out_dir: Path):
    for split, ann_path in [("train", train_ann), ("val", val_ann)]:
        _convert_split(split, ann_path, image_dir, out_dir)
    click.echo("\nDone. Run training on PC (GPU):")
    click.echo("  yolo train cfg=experiments/configs/yolo_finetune.yaml")


def _convert_split(split: str, ann_path: Path, image_dir: Path, out_dir: Path) -> None:
    data = json.loads(ann_path.read_text())
    id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    id_to_size = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    img_out = out_dir / split / "images"
    lbl_out = out_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    labels_by_image: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        cat_name = id_to_name[ann["category_id"]]
        if cat_name not in _CLASS_TO_IDX:
            continue
        w, h = id_to_size[img_id]
        x, y, bw, bh = ann["bbox"]
        # YOLO normalized format: class cx cy bw bh
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        line = f"{_CLASS_TO_IDX[cat_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        labels_by_image.setdefault(img_id, []).append(line)

    for img_id, file_name in id_to_file.items():
        src = image_dir / file_name
        if src.exists():
            shutil.copy2(src, img_out / file_name)
        lbl_file = lbl_out / Path(file_name).with_suffix(".txt").name
        lbl_file.write_text("\n".join(labels_by_image.get(img_id, [])))

    click.echo(f"{split}: {len(id_to_file)} images → {img_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import and help**

```bash
python -m holdgrasp.scripts.prepare_yolo_data --help
```
Expected: help text listing `--train-ann`, `--val-ann`, `--image-dir`, `--out-dir`

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```
Expected: all 19 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/holdgrasp/scripts/prepare_yolo_data.py
git commit -m "feat: COCO-to-YOLO data prep script for fine-tuning"
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| 4 models: LA-zero, LA-two, GDINO, YOLO-ft | Tasks 6, 7, 8 |
| Common `Detector` interface | Task 2 |
| Predictions saved as JSON, eval on Mac/CPU | Task 9 |
| COCO-style greedy matching at IoU ≥ 0.5 | Task 3 |
| Per-class precision / recall / F1 | Task 4 |
| Count error per class per image | Task 4 |
| 5×5 confusion matrix on matched boxes | Task 4 |
| Error gallery: worst-case images per class pair | Task 5 |
| Eval harness unit-tested before any inference | Tasks 3, 4 precede Tasks 6–8 |
| YOLO fine-tuned on train split | Task 6 (adapter), Task 10 (data prep) |
| Photo-level train/val split (no leakage) | Task 10 (respects Roboflow export split) |
| COCO JSON + YOLO format export | Task 10 produces YOLO; Task 4 loads COCO |
| CLI: detect, evaluate, compare | Task 9 |
| Taxonomy doc before annotation | Task 1 |
| Hold classes: jug, crimp, sloper, pinch, pocket | Task 2 (`HOLD_CLASSES` constant) |
| LocateAnything non-commercial noted | Task 8 (docstring) |
| Mac/PC hardware split | GPU deps optional; noted in Task 6 |
