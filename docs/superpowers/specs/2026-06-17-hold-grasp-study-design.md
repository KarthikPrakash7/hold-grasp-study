# Fine-Grained Contact Point Estimation: From Climbing Holds to Robotic Grasp Affordances

**Date:** 2026-06-17  
**Status:** Approved

---

## Research Question

Can open-vocabulary vision-language models (VLMs) detect and classify fine-grained contact affordances (climbing hold shapes), and does a detection→classification→placement pipeline transfer to robotic grasping?

Expected finding: zero-shot VLMs fail at fine-grained shape taxonomy; fine-tuning closes the gap; the taxonomy maps cleanly to grasp types, enabling cross-domain transfer.

---

## Taxonomy

| Hold type | Grasp analogue | Notes |
|-----------|----------------|-------|
| Jug | Power grip | Large incut, whole hand |
| Crimp | Pinch grip | Small edge, fingertips |
| Sloper | Lateral / open-hand | Smooth rounded surface |
| Pinch | Pinch grip (sided) | Squeeze between thumb and fingers |
| Pocket | Insertion grip | Finger(s) into hole |

A short class-definition document with example crops MUST be written before annotation begins. Inter-annotator consistency depends on it.

---

## Phases

### Phase 1 — Hold Detection + Classification Study
*Independently shippable. Target: arXiv preprint + HuggingFace dataset release.*

**Dataset**
- 15–20 bouldering wall photos (varied lighting, wall angle, hold color)
- ~500–800 hold instances total
- Annotated in Roboflow (free tier): bounding box + one of 5 taxonomy classes
- Split by **photo** (not by hold) to prevent leakage: ~70% train / 30% test
- Export: COCO JSON + YOLO format
- Public Roboflow/HF release after Phase 1 complete

**Models compared**

| ID | Model | Mode |
|----|-------|------|
| LA-zero | LocateAnything-3B | Zero-shot: direct shape prompt |
| LA-two | LocateAnything-3B | Two-stage: detect all holds → classify crop |
| GDINO | Grounding DINO | Zero-shot |
| YOLO-ft | YOLOv8/11 | Fine-tuned on train split |

All share a common `Detector` interface: `detect(image, classes) -> list[Detection(box, label, score)]`. Predictions saved as JSON so eval runs on Mac without GPU.

**Eval harness**
- COCO-style greedy matching at IoU ≥ 0.5
- Per-class precision / recall / F1
- Count error per class per image (predicted vs. true count — original use case)
- 5×5 confusion matrix on matched boxes
- Qualitative error gallery: worst-case images per class pair
- Eval harness unit-tested with synthetic boxes before any real inference

**Hardware split**
- Mac: annotation, eval harness dev, result analysis (CPU only)
- WSL2 PC (8–12GB GPU, CUDA): all inference, YOLO fine-tune. Cached predictions JSON bridges the two.

**Runtime stack (PC)**
```
transformers==4.57.1
torch (cu121)
ultralytics
pillow
opencv-python-headless
numpy
```

### Phase 2 — Contact Point / Limb Placement Estimation
*Extends Phase 1 dataset. Target: CVPR/ICCV Vision for Robotics workshop.*

**Geometric planner (baseline)**
- Input: detected hold bounding boxes + labels from Phase 1
- Build reachability graph: nodes = holds, edges = pairs within human reach (~70 cm), weighted by hold quality (jug > crimp > sloper)
- A* from bottom-zone holds to top-zone holds → limb sequence
- Fully explainable, no ML, serves as baseline

**Pose-supervised ML planner**
- Collect additional photos/frames of actual climbers on annotated walls (gym footage, YouTube)
- Run MediaPipe Pose or ViTPose on climber images → extract hand/foot keypoints
- Match keypoints to nearest detected hold → label: which hold is occupied by which limb
- Train binary classifier or assignment model: given hold layout + hold types → predict limb assignment
- Dataset artifact: wall image + hold annotations + climber pose = novel public dataset (does not currently exist with shape labels)

**Metrics**
- Geometric: route validity rate (no impossible reach), holds-per-move count
- ML: limb assignment accuracy vs. held-out climber images

### Phase 3 — Cross-Domain Transfer to Robotic Grasping
*Extends Phase 1 pipeline. Target: ICRA/IROS application paper.*

- Map taxonomy: jug→power, crimp→pinch, sloper→lateral, pocket→insertion
- Run Phase 1 pipeline on public grasp affordance dataset (GraspNet-1Billion has affordance labels; Jacquard dataset)
- Measure: does YOLO-ft pretrained on climbing holds outperform baseline on grasp affordance detection?
- Report cross-domain F1 delta per grasp type
- Main claim: "fine-grained contact affordance taxonomy transfers across domains"

---

## Code Layout

```
hold-grasp-study/
├── README.md                          # results table, paper link, setup
├── .gitignore
├── docs/
│   ├── superpowers/specs/             # this file
│   ├── taxonomy.md                    # class definitions + example crops
│   └── paper/                        # writeup sections
├── data/
│   ├── climbing/
│   │   ├── annotations/               # COCO JSON, YOLO labels
│   │   └── raw/                       # gitignored, large files
│   └── grasp/
│       ├── annotations/
│       └── raw/
├── src/holdgrasp/
│   ├── __init__.py
│   ├── models/
│   │   ├── base.py                    # Detector ABC + Detection dataclass
│   │   ├── locate_anything.py         # LA-zero + LA-two adapters
│   │   ├── grounding_dino.py
│   │   └── yolo.py
│   ├── eval/
│   │   ├── matching.py                # IoU greedy matcher
│   │   ├── metrics.py                 # P/R/F1, count error, confusion
│   │   └── viz.py                     # annotated images, error gallery
│   ├── planner/
│   │   ├── graph.py                   # reachability graph builder
│   │   ├── geometric.py               # A* route planner
│   │   └── ml_planner.py             # pose-supervised assignment model
│   └── transfer/
│       └── cross_domain.py            # grasp dataset adapter + transfer eval
├── experiments/
│   ├── configs/                       # YAML run configs
│   └── results/                       # cached prediction JSON (gitignored raw)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_transfer_analysis.ipynb
└── pyproject.toml
```

---

## CLI Interface

```bash
# Run inference, save predictions JSON
holdgrasp detect --model la-zero --image data/climbing/raw/wall01.jpg --out experiments/results/

# Evaluate saved predictions against COCO annotations
holdgrasp evaluate --predictions experiments/results/ --annotations data/climbing/annotations/test.json

# Compare all models, produce results table
holdgrasp compare --results-dir experiments/results/

# Run geometric planner on detected holds
holdgrasp plan --predictions experiments/results/wall01_la-zero.json --mode geometric
```

---

## Venues

| Milestone | Target |
|-----------|--------|
| Phase 1 complete | arXiv preprint + HuggingFace dataset |
| Phase 1+2 | CVPR/ICCV workshop (Vision for Robotics, EgoVis) |
| Phase 1+2+3 | ICRA/IROS application paper |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| LocateAnything zero-shot near-zero on shapes | Expected; still publishable as "VLMs fail fine-grained taxonomy" |
| 8–12GB VRAM tight for LocateAnything-3B BF16 (~7GB) | Reduce image resolution; fallback to INT8 quantization |
| LocateAnything non-commercial license | Non-commercial portfolio use OK; note in README; Phase 3 uses GDINO + YOLO (Apache/GPL) for commercial framing |
| Climber pose dataset collection slow | Phase 2 is independent; Phase 1 shippable without it |
| GraspNet-1Billion large (download heavy) | Use Jacquard dataset as lighter alternative |

---

## License Notes

- LocateAnything-3B: NVIDIA non-commercial only. Portfolio/research use OK. Note in README.
- Grounding DINO: Apache 2.0
- YOLOv8: AGPL-3.0 (non-commercial compatible; commercial needs Ultralytics license)
- Own code: MIT
