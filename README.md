# hold-grasp-study

Fine-grained contact point estimation: from climbing holds to robotic grasp affordances.

**Research question:** Can open-vocabulary vision-language models detect and classify climbing hold shapes, and does a detection→classification→placement pipeline transfer to robotic grasping?

---

## Taxonomy

| Hold type | Grasp analogue | Notes |
|-----------|----------------|-------|
| Jug | Power grip | Large incut, whole hand |
| Crimp | Pinch grip | Small edge, fingertips |
| Sloper | Lateral / open-hand | Smooth rounded surface |
| Pinch | Pinch grip (sided) | Squeeze between thumb and fingers |
| Pocket | Insertion grip | Finger(s) into hole |

See [`docs/taxonomy.md`](docs/taxonomy.md) for annotation guidelines.

---

## Phases

**Phase 1 — Hold Detection + Classification** *(this repo)*
Compare 4 models on a labeled climbing hold dataset. Target: arXiv preprint + HuggingFace dataset release.

**Phase 2 — Contact Point / Limb Placement** *(planned)*
Geometric reachability graph + pose-supervised ML planner. Target: CVPR/ICCV workshop.

**Phase 3 — Cross-Domain Transfer to Robotic Grasping** *(planned)*
Measure whether hold-taxonomy pretraining transfers to GraspNet / Jacquard. Target: ICRA/IROS.

---

## Models (Phase 1)

| ID | Model | Mode |
|----|-------|------|
| `la-zero` | LocateAnything-3B | Zero-shot: direct shape prompt |
| `la-two` | LocateAnything-3B | Two-stage: detect all → classify crop |
| `gdino` | Grounding DINO | Zero-shot |
| `yolo-ft` | YOLOv8/11 | Fine-tuned on train split |

---

## Results

*To be filled after experiments.*

| Model | jug F1 | crimp F1 | sloper F1 | pinch F1 | pocket F1 | macro F1 |
|-------|--------|----------|-----------|----------|-----------|----------|
| la-zero | — | — | — | — | — | — |
| la-two | — | — | — | — | — | — |
| gdino | — | — | — | — | — | — |
| yolo-ft | — | — | — | — | — | — |

---

## Setup

### Mac (eval only, no GPU required)

```bash
git clone https://github.com/KarthikPrakash7/hold-grasp-study.git
cd hold-grasp-study
pip install -e ".[dev]"
```

### PC / WSL2 (inference + fine-tuning, GPU required)

```bash
pip install -e ".[gpu]"
```

---

## Usage

### Run inference (PC, GPU)

```bash
holdgrasp detect --model gdino --image data/climbing/raw/wall01.jpg --out experiments/results/
holdgrasp detect --model la-zero --image data/climbing/raw/wall01.jpg --out experiments/results/
holdgrasp detect --model yolo-ft --checkpoint experiments/results/yolo-ft/weights/best.pt \
    --image data/climbing/raw/wall01.jpg --out experiments/results/
```

### Evaluate predictions (Mac, CPU)

```bash
holdgrasp evaluate \
    --predictions experiments/results/ \
    --annotations data/climbing/annotations/test.json
```

With error gallery:

```bash
holdgrasp evaluate \
    --predictions experiments/results/ \
    --annotations data/climbing/annotations/test.json \
    --gallery --image-dir data/climbing/raw/
```

### Compare all models

```bash
holdgrasp evaluate \
    --predictions experiments/results/ \
    --annotations data/climbing/annotations/test.json \
    --out experiments/results/metrics.json

holdgrasp compare --results-dir experiments/results/
```

---

## Fine-tuning YOLO

**Step 1 — Prepare data** (Mac):

```bash
python -m holdgrasp.scripts.prepare_yolo_data \
    --train-ann data/climbing/annotations/train.json \
    --val-ann   data/climbing/annotations/val.json \
    --image-dir data/climbing/raw/ \
    --out-dir   data/climbing/train_yolo/
```

**Step 2 — Train** (PC, GPU):

```bash
yolo train cfg=experiments/configs/yolo_finetune.yaml
```

Checkpoint saved to `experiments/results/yolo-ft/weights/best.pt`.

---

## Data

Annotations in Roboflow (free tier): bounding box + one of 5 taxonomy classes.
Split by **photo** (not by hold instance) to prevent leakage: ~70% train / 30% test.
Export formats: COCO JSON + YOLO.

Raw images and prediction JSON are gitignored. Annotated dataset will be released on HuggingFace after Phase 1.

---

## Hardware split

| Machine | Role |
|---------|------|
| Mac (CPU) | Annotation, eval harness, result analysis |
| WSL2 PC (8–12 GB GPU, CUDA) | Inference, YOLO fine-tune |

Cached prediction JSON bridges the two.

---

## License

Own code: MIT.

Third-party models:
- **LocateAnything-3B** — NVIDIA non-commercial license. Portfolio/research use only. Not for commercial deployment.
- **Grounding DINO** — Apache 2.0
- **YOLOv8/11** — AGPL-3.0
