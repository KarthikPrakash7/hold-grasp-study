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
@click.option("--gallery", is_flag=True, default=False, help="Save error gallery images to predictions-dir/gallery/")
@click.option("--image-dir", default=None, type=click.Path(path_type=Path), help="Directory containing raw images (required for --gallery)")
def evaluate(predictions: Path, annotations: Path, out: Path | None, gallery: bool, image_dir: Path | None):
    """Evaluate cached predictions against COCO ground-truth annotations."""
    from holdgrasp.eval.matching import MatchResult
    from holdgrasp.eval.metrics import count_error_per_image

    gt_by_image = load_coco_gt(annotations)
    by_model = _load_prediction_dir(predictions)

    all_results: dict = {}
    for model_id, pred_by_image in sorted(by_model.items()):
        # Bug #1 & #2 fix: match per image so predictions never cross image boundaries
        all_matched = []
        all_unmatched_gt = []
        all_unmatched_pred = []

        for image_id in set(gt_by_image) | set(pred_by_image):
            gt = gt_by_image.get(image_id, [])
            pred = pred_by_image.get(image_id, [])
            r = match_detections(gt, pred)
            all_matched.extend(r.matched)
            all_unmatched_gt.extend(r.unmatched_gt)
            all_unmatched_pred.extend(r.unmatched_pred)

        match_result = MatchResult(
            matched=all_matched,
            unmatched_gt=all_unmatched_gt,
            unmatched_pred=all_unmatched_pred,
        )

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

        # Bug #3 fix: wire count_error_per_image
        errors = count_error_per_image(gt_by_image, pred_by_image, HOLD_CLASSES)
        click.echo("\nCount error (pred - gt) per class, summed across images:")
        click.echo(f"{'Class':<10} {'Error':>6}")
        click.echo("-" * 20)
        for cls in HOLD_CLASSES:
            total_err = sum(img_errs.get(cls, 0) for img_errs in errors.values())
            click.echo(f"{cls:<10} {total_err:>+6}")

        # Bug #4 fix: wire error_gallery when --gallery flag is set
        if gallery:
            if not image_dir:
                raise click.ClickException("--image-dir required when --gallery is used")
            from holdgrasp.eval.viz import error_gallery as _error_gallery
            image_paths = {
                p.stem: p
                for p in image_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            }
            _error_gallery(image_paths, gt_by_image, pred_by_image, predictions, model_id)
            click.echo(f"Gallery saved → {predictions}/gallery/{model_id}/")

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
