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


def test_evaluate_per_image_matching(tmp_path):
    """Cross-image matching must not occur: pred in wall01 must not match GT in wall02."""
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    # Same box coordinates in both images
    box = [0.0, 0.0, 10.0, 10.0]
    # wall01: has a jug prediction at box, but NO GT
    pred1 = {
        "model": "test-model",
        "image_id": "wall01",
        "image_path": str(tmp_path / "wall01.jpg"),
        "detections": [
            {"box": box, "label": "jug", "score": 0.9, "image_id": "wall01"}
        ],
    }
    (pred_dir / "wall01_test-model.json").write_text(json.dumps(pred1))
    # wall02: has a GT jug at the same box coordinates, but NO predictions
    pred2 = {
        "model": "test-model",
        "image_id": "wall02",
        "image_path": str(tmp_path / "wall02.jpg"),
        "detections": [],
    }
    (pred_dir / "wall02_test-model.json").write_text(json.dumps(pred2))
    coco = {
        "images": [
            {"id": 1, "file_name": "wall01.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "wall02.jpg", "width": 640, "height": 480},
        ],
        "categories": [{"id": 1, "name": "jug"}],
        "annotations": [
            # GT jug in wall02 at the SAME coordinates as the prediction in wall01
            {"id": 1, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0}
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
    # With correct per-image matching:
    #   wall01: 1 FP (pred with no GT in this image)
    #   wall02: 1 FN (GT with no pred in this image)
    #   jug: precision = 0/(0+1) = 0.0, recall = 0/(0+1) = 0.0
    # Under old broken code (cross-image): pred would match GT → TP → precision=1.0, recall=1.0
    # This assertion distinguishes fixed from broken:
    assert " 0.000 " in result.output or "0.000" in result.output
    # Also assert NOT 1.000 precision (which old broken code would produce)
    lines = result.output.split("\n")
    jug_line = next((l for l in lines if l.strip().startswith("jug")), None)
    assert jug_line is not None, f"No jug line in output: {result.output}"
    assert "1.000" not in jug_line, f"Cross-image match detected — jug line shows 1.000: {jug_line}"
