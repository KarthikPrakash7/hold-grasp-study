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
    """Predictions must only match GT in the same image."""
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    # Image 1: no GT, has a jug prediction (should be FP)
    # Image 2: has GT jug, no predictions (should be FN)
    pred1 = {
        "model": "test-model",
        "image_id": "wall01",
        "image_path": str(tmp_path / "wall01.jpg"),
        "detections": [
            {"box": [0.0, 0.0, 10.0, 10.0], "label": "jug", "score": 0.9, "image_id": "wall01"}
        ],
    }
    (pred_dir / "wall01_test-model.json").write_text(json.dumps(pred1))
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
            # GT jug is in wall02, but prediction is in wall01
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
    # jug: 1 FP (wall01 pred with no GT) + 1 FN (wall02 GT with no pred) = precision=0, recall=0
    assert "0.000" in result.output  # precision or recall should be 0
