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
