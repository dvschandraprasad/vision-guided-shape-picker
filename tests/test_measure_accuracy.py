"""The accuracy report must count, fold and refuse to invent a pass."""

from __future__ import annotations

import cv2
import numpy as np

import measure_accuracy
import pipeline


def make_detection(shape: str, centre=(100.0, 100.0), angle_deg: float = 0.0,
                   confidence: float = 0.9) -> pipeline.Detection:
    class_id = pipeline.CLASSES.index(shape)
    return pipeline.Detection(
        class_id=class_id,
        confidence=confidence,
        center=centre,
        angle_deg=angle_deg,
        corners=((80.0, 80.0), (120.0, 80.0), (120.0, 120.0), (80.0, 120.0)),
    )


# Fake detector that returns the test's fixed answers.
class StubDetector:
    def __init__(self, detections, **_kwargs) -> None:
        self.detections = list(detections)

    def detect(self, _image):
        return self.detections


def write_case(tmp_path, class_id: int, corners_px: list[tuple[float, float]],
               size: int = 200) -> tuple[str, str]:
    """One image plus one 9-field label; returns the two folder paths."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    # Blank test image; the detector supplies fixed answers.
    cv2.imwrite(str(images / "frame_0001.png"), np.zeros((size, size, 3), np.uint8))
    fields = [str(class_id)]
    for x, y in corners_px:
        # Labels store each corner as a fraction of the image.
        fields += [f"{x / size:.6f}", f"{y / size:.6f}"]
    (labels / "frame_0001.txt").write_text(" ".join(fields) + "\n", encoding="utf-8")
    return str(images), str(labels)


# Run the report with a fake detector and no calibration.
def run(monkeypatch, images: str, labels: str, detections, extra: list[str] | None = None) -> int:
    monkeypatch.setattr(measure_accuracy.pipeline, "ShapeDetector",
                        lambda **_kwargs: StubDetector(detections))
    monkeypatch.setattr(measure_accuracy.calibrate, "load_calibration",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("none")))
    return measure_accuracy.main(["--images", images, "--labels", labels,
                                  *(extra or [])])


# Ground truth corners for a 60 px square.
SQUARE = [(70.0, 70.0), (130.0, 70.0), (130.0, 130.0), (70.0, 130.0)]


def test_perfect_match_passes_the_pocket_check(tmp_path, monkeypatch, capsys):
    images, labels = write_case(tmp_path, 1, SQUARE)

    code = run(monkeypatch, images, labels, [make_detection("square", angle_deg=0.0)])

    printed = capsys.readouterr().out
    assert code == 0
    assert "scored 1 frame(s), 1 matched object(s)" in printed
    assert "POCKET OK" in printed
    assert "millimetres are left out" in printed  # no calibration on this machine


def test_a_shift_and_a_turn_are_both_measured(tmp_path, monkeypatch, capsys):
    images, labels = write_case(tmp_path, 1, SQUARE)

    code = run(monkeypatch, images, labels,
               [make_detection("square", centre=(110.0, 100.0), angle_deg=10.0)])

    printed = capsys.readouterr().out
    assert code == 0
    assert "10.0px" in printed          # centre error straight off the label
    assert "10.0d" in printed           # folded angle error, square period 90


def test_a_missing_shape_is_counted_and_reported(tmp_path, monkeypatch, capsys):
    images, labels = write_case(tmp_path, 1, SQUARE)

    code = run(monkeypatch, images, labels, [])

    printed = capsys.readouterr().out
    assert code == 1
    assert "nothing matched" in printed


def test_a_circle_carries_no_heading_however_it_is_turned(tmp_path, monkeypatch, capsys):
    images, labels = write_case(tmp_path, 0, SQUARE)

    run(monkeypatch, images, labels, [make_detection("circle", angle_deg=129.0)])

    printed = capsys.readouterr().out
    assert "0.0d" in printed


def test_folded_error_wraps_inside_the_symmetry_half_range():
    # 45 deg is as far apart as two squares get, and 95 = 5 for a square.
    assert measure_accuracy.folded_angle_error("square", 44.0, -1.0) == -45.0
    assert measure_accuracy.folded_angle_error("square", 95.0, 5.0) == 0.0
    assert measure_accuracy.folded_angle_error("hexagon", 5.0, 25.0) == -20.0
    assert measure_accuracy.folded_angle_error("circle", 5.0, 129.0) == 0.0
    # A triangle turned 130 deg is 10 deg from where it started, not 130.
    assert measure_accuracy.folded_angle_error("triangle", 0.0, 130.0) == -10.0
