"""The grade-I dry run must send one line per part and notice a missing part."""

from __future__ import annotations

import cv2
import numpy as np

import pipeline
import rehearse_grade1


def make_detection(class_id: int, confidence: float) -> pipeline.Detection:
    """A square-ish box with real corners; the stub plane ignores the geometry."""
    return pipeline.Detection(
        class_id=class_id,
        confidence=confidence,
        center=(100.0, 100.0),
        angle_deg=12.5,
        corners=((80.0, 80.0), (120.0, 80.0), (120.0, 120.0), (80.0, 120.0)),
    )


# Fake detector that returns a fixed list of boxes.
class StubDetector:
    def __init__(self, detections: list[pipeline.Detection], **_kwargs) -> None:
        self.detections = detections

    # Ignores the picture and returns the same boxes.
    def detect(self, _image) -> list[pipeline.Detection]:
        return self.detections


class StubPlane:
    """Stands in for PlaneCalibration: one command per detection, angle kept."""

    def to_pick_command(self, detection: pipeline.Detection) -> pipeline.PickCommand:
        return pipeline.PickCommand(
            shape=detection.name,
            x_mm=10.0 * pipeline.SLOT[detection.name],
            y_mm=20.0,
            angle_deg=detection.angle_deg,
            z_mm=pipeline.PICK_Z_MM,
        )


def prepare(tmp_path, monkeypatch, detections: list[pipeline.Detection]) -> str:
    """A readable frame on disk, a stub detector and a stub plane."""
    frame = np.full((40, 40, 3), 127, np.uint8)
    cv2.imwrite(str(tmp_path / "frame.png"), frame)
    # Swap the real detector for the fake one.
    monkeypatch.setattr(rehearse_grade1.pipeline, "ShapeDetector", StubDetector)
    monkeypatch.setattr(rehearse_grade1.pipeline, "MODEL_PATH", "stub.pt")
    monkeypatch.setattr(rehearse_grade1, "StubDetector", StubDetector, raising=False)
    monkeypatch.setattr(rehearse_grade1.calibrate, "marker_scale",
                        lambda *_args, **_kwargs: StubPlane())
    monkeypatch.setattr(rehearse_grade1.pipeline, "ShapeDetector",
                        lambda **_kwargs: StubDetector(detections))
    return str(tmp_path / "frame.png")


def test_rehearsal_sends_one_line_per_part_and_reads_it_back(tmp_path, monkeypatch, capsys):
    detections = [make_detection(0, 0.9), make_detection(2, 0.8), make_detection(4, 0.7)]
    image = prepare(tmp_path, monkeypatch, detections)

    code = rehearse_grade1.main(["--image", image])

    printed = capsys.readouterr().out
    assert code == 0
    # Every part sent once and came back unchanged.
    assert "found 3 of 3 shape(s) | sent 3 | read back intact 3" in printed
    for shape in ("circle", "hexagon", "triangle"):
        assert shape in printed


def test_rehearsal_names_a_missing_shape_and_fails(tmp_path, monkeypatch, capsys):
    detections = [make_detection(0, 0.9), make_detection(2, 0.8)]
    image = prepare(tmp_path, monkeypatch, detections)

    code = rehearse_grade1.main(["--image", image])

    printed = capsys.readouterr().out
    assert code == 1
    # The missing shape is named and the run fails.
    assert "triangle: not in this frame" in printed


def test_choose_keeps_the_most_confident_detection_of_each_shape():
    detections = [make_detection(0, 0.5), make_detection(0, 0.9), make_detection(3, 0.9)]

    # Two circles seen; keep only the surest.
    chosen = rehearse_grade1.choose(detections, ("circle", "star"))

    assert [(name, detection.confidence) for name, detection in chosen] == [
        ("circle", 0.9), ("star", 0.9)]
