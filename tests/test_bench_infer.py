"""Headless tests for the advisory benchmark: it reports, it never stops."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

import bench_infer


class CountingDetector:
    """Stands in for `ShapeDetector`: counts calls and can be made slow."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls = 0

    # Count each call, optionally sleep, then find nothing.
    def detect(self, _frame):
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return []


# Write tiny test images whose brightness is the frame index.
def write_frames(folder, count: int) -> None:
    for index in range(count):
        cv2.imwrite(str(folder / f"frame_{index}.png"),
                    np.full((8, 8, 3), index, np.uint8))


def test_load_frames_keeps_name_order_and_the_limit(tmp_path):
    write_frames(tmp_path, 3)
    # Load only the first two frames, in name order.
    frames = bench_infer.load_frames(tmp_path, limit=2)
    assert len(frames) == 2
    assert int(frames[0][0, 0, 0]) == 0
    assert int(frames[1][0, 0, 0]) == 1


def test_bench_times_every_frame_even_over_the_gate(tmp_path, monkeypatch, capsys):
    """Being over budget is a report, not a stop: every frame still gets timed."""
    write_frames(tmp_path, 3)
    detector = CountingDetector(delay_s=0.002)
    monkeypatch.setattr(bench_infer, "ShapeDetector", lambda _model: detector)

    # Tiny budget makes every frame over the gate.
    code = bench_infer.main(["--frames", str(tmp_path), "--budget-ms", "0.001"])

    printed = capsys.readouterr().out
    assert code == 0
    assert detector.calls == 4  # one warm-up plus every frame
    assert "OVER THE GATE: 3 of 3" in printed
    assert "nothing stopped" in printed


def test_bench_says_within_the_gate_when_fast(tmp_path, monkeypatch, capsys):
    write_frames(tmp_path, 2)
    detector = CountingDetector()
    monkeypatch.setattr(bench_infer, "ShapeDetector", lambda _model: detector)

    assert bench_infer.main(["--frames", str(tmp_path), "--budget-ms", "60000"]) == 0
    assert "WITHIN THE GATE" in capsys.readouterr().out


def test_bench_refuses_an_empty_folder(tmp_path):
    # Empty folder is rejected before any timing starts.
    with pytest.raises(SystemExit) as stop:
        bench_infer.main(["--frames", str(tmp_path)])

    assert "no images" in str(stop.value)
