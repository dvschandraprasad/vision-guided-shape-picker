from __future__ import annotations

import pipeline

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline import IMAGE_SUFFIXES, MODEL_PATH, ShapeDetector

# Time limit for one frame, in milliseconds.
BUDGET_MS = 3000.0
# Default folder holding the captured test frames.
FRAME_DIR = pipeline.REPO_ROOT / "data" / "captured_frames"


# Load images from a folder in name order.
def load_frames(folder: Path, limit: int | None = None) -> list[np.ndarray]:
    if not folder.is_dir():
        raise SystemExit(f"no frame folder: {folder}")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        paths = paths[:limit]
    frames: list[np.ndarray] = []
    for path in paths:
        frame: np.ndarray | None = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"could not read: {path}")
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no images in {folder}")
    return frames


# Value at a given fraction of the sorted list.
def percentile(ordered: list[float], fraction: float) -> float:
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


# Time one detect call and count the shapes found.
def time_frame(detector: ShapeDetector, frame: np.ndarray) -> tuple[float, int]:
    started = time.perf_counter()
    detections = detector.detect(frame)
    return (time.perf_counter() - started) * 1000.0, len(detections)


def print_timings(rows: list[tuple[float, int]], budget_ms: float) -> None:
    for index, (elapsed_ms, shapes) in enumerate(rows, start=1):
        over = elapsed_ms > budget_ms
        note = f"  OVER {budget_ms:.0f} ms - still timing" if over else ""
        print(f"frame {index}/{len(rows)}: {elapsed_ms:.0f} ms, {shapes} shape(s){note}")


# Print average, middle and slowest frame times.
def print_summary(rows: list[tuple[float, int]], budget_ms: float) -> None:
    ordered = sorted(elapsed for elapsed, _shapes in rows)
    print(f"mean {statistics.fmean(ordered):.0f} ms | p50 {statistics.median(ordered):.0f} ms | "
          f"p95 {percentile(ordered, 0.95):.0f} ms | max {ordered[-1]:.0f} ms")
    slow = sum(1 for elapsed in ordered if elapsed > budget_ms)
    if slow:
        print(f"OVER THE GATE: {slow} of {len(ordered)} frame(s) above "
              f"{budget_ms:.0f} ms inference - reported, nothing stopped")
    else:
        print(f"WITHIN THE GATE: every timed frame under {budget_ms:.0f} ms inference")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="time inference against the 3 s gate (advisory, never fatal)")
    parser.add_argument("--model", default=str(MODEL_PATH),
                        help="weights: .pt, or a *_openvino_model / *_ncnn_model folder")
    parser.add_argument("--frames", default=str(FRAME_DIR),
                        help="folder of images to time")
    parser.add_argument("--limit", type=int, default=None,
                        help="time only the first N frames")
    parser.add_argument("--repeat", type=int, default=1,
                        help="how many passes over the frames")
    parser.add_argument("--budget-ms", type=float, default=BUDGET_MS,
                        help="the gate, on inference only")
    arguments = parser.parse_args(argv)

    frames = load_frames(Path(arguments.frames), arguments.limit)
    passes = max(1, arguments.repeat)
    detector = ShapeDetector(arguments.model)
    print(f"model {arguments.model}")
    print(f"{len(frames)} frame(s) x {passes} pass(es) | "
          f"gate {arguments.budget_ms:.0f} ms inference")

    # First frame is a warm-up, its time is ignored.
    detector.detect(frames[0])
    # Time every frame again for each repeat pass.
    rows = [time_frame(detector, frame) for frame in frames * passes]
    print_timings(rows, arguments.budget_ms)
    print_summary(rows, arguments.budget_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
