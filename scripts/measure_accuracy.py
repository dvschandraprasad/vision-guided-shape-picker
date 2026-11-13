from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import calibrate
import pipeline

LABELS_DIR = Path("data/dataset/labels/val")
IMAGES_DIR = Path("data/dataset/images/val")
# Assumed pocket angle tolerance used for the report.
POCKET_DEG = 15.0

@dataclass(frozen=True)
class Truth:

    shape: str
    centre: tuple[float, float]
    angle_deg: float

@dataclass(frozen=True)
class Sample:

    shape: str
    centre_px: float
    centre_mm: float | None
    angle_deg: float

# Finds the shape angle from its longest edge.
def obb_angle_deg(corners: np.ndarray) -> float:
    edges = [(corners[index], corners[(index + 1) % 4]) for index in range(4)]
    start, end = max(edges, key=lambda edge: float(np.linalg.norm(edge[1] - edge[0])))
    delta = end - start
    return (math.degrees(math.atan2(float(delta[1]), float(delta[0]))) + 45.0) % 180.0 - 45.0

def load_truth(label_path: Path, width: int, height: int) -> list[Truth]:
    truths: list[Truth] = []
    if not label_path.is_file():
        return truths
    for number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 9:
            raise ValueError(f"{label_path}:{number}: expected 9 fields, got {len(fields)}")
        class_id = int(fields[0])
        if class_id not in range(len(pipeline.CLASSES)):
            raise ValueError(f"{label_path}:{number}: class {class_id} outside 0..4")
        values = np.asarray([float(value) for value in fields[1:]], np.float64)
        # Label corners are fractions of image width and height.
        corners = np.stack([values[0::2] * width, values[1::2] * height], axis=1)
        truths.append(Truth(
            shape=pipeline.CLASSES[class_id],
            centre=(float(corners[:, 0].mean()), float(corners[:, 1].mean())),
            angle_deg=obb_angle_deg(corners),
        ))
    return truths

# Removes each shape's own symmetry before comparing angles.
def folded_angle_error(shape: str, first: float, second: float) -> float:
    return calibrate.fold_to_symmetry(shape, first - second)

def match(truths: list[Truth], detections: list[pipeline.Detection], max_px: float
          ) -> tuple[list[tuple[Truth, pipeline.Detection]], dict[str, int], dict[str, int]]:
    candidates: list[tuple[float, int, int]] = []
    for truth_index, truth in enumerate(truths):
        for detection_index, detection in enumerate(detections):
            if detection.name != truth.shape:
                continue
            distance = math.dist(truth.centre, detection.center)
            if distance <= max_px:
                candidates.append((distance, truth_index, detection_index))
    candidates.sort()

    used_truths: set[int] = set()
    used_detections: set[int] = set()
    pairs: list[tuple[Truth, pipeline.Detection]] = []
    # Closest pairs first; each box is used once.
    for _distance, truth_index, detection_index in candidates:
        if truth_index in used_truths or detection_index in used_detections:
            continue
        used_truths.add(truth_index)
        used_detections.add(detection_index)
        pairs.append((truths[truth_index], detections[detection_index]))

    missed: dict[str, int] = {}
    for index, truth in enumerate(truths):
        if index not in used_truths:
            missed[truth.shape] = missed.get(truth.shape, 0) + 1
    extra: dict[str, int] = {}
    for index, detection in enumerate(detections):
        if index not in used_detections:
            extra[detection.name] = extra.get(detection.name, 0) + 1
    return pairs, missed, extra

def percentile(ordered: list[float], fraction: float) -> float:
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="centre and folded-angle error per shape, against the val labels")
    parser.add_argument("--model", default=pipeline.MODEL_PATH, help="OBB weights")
    parser.add_argument("--labels", default=str(LABELS_DIR), help="label folder")
    parser.add_argument("--images", default=str(IMAGES_DIR), help="image folder")
    parser.add_argument("--calibration", default=str(calibrate.CALIBRATION_PATH),
                        help="used only for the millimetre column")
    parser.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    parser.add_argument("--max-px", type=float, default=30.0,
                        help="a detection further than this from a label is extra")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N frames")
    parser.add_argument("--pocket-deg", type=float, default=POCKET_DEG,
                        help="how much angle error the mould pocket absorbs")
    arguments = parser.parse_args(argv)

    labels = Path(arguments.labels)
    images = Path(arguments.images)
    if not labels.is_dir():
        print(f"no label folder: {labels}")
        return 2
    pairs_on_disk = sorted(path for path in labels.glob("*.txt"))
    if arguments.limit is not None:
        pairs_on_disk = pairs_on_disk[:arguments.limit]
    if not pairs_on_disk:
        print(f"no labels in {labels}")
        return 2

    try:
        plane: calibrate.PlaneCalibration | None = calibrate.load_calibration(arguments.calibration)
    except (FileNotFoundError, ValueError) as error:
        plane = None
        print(f"note: no usable calibration ({error}) - millimetres are left out")
    if plane is not None:
        print(f"calibration {arguments.calibration}")

    detector = pipeline.ShapeDetector(model_path=arguments.model, conf=arguments.conf)
    samples: list[Sample] = []
    missed: dict[str, int] = {}
    extra: dict[str, int] = {}
    scored = 0
    warmed = False
    for label_path in pairs_on_disk:
        image_path = next((images / (label_path.stem + suffix)
                           for suffix in pipeline.IMAGE_SUFFIXES
                           if (images / (label_path.stem + suffix)).is_file()), None)
        if image_path is None:
            print(f"  {label_path.stem}: no image, skipped")
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  {image_path}: unreadable, skipped")
            continue
        # Image size turns label fractions into pixel positions.
        height, width = image.shape[:2]
        truths = load_truth(label_path, width, height)
        if not warmed:
            # Warm up the detector before timing this image.
            detector.detect(image)
            warmed = True
        detections = detector.detect(image)
        pairs, gone, spare = match(truths, detections, arguments.max_px)
        for shape, count in gone.items():
            missed[shape] = missed.get(shape, 0) + count
        for shape, count in spare.items():
            extra[shape] = extra.get(shape, 0) + count
        scored += 1
        for truth, detection in pairs:
            centre_px = math.dist(truth.centre, detection.center)
            centre_mm = None
            if plane is not None:
                first = plane.to_mm(truth.centre)
                second = plane.to_mm(detection.center)
                centre_mm = math.dist(first, second)
            samples.append(Sample(
                shape=truth.shape,
                centre_px=centre_px,
                centre_mm=centre_mm,
                angle_deg=folded_angle_error(truth.shape, truth.angle_deg, detection.angle_deg),
            ))

    print()
    print(f"scored {scored} frame(s), {len(samples)} matched object(s)")
    if not samples:
        print("nothing matched - check --conf and the label folder")
        return 1

    header = f"{'shape':<10}{'n':>4}{'miss':>6}{'extra':>7}{'centre p50':>12}{'p95':>8}"
    if plane is not None:
        header += f"{'p95 mm':>9}"
    header += f"{'angle p50':>11}{'p95':>8}{'max':>8}"
    print(header)
    verdicts: list[tuple[str, float]] = []
    for shape in pipeline.CLASSES:
        group = [sample for sample in samples if sample.shape == shape]
        if not group:
            continue
        centres = sorted(sample.centre_px for sample in group)
        angles = sorted(sample.angle_deg for sample in group)
        row = (f"{shape:<10}{len(group):>4}{missed.get(shape, 0):>6}{extra.get(shape, 0):>7}"
               f"{statistics.median(centres):>11.1f}px{percentile(centres, 0.95):>7.1f}")
        if plane is not None:
            metres = sorted(sample.centre_mm or 0.0 for sample in group)
            row += f"{percentile(metres, 0.95):>8.1f}"
        row += (f"{statistics.median(angles):>10.1f}d{percentile(angles, 0.95):>7.1f}"
                f"{angles[-1]:>7.1f}")
        print(row)
        verdicts.append((shape, percentile(angles, 0.95)))

    print()
    usable: list[str] = []
    noisy: list[str] = []
    for shape, angle in verdicts:
        half = calibrate.SYMMETRY_DEG[shape] / 2.0
        if half <= 0.0:
            print(f"{shape:<10}no heading at all - the robot never turns it")
        # Flag errors near the shape symmetry limit.
        elif angle >= 0.75 * half:
            print(f"{shape:<10}p95 {angle:5.1f} of a possible {half:.0f} deg - "
                  "the box angle carries no heading for this shape")
            noisy.append(shape)
        elif angle <= arguments.pocket_deg:
            print(f"{shape:<10}p95 {angle:5.1f} deg of {half:.0f} - fits a "
                  f"{arguments.pocket_deg:.0f} deg pocket")
            usable.append(shape)
        else:
            print(f"{shape:<10}p95 {angle:5.1f} of {half:.0f} deg - beyond a "
                  f"{arguments.pocket_deg:.0f} deg pocket")
            noisy.append(shape)

    print()
    if noisy:
        print(f"POCKET TIGHT for {', '.join(noisy)}: those angles are not usable, so the pocket "
              "has to align the part itself (decision 1A) - measure the real gap (open question 22)")
    else:
        print(f"POCKET OK: {', '.join(usable)} fit inside {arguments.pocket_deg:.0f} deg")
    print("Every angle here is measured after the shape's own symmetry fold, so an upside-down "
          "triangle still scores well; the bench has to catch that one.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
