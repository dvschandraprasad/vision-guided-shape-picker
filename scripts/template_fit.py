from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

import pipeline

# Part sizes in millimetres.
SQUARE_SIDE_MM = 45.0
TRIANGLE_SIDE_MM = 50.0
HEXAGON_FLATS_MM = 50.0
STAR_TIP_CHORD_MM = 34.0
STAR_NOTCH_CHORD_MM = 14.0

# Raster resolution: one cell is a quarter millimetre.
MM_PER_CELL = 0.25
COARSE_STEP_DEG = 1.0
FINE_STEP_DEG = 0.1
SHIFT_STEPS_MM = (-1.0, -0.5, 0.0, 0.5, 1.0)
ENABLED = True
# Minimum fit scores accepted for each shape.
MIN_SCORE = {"square": 0.85, "hexagon": 0.85, "triangle": 0.85, "star": 0.72}
SIZE_LIMITS = (0.80, 1.20)


# Points evenly spaced on a circle, in millimetres.
def _ring(radius: float, count: int, phase_deg: float = 0.0) -> np.ndarray:
    angles = np.radians(phase_deg + 360.0 * np.arange(count) / count)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _chord_to_radius(chord: float, count: int) -> float:
    return chord / (2.0 * math.sin(math.pi / count))


# Build the ideal outline of every shape in millimetres.
def build_templates() -> dict[str, np.ndarray]:
    half = SQUARE_SIDE_MM / 2.0
    outer = _chord_to_radius(STAR_TIP_CHORD_MM, 5)
    inner = _chord_to_radius(STAR_NOTCH_CHORD_MM, 5)
    star = []
    for k in range(5):
        for radius, phase in ((outer, 72.0 * k), (inner, 72.0 * k + 36.0)):
            angle = math.radians(phase)
            star.append((radius * math.cos(angle), radius * math.sin(angle)))
    return {
        "square": np.array([[half, half], [-half, half], [-half, -half], [half, -half]]),
        "triangle": _ring(TRIANGLE_SIDE_MM / math.sqrt(3.0), 3),
        "hexagon": _ring(HEXAGON_FLATS_MM / math.sqrt(3.0), 6),
        "star": np.array(star),
    }


TEMPLATES = build_templates()


@dataclass(frozen=True)
class Fit:
    angle_deg: float
    score: float
    size_ratio: float
    centre_mm: tuple[float, float]


# Rotate points by an angle, anticlockwise.
def _rotate(points: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


# Draw outline into a grid; rows count downwards.
def _raster(points_mm: np.ndarray, size: int) -> np.ndarray:
    canvas = np.zeros((size, size), np.uint8)
    cells = np.empty_like(points_mm, dtype=np.float64)
    cells[:, 0] = size / 2.0 + points_mm[:, 0] / MM_PER_CELL
    cells[:, 1] = size / 2.0 - points_mm[:, 1] / MM_PER_CELL
    polygon = np.round(cells * 16.0).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [polygon], 1, lineType=cv2.LINE_8, shift=4)
    return canvas.astype(bool)


def template_mm(shape: str, angle_deg: float, scale: float = 1.0,
                centre_mm: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    return _rotate(TEMPLATES[shape] * scale, angle_deg) + np.asarray(centre_mm)


# Try many angles and keep the best overlap with the shape.
def fit_outline_mm(shape: str, outline_mm: Any) -> Fit | None:
    if shape not in TEMPLATES:
        return None
    period = pipeline.SYMMETRY_DEG[shape]
    polygon = np.asarray(outline_mm, np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        return None
    # Use the area centre as the rotation origin.
    moments = cv2.moments(polygon.astype(np.float32).reshape(-1, 1, 2))
    if abs(moments["m00"]) <= 1e-9:
        return None
    cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    centred = polygon - (cx, cy)

    template = TEMPLATES[shape]
    ratio = math.sqrt(abs(moments["m00"]) /
                      abs(cv2.contourArea(template.astype(np.float32).reshape(-1, 1, 2))))
    scale = float(np.clip(ratio, *SIZE_LIMITS))
    scaled = template * scale
    reach = max(float(np.hypot(*centred.T).max()), float(np.hypot(*scaled.T).max()))
    size = int(2.0 * (reach + 2.0) / MM_PER_CELL) + 8
    part = _raster(centred, size)
    part_count = int(np.count_nonzero(part))
    if part_count == 0:
        return None

    def score(angle: float, dx: float = 0.0, dy: float = 0.0) -> float:
        shape_cells = _raster(_rotate(scaled, angle) + (dx, dy), size)
        inter = np.count_nonzero(part & shape_cells)
        union = np.count_nonzero(part | shape_cells)
        return inter / union if union else 0.0

    coarse = np.arange(0.0, period, COARSE_STEP_DEG)
    best_angle = float(coarse[int(np.argmax([score(a) for a in coarse]))])

    shifts = [(dx, dy) for dx in SHIFT_STEPS_MM for dy in SHIFT_STEPS_MM]
    best_dx, best_dy = max(shifts, key=lambda shift: score(best_angle, *shift))

    fine = np.arange(best_angle - COARSE_STEP_DEG,
                     best_angle + COARSE_STEP_DEG + 1e-9, FINE_STEP_DEG)
    fine_scores = [score(a, best_dx, best_dy) for a in fine]
    index = int(np.argmax(fine_scores))
    return Fit(
        angle_deg=float(fine[index] % period),
        score=float(fine_scores[index]),
        size_ratio=float(ratio),
        centre_mm=(float(cx + best_dx), float(cy + best_dy)),
    )


# Several colour based guesses of which pixels are the part.
def _candidate_masks(crop: np.ndarray, box: tuple[int, int, int, int]) -> list[np.ndarray]:
    masks = [
        pipeline.foreground_mask(crop),
        pipeline.foreground_mask(crop, pipeline._TOLERANCE_LOW),
    ]
    annulus = pipeline._annulus_mask(crop, box)
    if annulus is not None:
        masks.append(annulus)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1]
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    left, top, right, bottom = box
    cy, cx = (top + bottom) // 2, (left + right) // 2
    ry, rx = max(2, (bottom - top) // 10), max(2, (right - left) // 10)
    core = np.median(lab[cy - ry:cy + ry + 1, cx - rx:cx + rx + 1].reshape(-1, 3), axis=0)
    distance = np.linalg.norm(lab - core, axis=2)
    distance = np.clip(distance * (255.0 / max(1.0, float(distance.max()))), 0, 255).astype(np.uint8)
    _level, near = cv2.threshold(distance, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(near)
    for channel in (grey, saturation):
        _level, bright = cv2.threshold(cv2.GaussianBlur(channel, (5, 5), 0), 0, 1,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.extend([bright, 1 - bright])
    return [pipeline._clip_to_box(mask.astype(np.uint8), box) for mask in masks]


# Crop around one detection and keep plausible outlines.
def part_contours_px(image: np.ndarray, detection: pipeline.Detection) -> list[np.ndarray]:
    corners = np.asarray(detection.corners, np.float64)
    left, top = corners.min(axis=0)
    right, bottom = corners.max(axis=0)
    pad = max(4, int(pipeline._PAD_FRACTION * max(right - left, bottom - top)))
    height, width = image.shape[:2]
    x0, x1 = max(0, int(left) - pad), min(width, int(right) + pad + 1)
    y0, y1 = max(0, int(top) - pad), min(height, int(bottom) + pad + 1)
    crop = image[y0:y1, x0:x1]
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return []
    grow = 6
    box = (int(left) - x0 - grow, int(top) - y0 - grow,
           int(right) - x0 + grow, int(bottom) - y0 + grow)
    box_area = max(1.0, (right - left) * (bottom - top))
    limit = pipeline._MIN_FILL.get(detection.name, pipeline._MIN_FILL_DEFAULT)
    centre = (detection.center[0] - x0, detection.center[1] - y0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    found = []
    for mask in _candidate_masks(crop, box):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        keep = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area / box_area < 0.5 * limit or area < 200.0:
                continue
            if pipeline._touches_border(contour, crop.shape[0], crop.shape[1]):
                continue
            inside = cv2.pointPolygonTest(contour, (float(centre[0]), float(centre[1])), False) >= 0
            keep.append((inside, area, contour))
        keep.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _inside, _area, contour in keep[:2]:
            found.append(np.asarray(contour, np.float64).reshape(-1, 2) + (x0, y0))
    return found


def part_contour_px(image: np.ndarray, detection: pipeline.Detection) -> np.ndarray | None:
    found = part_contours_px(image, detection)
    return found[0] if found else None


# True when the fit is good enough to use its angle.
def usable(fit: Fit | None, shape: str) -> bool:
    return (fit is not None
            and fit.score >= MIN_SCORE.get(shape, 1.0)
            and SIZE_LIMITS[0] <= fit.size_ratio <= SIZE_LIMITS[1])


def fit_detection_detail(image: np.ndarray, detection: pipeline.Detection,
                         plane: Any) -> tuple[Fit | None, np.ndarray | None, int]:
    if detection.name not in TEMPLATES:
        return None, None, 0
    best, best_contour = None, None
    candidates = part_contours_px(image, detection)
    for contour in candidates:
        fit = fit_outline_mm(detection.name, plane.corners_mm(contour))
        if fit is None or not SIZE_LIMITS[0] <= fit.size_ratio <= SIZE_LIMITS[1]:
            continue
        if best is None or fit.score > best.score:
            best, best_contour = fit, contour
    return best, best_contour, len(candidates)


def fit_detection(image: np.ndarray, detection: pipeline.Detection, plane: Any) -> Fit | None:
    return fit_detection_detail(image, detection, plane)[0]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from pathlib import Path

    import calibrate

    parser = argparse.ArgumentParser(
        description="compare the template-fit angle with the current angle, part by part")
    parser.add_argument("--image", help="saved frame; omit to grab the Basler once")
    parser.add_argument("--calibration", default=str(calibrate.CALIBRATION_PATH),
                        help="used only when the frame has no visible marker")
    parser.add_argument("--model", default=pipeline.MODEL_PATH, help="OBB weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--overlay", default="template_check.png")
    arguments = parser.parse_args(argv)

    loaded = pipeline.read_image(arguments.image, "live frame")
    if loaded is None:
        print(f"cannot read {arguments.image}", file=sys.stderr)
        return 2
    image, source = loaded
    try:
        plane = calibrate.marker_scale(image)
        origin = "marker in this frame"
    except calibrate.MarkerNotFound:
        plane = calibrate.load_plane(arguments.calibration)
        if plane is None:
            return 2
        origin = arguments.calibration

    detections = pipeline.ShapeDetector(model_path=arguments.model, conf=arguments.conf).detect(image)
    print(f"frame        {source}")
    print(f"calibration  {origin}")
    print("shape      conf   current   template  score  size   used")
    canvas = image.copy()
    inverse = np.linalg.inv(plane.matrix_3x3)
    for detection in sorted(detections, key=lambda d: d.center[0]):
        shape = detection.name
        if shape not in TEMPLATES:
            print(f"{shape:9s} {detection.confidence:5.2f}   (no angle for this shape)")
            continue
        import template_fit as switch
        # Turn off template fit to measure the old angle.
        switch.ENABLED = False
        _x, _y, old_raw = plane.parts_mm(detection, image)
        switch.ENABLED = True
        old = pipeline.pick_angle(shape, old_raw)
        fit = fit_detection(image, detection, plane)
        if fit is None:
            print(f"{shape:9s} {detection.confidence:5.2f}  {old:+7.1f}   no outline        current")
            continue
        new = pipeline.pick_angle(shape, fit.angle_deg)
        good = usable(fit, shape)
        print(f"{shape:9s} {detection.confidence:5.2f}  {old:+7.1f}   {new:+7.1f}   {fit.score:5.2f}  x{fit.size_ratio:4.2f}  "
              f"{'template' if good else 'current (fit too poor)'}")
        outline = template_mm(shape, fit.angle_deg, min(max(fit.size_ratio, SIZE_LIMITS[0]), SIZE_LIMITS[1]),
                              fit.centre_mm)
        pixels = cv2.perspectiveTransform(outline.reshape(-1, 1, 2), inverse).reshape(-1, 2)
        # Yellow means template angle used, red means fallback.
        colour = (0, 255, 255) if good else (0, 0, 255)
        cv2.polylines(canvas, [np.round(pixels).astype(np.int32).reshape(-1, 1, 2)], True, colour, 2)
        cv2.putText(canvas, f"{shape} {new:+.1f}", (int(detection.center[0]) + 8, int(detection.center[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    target = Path(arguments.overlay)
    cv2.imwrite(str(target), canvas)
    print(f"overlay      {target.resolve()}  (yellow = template used, red = fell back)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
