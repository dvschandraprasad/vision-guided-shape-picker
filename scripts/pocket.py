from __future__ import annotations

import argparse
import math

import cv2
import numpy as np

# Part shapes this script can check.
SHAPES = ("square", "hexagon", "triangle", "star", "circle")
HALF_PERIOD_DEG = {"square": 45.0, "hexagon": 30.0, "triangle": 60.0, "star": 36.0}
STAR_INNER_RATIO = 0.40
# Measured rotation errors per shape, used when none given.
MEASURED_P95_DEG = {"square": 5.5, "hexagon": 28.0, "triangle": 55.3, "star": 30.9,
                    "circle": 0.0}

# Corner points of the part outline, centred on zero.
def outline(shape: str, size_mm: float) -> np.ndarray | None:
    radius = size_mm / 2.0
    if shape == "circle":
        return None
    if shape == "square":
        corners = [(radius * math.cos(math.radians(45 + 90 * i)),
                    radius * math.sin(math.radians(45 + 90 * i))) for i in range(4)]
    elif shape == "hexagon":
        corners = [(radius * math.cos(math.radians(30 + 60 * i)),
                    radius * math.sin(math.radians(30 + 60 * i))) for i in range(6)]
    elif shape == "triangle":
        corners = [(radius * math.cos(math.radians(90 + 120 * i)),
                    radius * math.sin(math.radians(90 + 120 * i))) for i in range(3)]
    elif shape == "star":
        corners = []
        for index in range(10):
            reach = radius if index % 2 == 0 else radius * STAR_INNER_RATIO
            angle = math.radians(90 + 36 * index)
            corners.append((reach * math.cos(angle), reach * math.sin(angle)))
    else:
        raise ValueError(f"unknown shape: {shape}")
    return np.array(corners, dtype=float)

# Shortest distance from the centre to any wall.
def _nearest_wall(polygon: np.ndarray) -> float:
    best = math.inf
    for index in range(len(polygon)):
        start, end = polygon[index], polygon[(index + 1) % len(polygon)]
        edge = end - start
        length2 = float(edge @ edge)
        along = 0.0 if length2 == 0 else float(np.clip(-(start @ edge) / length2, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(start + edge * along)))
    return best

# Sample points spread along the part outline.
def _wall_points(polygon: np.ndarray, spacing_mm: float) -> np.ndarray:
    points = []
    for index in range(len(polygon)):
        start, end = polygon[index], polygon[(index + 1) % len(polygon)]
        steps = max(2, int(float(np.linalg.norm(end - start)) / spacing_mm))
        for step in range(steps):
            points.append(start + (end - start) * (step / steps))
    return np.array(points)

# Check whether the rotated outline fits the pocket.
def fits(shape: str, gap_mm: float, angle_deg: float, size_mm: float = 50.0) -> bool:
    part = outline(shape, size_mm)
    if part is None:
        return True
    # Pocket is the outline grown by the wall gap.
    pocket_outline = part * (1.0 + gap_mm / _nearest_wall(part))
    rotation = cv2.getRotationMatrix2D((0.0, 0.0), angle_deg, 1.0)
    turned = _wall_points(part, 0.5) @ rotation[:, :2].T
    pocket32 = pocket_outline.astype(np.float32)
    return all(cv2.pointPolygonTest(pocket32, (float(x), float(y)), False) >= 0
               for x, y in turned)

# Find the largest angle whose outline fits the pocket.
def insert_angle_deg(shape: str, gap_mm: float, size_mm: float = 50.0,
                     step_deg: float = 0.1) -> float | None:
    if shape == "circle":
        return None
    if gap_mm < 0:
        raise ValueError("gap must not be negative")
    limit = HALF_PERIOD_DEG[shape]
    angle = 0.0
    while angle + step_deg < limit:
        angle += step_deg
        if not fits(shape, gap_mm, angle, size_mm):
            return round(angle - step_deg, 1)
    return round(limit, 1)

# Chamfer widens the mouth, so it acts like more gap.
def mouth_angle_deg(shape: str, clearance_mm: float, chamfer_mm: float,
                    size_mm: float = 50.0) -> float | None:
    if chamfer_mm < 0:
        raise ValueError("chamfer must not be negative")
    return insert_angle_deg(shape, clearance_mm + chamfer_mm, size_mm)

# Smallest extra mouth width that lets this error angle in.
def chamfer_needed_mm(shape: str, angle_deg: float, clearance_mm: float = 0.5,
                      size_mm: float = 50.0, step_mm: float = 0.1) -> float | None:
    if shape not in SHAPES:
        raise ValueError(f"unknown shape: {shape}")
    if angle_deg < 0 or clearance_mm < 0:
        raise ValueError("angles and clearances must not be negative")
    if shape == "circle":
        return 0.0
    if angle_deg >= HALF_PERIOD_DEG[shape]:
        return None
    extra = 0.0
    while extra < size_mm:
        if (insert_angle_deg(shape, clearance_mm + extra, size_mm) or 0.0) >= angle_deg:
            return round(extra, 1)
        extra += step_mm
    return None

def table(gaps_mm: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
          size_mm: float = 50.0) -> list[tuple[str, list[float | None]]]:
    return [(shape, [insert_angle_deg(shape, gap, size_mm) for gap in gaps_mm])
            for shape in SHAPES]

# Command line entry: prints the angle tables only.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="angle window a pocket accepts, plain or chamfered")
    parser.add_argument("--size-mm", type=float, default=50.0, help="long diameter of the part")
    parser.add_argument("--gap-mm", type=float, nargs="*", default=[0.5, 1.0, 2.0, 3.0],
                        help="straight-wall clearance to test, from the calipers")
    parser.add_argument("--chamfer-mm", type=float, nargs="*", default=[0.0, 1.0, 2.0, 4.0],
                        help="mouth widening to test")
    parser.add_argument("--clearance-mm", type=float, default=0.5,
                        help="clearance assumed when asking for a chamfer")
    parser.add_argument("--error-deg", type=float, nargs="*", default=None,
                        help="angle errors to size a chamfer for; default: the measured p95")
    arguments = parser.parse_args(argv)

    gaps = tuple(arguments.gap_mm)
    print(f"part {arguments.size_mm:.0f} mm | largest rotation a straight wall accepts, degrees")
    print(f"{'shape':9s}" + "".join(f"{gap:>8.1f}" for gap in gaps))
    for shape, angles in table(gaps, arguments.size_mm):
        cells = "".join(f"{'any' if angle is None else f'{angle:.1f}':>8s}" for angle in angles)
        print(f"{shape:9s}{cells}")

    chamfers = tuple(arguments.chamfer_mm)
    print()
    print(f"mouth {chamfers[0]:.0f}-{chamfers[-1]:.0f} mm wider, {arguments.clearance_mm:.1f} mm "
          f"clearance | largest rotation that still enters, degrees")
    print(f"{'shape':9s}" + "".join(f"{chamfer:>8.1f}" for chamfer in chamfers))
    for shape in SHAPES:
        cells = "".join(f"{'any' if (a := mouth_angle_deg(shape, arguments.clearance_mm, chamfer,
                                                        arguments.size_mm)) is None else f'{a:.1f}':>8s}"
                        for chamfer in chamfers)
        print(f"{shape:9s}{cells}")

    errors = arguments.error_deg
    if errors:
        pairs = [(shape, errors[index] if index < len(errors) else None)
                 for index, shape in enumerate(SHAPES)]
    else:
        pairs = list(MEASURED_P95_DEG.items())
    print()
    print(f"chamfer needed for that error at {arguments.clearance_mm:.1f} mm clearance")
    for shape, error in pairs:
        if error is None:
            continue
        needed = chamfer_needed_mm(shape, error, arguments.clearance_mm, arguments.size_mm)
        verdict = "never - the mouth cannot reach it" if needed is None else f"{needed:.1f} mm wider"
        print(f"  {shape:9s} {error:5.1f} deg -> {verdict}")

    print("half-symmetry limits: "
          + ", ".join(f"{name} {value:.0f}" for name, value in HALF_PERIOD_DEG.items()))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
