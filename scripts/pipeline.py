from __future__ import annotations

import contextlib
import errno
import math
import socket
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import cv2
import numpy as np

CLASSES = ("circle", "square", "hexagon", "star", "triangle")

ALLOWED_CLASS_IDS = (0, 2, 4)

SLOT = {"circle": 1, "square": 2, "hexagon": 3, "triangle": 4, "star": 5}
SHAPE_BY_SLOT = {slot: shape for shape, slot in SLOT.items()}

PICK_Z_MM = 10.0

SYMMETRY_DEG = {
    "circle": 360.0,
    "square": 90.0,
    "hexagon": 60.0,
    "triangle": 120.0,
    "star": 72.0,
}

ANGLE_SOURCE = {
    "circle": None,
    "square": "box",
    "hexagon": "outline",
    "triangle": "outline",
    "star": "outline",
}

SENDS_ANGLE_DEG = {shape: source is not None for shape, source in ANGLE_SOURCE.items()}

ANGLE_OFFSET_DEG = 0.0

ANGLE_SIGN = 1.0

SHAPE_OFFSET_DEG: dict[str, float] = {
    "circle": 0.0,
    "square": 0.0,
    "hexagon": 0.0,
    "triangle": 0.0,
    "star": 0.0,
}

def fold_to_symmetry(shape: str, angle_deg: float) -> float:
    if shape not in SYMMETRY_DEG:
        raise ValueError(f"unknown shape {shape!r}; expected one of {list(SYMMETRY_DEG)}")

    # Fold the angle into one symmetry period, centred on zero.
    period = SYMMETRY_DEG[shape]
    if period >= 360.0:
        return 0.0
    return float((angle_deg + period / 2.0) % period - period / 2.0)

def pick_angle(shape: str, angle_deg: float, offset_deg: float = ANGLE_OFFSET_DEG) -> float:
    if shape not in SYMMETRY_DEG:
        raise ValueError(f"unknown shape {shape!r}; expected one of {list(SYMMETRY_DEG)}")
    if not SENDS_ANGLE_DEG[shape]:
        return 0.0
    reference = SHAPE_OFFSET_DEG.get(shape, 0.0)
    # Add offsets, fold the angle, then apply the turn sign.
    return ANGLE_SIGN * fold_to_symmetry(shape, angle_deg + offset_deg + reference)

def sending_shapes() -> str:
    names = [shape for shape, sends in SENDS_ANGLE_DEG.items() if sends]
    return ", ".join(names) if names else "none"

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})

def _find_repo_root(start: Path) -> Path:
    """Walk up from this file to the folder that holds runs/obb (or data/).

    The old check looked one level up only, and skipped that step as soon as a
    stray scripts/data existed - which left MODEL_PATH at scripts/runs/... and
    made YOLO() die on a missing best.pt.
    """
    for candidate in (start, *start.parents):
        if (candidate / "runs" / "obb").is_dir():
            return candidate
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
WEIGHTS_DIR = REPO_ROOT / "runs" / "obb" / "obb_train" / "weights"


def _default_weights() -> Path:
    """The weights this machine actually has.

    best.pt is the training artefact and is deliberately not shipped in the Pi
    archive, so resolve to the ncnn export first (ARM, no torch compute), then
    the onnx file, then best.pt for a PC that trained the model.
    """
    for name in ("best_ncnn_model", "best.onnx", "best.pt"):
        candidate = WEIGHTS_DIR / name
        if candidate.exists():
            return candidate
    return WEIGHTS_DIR / "best.pt"


MODEL_PATH = _default_weights()
ANGLE_TEACH_PATH = REPO_ROOT / "data" / "angle_teach.json"


# shape -> taught zero theta0 (camera frame) actually applied by apply_angle_teach
TAUGHT_ZERO_DEG: dict[str, float] = {}
# "none" (no file, defaults), "applied" (sign + taught zeros in use) or
# "ignored" (a file exists but could not be used; defaults in use)
ANGLE_TEACH_STATE = "none"
ANGLE_TEACH_DETAIL = ""


def untaught(shape: str) -> bool:
    """True for an angled shape that would get the taught sign with offset 0."""
    return (ANGLE_TEACH_STATE == "applied" and SENDS_ANGLE_DEG.get(shape, False)
            and shape not in TAUGHT_ZERO_DEG)


def send_blocked() -> str | None:
    """Why sending to the robot must not start, or None."""
    if ANGLE_TEACH_STATE == "ignored":
        return (f"REFUSED to send: the angle teach file exists but is not used "
                f"({ANGLE_TEACH_DETAIL}). Fix or re-teach it (teach_angle.py --show), "
                f"or move {Path(ANGLE_TEACH_PATH).name} away to run on the defaults.")
    return None


def pick_blocked(shape: str) -> str | None:
    """Why one pick of this shape must not be sent, or None."""
    if untaught(shape):
        return (f"REFUSED {shape}: angle teach is active but {shape} is not taught "
                f"(run AngleTeach.mod TeachShape + teach_angle.py for it)")
    return None


def apply_angle_teach(path: str | Path = ANGLE_TEACH_PATH) -> str:
    # Robot reverse teach (AngleTeach.mod + teach_angle.py): sign and per-shape zero.
    # Applied only when every taught shape agrees on the sign; otherwise nothing changes.
    global ANGLE_SIGN, ANGLE_TEACH_STATE, ANGLE_TEACH_DETAIL
    import json
    defaults = f"angle teach: none (defaults: sign {ANGLE_SIGN:+.0f}, offsets 0)"
    source = Path(path)
    ANGLE_TEACH_STATE, ANGLE_TEACH_DETAIL = "none", ""
    if not source.is_file():
        return defaults
    ANGLE_TEACH_STATE = "ignored"
    try:
        values = json.loads(source.read_text(encoding="utf-8"))
        sign = None if values.get("sign") is None else float(values["sign"])
        complete = {name: entry for name, entry in values.get("shapes", {}).items()
                    if name in SHAPE_OFFSET_DEG and entry.get("k") is not None}
        shapes = {name: float(entry["theta0"]) for name, entry in complete.items()}
        ks = {name: int(entry["k"]) for name, entry in complete.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        ANGLE_TEACH_DETAIL = f"unreadable: {error}"
        return f"angle teach: IGNORED {source.name} ({error}); defaults in use"
    # All taught shapes must agree on the sign, or none is used.
    if len(set(ks.values())) > 1:
        detail = ", ".join(f"{name} {k:+d}" for name, k in sorted(ks.items()))
        ANGLE_TEACH_DETAIL = f"shapes disagree on the sign: {detail}"
        return (f"angle teach: IGNORED {source.name} - SHAPES DISAGREE ON THE SIGN ({detail}); "
                "defaults in use - re-teach")
    if sign not in (1.0, -1.0) or not shapes:
        ANGLE_TEACH_DETAIL = "no complete shape"
        return f"angle teach: IGNORED {source.name} (no complete shape); defaults in use"
    if set(ks.values()) != {int(sign)}:
        ANGLE_TEACH_DETAIL = f"sign {sign:+.0f} does not match the taught shapes"
        return (f"angle teach: IGNORED {source.name} (sign {sign:+.0f} does not match the "
                "taught shapes); defaults in use - re-teach")
    ANGLE_SIGN = sign
    for name, theta0 in shapes.items():
        SHAPE_OFFSET_DEG[name] = -theta0
        TAUGHT_ZERO_DEG[name] = theta0
    ANGLE_TEACH_STATE = "applied"
    missing = [name for name, sends in SENDS_ANGLE_DEG.items() if sends and name not in shapes]
    text = f"angle teach: sign {sign:+.0f}, taught {', '.join(sorted(shapes))}"
    if missing:
        text += f"; UNTAUGHT (picks refused when sending): {', '.join(missing)}"
    return text


def _weights_or_die(model_path: str | Path) -> Path:
    """Return the weights path, or stop with one clear line instead of a traceback."""
    path = Path(model_path)
    if path.is_file():
        return path
    if path.is_dir():  # pi4: ultralytics also loads an exported *_ncnn_model folder
        return path
    for candidate in (MODEL_PATH,
                      WEIGHTS_DIR / "best_ncnn_model",
                      WEIGHTS_DIR / "best.onnx",
                      REPO_ROOT.parent / "runs" / "obb" / "obb_train" / "weights" / "best.pt"):
        if candidate.exists():
            return candidate
    found = sorted(
        (item for item in REPO_ROOT.glob("runs/obb/**/weights/*")
         if item.suffix in {".pt", ".onnx"} or item.name.endswith("_ncnn_model")),
        key=lambda item: item.stat().st_mtime, reverse=True)
    if found:
        return found[0]
    raise SystemExit(
        f"weights not found: {path}\n"
        f"  also looked at {WEIGHTS_DIR / 'best_ncnn_model'}\n"
        f"  and {WEIGHTS_DIR / 'best.onnx'}\n"
        f"  and {WEIGHTS_DIR / 'best.pt'}\n"
        "  the archive ships best_ncnn_model (ARM) and best.onnx, not best.pt\n"
        "  pass --model runs/obb/obb_train/weights/best_ncnn_model"
    )

@dataclass(frozen=True)
class Detection:

    class_id: int
    confidence: float
    center: tuple[float, float]
    angle_deg: float
    corners: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.class_id not in range(len(CLASSES)):
            raise ValueError(f"class_id {self.class_id} is outside 0..{len(CLASSES) - 1}")

    @property
    def name(self) -> str:
        return CLASSES[self.class_id]

COLOURS = ("red", "blue", "green", "black", "white")
UNKNOWN_COLOUR = "unknown"

_HUE_BANDS: dict[str, tuple[tuple[int, int], ...]] = {
    "red": ((0, 10), (170, 180)),
    "green": ((35, 85),),
    "blue": ((100, 140),),
}
_MIN_SATURATION = 60
_MIN_VALUE = 40
_MAX_VALUE = 215
_MIN_PIXELS = 4

def colour_of(image: np.ndarray, detection: Detection | None = None, pad: int = 2) -> str:
    crop = image if detection is None else _crop_of(image, detection, pad)
    if crop.size == 0:
        return UNKNOWN_COLOUR

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (saturation >= _MIN_SATURATION) & (value >= _MIN_VALUE)
    dark = int((value < _MIN_VALUE).sum())
    pale = int(((saturation < _MIN_SATURATION) & (value >= _MAX_VALUE)).sum())
    if int(lit.sum()) < _MIN_PIXELS and dark < _MIN_PIXELS and pale < _MIN_PIXELS:
        return UNKNOWN_COLOUR

    hues = hue[lit]
    votes = {
        colour: int(sum(int(((hues >= low) & (hues < high)).sum()) for low, high in bands))
        for colour, bands in _HUE_BANDS.items()
    }
    votes["black"] = dark
    votes["white"] = pale
    best = max(votes.values())
    winners = [colour for colour, count in votes.items() if count == best]
    return winners[0] if len(winners) == 1 else UNKNOWN_COLOUR

def _crop_of(image: np.ndarray, detection: Detection, pad: int) -> np.ndarray:
    corners = np.array(detection.corners, dtype=float)
    height, width = image.shape[:2]
    left = max(0, int(corners[:, 0].min()) - pad)
    right = min(width, int(corners[:, 0].max()) + pad + 1)
    top = max(0, int(corners[:, 1].min()) - pad)
    bottom = min(height, int(corners[:, 1].max()) + pad + 1)
    return image[top:bottom, left:right]

CENTRALLY_SYMMETRIC = ("circle", "square", "hexagon")
_RING = 3
_TOLERANCE = 30
_TOLERANCE_LOW = 8.0
_IDEAL_FILL = {"circle": 0.785, "square": 1.0, "hexagon": 0.75,
               "triangle": 0.5, "star": 0.31}
_PAD_FRACTION = 0.4
_MIN_FILL = {"star": 0.10}
_MIN_FILL_DEFAULT = 0.25
_MARGIN = 6

def foreground_mask(crop: np.ndarray, tolerance: float = _TOLERANCE) -> np.ndarray:
    # Take the background colour from the crop border.
    border = np.concatenate([
        crop[:_RING].reshape(-1, 3), crop[-_RING:].reshape(-1, 3),
        crop[:, :_RING].reshape(-1, 3), crop[:, -_RING:].reshape(-1, 3),
    ])
    background = np.median(border, axis=0)
    difference = np.abs(crop.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    return (difference > tolerance).astype(np.uint8)

def outline_heading_deg(contour: Any, centre: tuple[float, float],
                        period_deg: float) -> float | None:
    if period_deg >= 360.0:
        return None
    points = np.asarray(contour, np.float64).reshape(-1, 2)
    if points.size == 0:
        return None
    dx = points[:, 0] - centre[0]
    dy = points[:, 1] - centre[1]
    radius = np.hypot(dx, dy)
    if float(radius.max()) <= 0.0:
        return None
    angles = np.degrees(np.arctan2(dy, dx)) % 360.0
    bins = np.zeros(360)
    np.maximum.at(bins, angles.astype(int) % 360, radius)
    top = float(bins.max())
    peaks = [a for a in range(360)
             if bins[a] >= 0.97 * top and bins[a] >= bins[(a - 1) % 360]
             and bins[a] >= bins[(a + 1) % 360]]
    if not peaks:
        return None
    phases = np.radians([float(a % period_deg) * 360.0 / period_deg for a in peaks])
    length = float(np.hypot(np.cos(phases).sum(), np.sin(phases).sum()))
    if length < 0.5 * len(peaks):
        return None
    mean = math.degrees(math.atan2(float(np.sin(phases).sum()),
                                   float(np.cos(phases).sum()))) % 360.0
    return float(mean * period_deg / 360.0 % period_deg)

_POLAR_BINS = 720
_MIN_POLAR_CONFIDENCE = 0.10
_FRAGMENT_SHARE = 0.15

def outline_phase_deg(mask: np.ndarray, period_deg: float) -> tuple[float | None, float]:
    if period_deg >= 360.0:
        return None, 0.0
    order = 360.0 / period_deg
    if abs(order - round(order)) > 1e-6:
        return None, 0.0
    order = int(round(order))
    points = np.nonzero(mask)
    if points[0].size < 64:
        return None, 0.0
    ys, xs = points[0].astype(np.float64), points[1].astype(np.float64)
    centre_x, centre_y = float(xs.mean()), float(ys.mean())
    reach = float(np.hypot(xs - centre_x, ys - centre_y).max())
    radii = np.arange(0.35 * reach, reach, 1.0)
    if radii.size < 8:
        return None, 0.0
    angles = np.linspace(0.0, 2.0 * math.pi, _POLAR_BINS, endpoint=False)
    sample_x = np.rint(centre_x + np.outer(radii, np.cos(angles))).astype(np.int32)
    sample_y = np.rint(centre_y + np.outer(radii, np.sin(angles))).astype(np.int32)
    inside = ((sample_x >= 0) & (sample_x < mask.shape[1])
              & (sample_y >= 0) & (sample_y < mask.shape[0]))
    hits = np.zeros_like(sample_x, dtype=bool)
    hits[inside] = mask[sample_y[inside], sample_x[inside]] > 0
    profile = hits.sum(axis=0).astype(np.float64)
    if float(profile.max()) <= 0.0:
        return None, 0.0
    # The shape symmetry is the frequency we look for.
    spectrum = np.fft.rfft(profile - profile.mean())
    wanted = spectrum[order]
    rest = float(np.abs(spectrum[1:]).sum()) - float(abs(wanted))
    confidence = float(abs(wanted)) / (rest + 1e-9)
    heading = -math.degrees(float(np.angle(wanted))) / order
    return float(heading % period_deg), confidence

def _clip_to_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    keep = np.zeros_like(mask)
    keep[max(0, top):max(0, bottom), max(0, left):max(0, right)] = \
        mask[max(0, top):max(0, bottom), max(0, left):max(0, right)]
    return keep

def _fragmented(mask: np.ndarray, box_area: float) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [contour for contour in contours
           if float(cv2.contourArea(contour)) >= _FRAGMENT_SHARE * box_area]
    return len(big) >= 2

def _largest_contour(mask: np.ndarray) -> Any:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None

def _annulus_mask(crop: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    left, top, right, bottom = box
    keep = np.ones(crop.shape[:2], bool)
    keep[max(0, top - _MARGIN):bottom + _MARGIN, max(0, left - _MARGIN):right + _MARGIN] = False
    if int(keep.sum()) < 64:
        return None
    background = np.median(crop[keep].reshape(-1, 3), axis=0)
    difference = np.abs(crop.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    return (difference > _TOLERANCE).astype(np.uint8)

def _centred_contour(mask: np.ndarray, point: tuple[float, float]) -> Any:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0:
            return contour
    return None

def _touches_border(contour: Any, height: int, width: int) -> bool:
    points = np.asarray(contour, np.float64).reshape(-1, 2)
    return bool((points[:, 0] <= 0).any() or (points[:, 1] <= 0).any()
                or (points[:, 0] >= width - 1).any() or (points[:, 1] >= height - 1).any())

def refine(image: np.ndarray, detection: Detection) -> tuple[tuple[float, float], float | None]:
    corners = np.asarray(detection.corners, np.float64)
    left, top = float(corners[:, 0].min()), float(corners[:, 1].min())
    right, bottom = float(corners[:, 0].max()), float(corners[:, 1].max())
    pad = max(4, int(_PAD_FRACTION * max(right - left, bottom - top)))
    height, width = image.shape[:2]
    x0, x1 = max(0, int(left) - pad), min(width, int(right) + pad + 1)
    y0, y1 = max(0, int(top) - pad), min(height, int(bottom) + pad + 1)
    crop = image[y0:y1, x0:x1]
    if crop.shape[0] < 2 * _RING + 2 or crop.shape[1] < 2 * _RING + 2:
        return detection.center, None
    box_area = max(1.0, (right - left) * (bottom - top))
    limit = _MIN_FILL.get(detection.name, _MIN_FILL_DEFAULT)

    def usable(contour: Any) -> bool:
        return (contour is not None
                and float(cv2.contourArea(contour)) / box_area >= limit
                and not _touches_border(contour, crop.shape[0], crop.shape[1]))

    box = (int(left) - x0, int(top) - y0, int(right) - x0, int(bottom) - y0)
    mask = _clip_to_box(foreground_mask(crop), box)
    ideal = _IDEAL_FILL.get(detection.name, 0.5)
    if float(mask.sum()) / box_area < 0.6 * ideal:
        loose = _clip_to_box(foreground_mask(crop, _TOLERANCE_LOW), box)
        if int(loose.sum()) > int(mask.sum()):
            mask = loose
    # Use the biggest blob as the shape outline.
    contour = _largest_contour(mask)
    if not usable(contour):
        annulus = _annulus_mask(crop, box)
        mask = None if annulus is None else _clip_to_box(annulus, box)
        contour = None if mask is None else _centred_contour(
            mask, (detection.center[0] - x0, detection.center[1] - y0))
        # If the blob looks broken, retry with the ring around the box.
        if not usable(contour):
            return detection.center, None
    assert contour is not None
    moments = cv2.moments(contour)
    if moments["m00"] <= 0.0:
        return detection.center, None
    inside = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    heading = outline_heading_deg(contour, inside, SYMMETRY_DEG[detection.name])
    if mask is not None and _fragmented(mask, box_area):
        phase, confidence = outline_phase_deg(mask, SYMMETRY_DEG[detection.name])
        if phase is not None and confidence >= _MIN_POLAR_CONFIDENCE:
            whole = cv2.moments(mask, True)
            if whole["m00"] > 0.0:
                inside = (whole["m10"] / whole["m00"], whole["m01"] / whole["m00"])
            return (x0 + inside[0], y0 + inside[1]), phase
    return (x0 + inside[0], y0 + inside[1]), heading

@dataclass(frozen=True)
class PickCommand:

    shape: str
    x_mm: float
    y_mm: float
    angle_deg: float
    z_mm: float = PICK_Z_MM

    def __post_init__(self) -> None:
        if self.shape not in CLASSES:
            raise ValueError(f"shape {self.shape!r} is not one of {list(CLASSES)}")

    @property
    def slot(self) -> int:
        return SLOT[self.shape]

@dataclass(frozen=True)
class Calibration:

    mm_per_px: float
    origin_px: tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0
    flip_y: bool = False

    def __post_init__(self) -> None:
        if self.mm_per_px <= 0.0:
            raise ValueError(f"mm_per_px must be positive, got {self.mm_per_px}")

def canonical_angle(angle_deg: float) -> float:
    # Keep this angle from -45 up to, but below, 135 degrees.
    return (angle_deg + 45.0) % 180.0 - 45.0

class CoordinateMapper:

    def __init__(self, calibration: Calibration) -> None:
        self.calibration = calibration

    def to_mm(self, point_px: tuple[float, float]) -> tuple[float, float]:
        calibration = self.calibration
        # Rotate the pixel offset into the robot frame, then scale to mm.
        theta = math.radians(calibration.rotation_deg)
        dx = point_px[0] - calibration.origin_px[0]
        dy = point_px[1] - calibration.origin_px[1]
        x_mm = calibration.mm_per_px * (dx * math.cos(theta) + dy * math.sin(theta))
        y_mm = calibration.mm_per_px * (-dx * math.sin(theta) + dy * math.cos(theta))
        return (x_mm, -y_mm if calibration.flip_y else y_mm)

    def angle_in_frame(self, angle_deg: float) -> float:
        angle = angle_deg - self.calibration.rotation_deg
        if self.calibration.flip_y:
            # Mirror the angle when the image y axis is flipped.
            angle = -angle
        return canonical_angle(angle)

    def to_pick_command(
        self,
        detection: Detection,
        angle_offset_deg: float = ANGLE_OFFSET_DEG,
        image: np.ndarray | None = None,
    ) -> PickCommand:
        # Use box centres for circle, square and hexagon; refine others.
        if image is None or detection.name in CENTRALLY_SYMMETRIC:
            centre = detection.center
        else:
            centre = refine(image, detection)[0]
        x_mm, y_mm = self.to_mm(centre)
        angle = pick_angle(
            detection.name, self.angle_in_frame(detection.angle_deg), angle_offset_deg)
        return PickCommand(detection.name, x_mm, y_mm, angle)

def check_model_names(names: dict[int, str]) -> None:
    ordered = tuple(names[index] for index in sorted(names))
    if ordered != CLASSES:
        raise ValueError(f"model classes {ordered} != frozen CLASSES {CLASSES}")

def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)

def detections_from_result(result: Any) -> list[Detection]:
    from ultralytics.utils import ops

    obb = result.obb
    if obb is None or len(obb) == 0:
        return []

    corners = _to_numpy(obb.xyxyxyxy).reshape(-1, 4, 2)
    boxes = ops.xyxyxyxy2xywhr(corners)
    confidences = _to_numpy(obb.conf).reshape(-1)
    class_ids = _to_numpy(obb.cls).reshape(-1)

    detections = []
    for index, (cx, cy, _w, _h, theta) in enumerate(boxes):
        detections.append(Detection(
            class_id=int(class_ids[index]),
            confidence=float(confidences[index]),
            center=(float(cx), float(cy)),
            angle_deg=float(np.degrees(theta)),
            corners=tuple((float(x), float(y)) for x, y in corners[index]),
        ))
    return detections

class ShapeDetector:

    def __init__(self, model_path: str | Path = MODEL_PATH, conf: float = 0.25,
                 device: str | int | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(_weights_or_die(model_path)))
        check_model_names(dict(self.model.names))
        self.conf = conf
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        predictions = list(self.model.predict(image, conf=self.conf,
                                                device=self.device,
                                                verbose=False))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        print(f"inference: {elapsed_ms:.1f} ms")
        if not predictions:
            return []
        return detections_from_result(predictions[0])

class FrameSource(ABC):

    @abstractmethod
    def read(self) -> np.ndarray | None:
        ...

    def close(self) -> None:
        return

class ImageFolderSource(FrameSource):

    def __init__(self, folder: str | Path, loop: bool = False,
                 suffixes: frozenset[str] = IMAGE_SUFFIXES) -> None:
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"image folder not found: '{self.folder}'")

        self.paths = sorted(p for p in self.folder.iterdir()
                            if p.suffix.lower() in suffixes)
        if not self.paths:
            raise ValueError(f"no images in '{self.folder}'")

        self.loop = loop
        self._index = 0

    def read(self) -> np.ndarray | None:
        if self._index >= len(self.paths):
            if not self.loop:
                return None
            self._index = 0

        path = self.paths[self._index]
        self._index += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode image: '{path}'")
        return image

class VideoFileSource(FrameSource):

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: '{self.path}'")

        self.loop = loop
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise ValueError(f"cannot open video: '{self.path}'")

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if ok:
            return frame

        if not self.loop:
            return None

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        self._capture.release()

# Pack one pick into the five fields the robot reads.
def encode_command(command: PickCommand) -> str:
    return (
        f"{command.slot};{command.x_mm:.1f};{command.y_mm:.1f};"
        f"{command.z_mm:.1f};{command.angle_deg:.1f}"
    )

def decode_command(line: str) -> PickCommand:
    fields = line.strip().split(";")
    if len(fields) != 5:
        raise ValueError(f"expected 5 ';'-separated fields, got {len(fields)}: {line!r}")

    slot_text, x_mm, y_mm, z_mm, angle_deg = (field.strip() for field in fields)
    try:
        slot = int(slot_text)
        x_value, y_value, z_value, angle_value = (
            float(x_mm), float(y_mm), float(z_mm), float(angle_deg)
        )
    except ValueError as error:
        raise ValueError(f"non-numeric field in {line!r}") from error

    if slot not in SHAPE_BY_SLOT:
        raise ValueError(f"unknown slot {slot}; the tray has {sorted(SHAPE_BY_SLOT)}")
    return PickCommand(SHAPE_BY_SLOT[slot], x_value, y_value, angle_value, z_value)

class TransportError(RuntimeError):
    ...

class Transport(ABC):

    def send_command(self, command: PickCommand) -> None:
        self.send_line(encode_command(command))

    @abstractmethod
    def send_line(self, line: str) -> None:
        ...

    @abstractmethod
    def receive_line(self, timeout_s: float | None = None) -> str | None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

class FakeTransport(Transport):

    def __init__(self, incoming: Iterable[str] = ()) -> None:
        self.sent: list[str] = []
        self._incoming = deque(incoming)
        self._closed = False

    def inject(self, line: str) -> None:
        self._incoming.append(line)

    def sent_commands(self) -> list[PickCommand]:
        return [decode_command(line) for line in self.sent]

    def send_line(self, line: str) -> None:
        if self._closed:
            raise TransportError("transport is closed")
        self.sent.append(line)

    def receive_line(self, timeout_s: float | None = None) -> str | None:
        if self._closed or not self._incoming:
            return None
        return self._incoming.popleft().rstrip("\r\n")

    def close(self) -> None:
        self._closed = True

def bind_error_message(host: str, port: int, error: OSError) -> str:
    """Explain a failed server bind in terms a student can act on."""
    if error.errno == errno.EADDRNOTAVAIL:
        return (f"cannot listen on {host}:{port}: that address is not one of this"
                " machine's addresses; on the Pi run 'ip -4 addr' (or 'hostname -I')"
                " and pass THIS device's address on the robot subnet, never the"
                " robot's own address (ABB default 192.168.125.2)")
    if error.errno == errno.EADDRINUSE:
        return (f"cannot listen on {host}:{port}: another program already listens"
                " on this port, pick --port 5001")
    return f"cannot listen on {host}:{port}: {error}"


class TcpTransport(Transport):

    def __init__(self, host: str = "192.168.125.2", port: int = 5000,
                 server: bool = True) -> None:
        self.host = host
        self.port = port
        self.server = server
        self._listener: socket.socket | None = None
        self._socket: socket.socket | None = None
        self._buffer = ""

    def open(self) -> None:
        if not self.server:
            self._socket = socket.create_connection((self.host, self.port), timeout=10.0)
            return

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.host, self.port))
        except OSError as error:
            raise TransportError(bind_error_message(self.host, self.port, error)) from error
        # Accept a single robot connection only.
        listener.listen(1)
        self._listener = listener
        self.port = listener.getsockname()[1]

    def accept(self, timeout_s: float | None = 30.0) -> None:
        if self._listener is None:
            raise TransportError("open() the server before accept()")
        self._listener.settimeout(timeout_s)
        try:
            self._socket, _address = self._listener.accept()
        except TimeoutError as error:
            raise TransportError(f"no connection within {timeout_s} s") from error
        finally:
            self._listener.close()
            self._listener = None

    def send_line(self, line: str) -> None:
        try:
            self._connection().sendall((line + "\n").encode("utf-8"))
        except OSError as error:
            raise TransportError(f"send failed: {error}") from error

    def receive_line(self, timeout_s: float | None = None) -> str | None:
        connection = self._connection()
        while "\n" not in self._buffer:
            connection.settimeout(timeout_s)
            try:
                chunk = connection.recv(4096)
            except TimeoutError:
                return None
            except OSError as error:
                raise TransportError(f"receive failed: {error}") from error
            if not chunk:
                raise TransportError("the peer closed the connection")
            self._buffer += chunk.decode("utf-8", errors="replace")

        # Return one complete line and keep the rest buffered.
        line, _, self._buffer = self._buffer.partition("\n")
        return line.rstrip("\r")

    def close(self) -> None:
        for sock in (self._socket, self._listener):
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()
        self._socket = None
        self._listener = None
        self._buffer = ""

    def _connection(self) -> socket.socket:
        if self._socket is None:
            raise TransportError("no connection; open() and accept() first")
        return self._socket

class RobotClient:

    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    def wait_for_request(self, timeout_s: float | None = None) -> str | None:
        return self.transport.receive_line(timeout_s)

    def send_pick(self, command: PickCommand) -> None:
        self.transport.send_command(command)

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

class Detector(Protocol):

    def detect(self, image: np.ndarray) -> list[Detection]: ...

class PickCycle:

    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        mapper: CoordinateMapper,
        robot: RobotClient,
        request_timeout_s: float | None = None,
    ) -> None:
        self.source = source
        self.detector = detector
        self.mapper = mapper
        self.robot = robot
        self.request_timeout_s = request_timeout_s

    def pick_next(self) -> PickCommand | None:
        # Wait for the robot to ask, then answer with one pick.
        request = self.robot.wait_for_request(self.request_timeout_s)
        if request is None:
            return None
        frame = self.source.read()
        if frame is None:
            return None
        detections = self.detector.detect(frame)
        if not detections:
            return None
        target = max(detections, key=lambda detection: detection.confidence)
        command = self.mapper.to_pick_command(target)
        self.robot.send_pick(command)
        return command

Row = tuple[Detection, PickCommand, float, float]


def draw_overlay(image: np.ndarray, rows: list[Row]) -> np.ndarray:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for detection, command, _raw, _distance in rows:
        corners = np.asarray(detection.corners, np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [corners], True, (0, 255, 0), 2)
        centre = (int(detection.center[0]), int(detection.center[1]))
        cv2.circle(canvas, centre, 4, (0, 255, 0), -1)
        label = f"{command.shape} {command.x_mm:.0f},{command.y_mm:.0f} {command.angle_deg:.0f}dg"
        cv2.putText(canvas, label, (centre[0] + 6, centre[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def read_image(path: str | None, live_label: str) -> tuple[np.ndarray, str] | None:
    if path:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None if image is None else (image, str(path))
    from live_pick import grab_frame
    return grab_frame(), live_label


def write_overlay(image: np.ndarray, rows: list[Row], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), draw_overlay(image, rows))
    return path
