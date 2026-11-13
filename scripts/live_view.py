from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import calibrate
import pipeline
import template_fit as tf

MAIN_WINDOW = "INA700 - live"
PARTS_WINDOW = "INA700 - parts"
VIEW_WIDTH = 1100
TILE = 260
TILE_TEXT = 78
TILES_PER_ROW = 4
SAVE_DIR = pipeline.REPO_ROOT / "data" / "captured_frames"
MOULD_WINDOW = "INA700 - mould"
MOULD_PATH = pipeline.REPO_ROOT / "data" / "mould.json"
MOULD_DIR = pipeline.REPO_ROOT / "data" / "mould"
MOULD_PX_PER_MM = 3.6
# Pocket angles, in degrees, from the mould teaching.
DEFAULT_POCKETS_DEG = {"square": 3.0, "hexagon": 0.1, "triangle": -30.9, "star": 15.8}
# How many degrees of turn fit one mm of gap.
WINDOW_1MM_DEG = {"square": 3.3, "hexagon": 4.9, "triangle": 2.6, "star": 2.0}

YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
ORANGE = (0, 140, 255)
MAGENTA = (255, 0, 255)


@dataclass
class PartView:
    detection: pipeline.Detection
    x_mm: float | None = None
    y_mm: float | None = None
    current: float | None = None
    template: float | None = None
    sent: float | None = None
    raw: float | None = None          # camera-frame angle behind `sent` (before pick_angle)
    used: str = "none"
    fit: tf.Fit | None = None
    contour: np.ndarray | None = None
    candidates: int = 0


@dataclass
class Result:
    frame: np.ndarray
    plane: calibrate.PlaneCalibration | None
    plane_source: str
    parts: list[PartView] = field(default_factory=list)
    detect_ms: float = 0.0
    fit_ms: float = 0.0
    finished: float = 0.0


def analyse_part(frame: np.ndarray, plane: calibrate.PlaneCalibration,
                 detection: pipeline.Detection) -> PartView:
    view = PartView(detection)
    shape = detection.name
    # Measure x and y without the template fit.
    tf.ENABLED = False
    view.x_mm, view.y_mm, raw = plane.parts_mm(detection, frame)
    tf.ENABLED = True
    # Calculate the pick angle from the measured heading.
    view.current = pipeline.pick_angle(shape, raw)
    view.sent, view.raw, view.used = view.current, raw, "current"
    if not pipeline.SENDS_ANGLE_DEG[shape]:
        view.used = "no angle"
        return view
    fit, contour, count = tf.fit_detection_detail(frame, detection, plane)
    view.fit, view.contour, view.candidates = fit, contour, count
    if fit is not None:
        view.template = pipeline.pick_angle(shape, fit.angle_deg)
        # Use the fitted angle only when its quality check passes.
        if tf.usable(fit, shape):
            view.sent, view.raw, view.used = view.template, fit.angle_deg, "template"
            if shape not in pipeline.CENTRALLY_SYMMETRIC:
                view.x_mm, view.y_mm = fit.centre_mm
    return view


# One full check: marker, then parts, then fit.
def analyse(frame: np.ndarray, detector: pipeline.Detector, calibration_path: str,
            report=lambda step: None) -> Result:
    report("1/4  finding the marker")
    try:
        plane = calibrate.marker_scale(frame)
        source = "marker in this frame"
    except calibrate.MarkerNotFound:
        try:
            plane = calibrate.load_calibration(calibration_path)
            source = f"saved file (no marker in view): {calibration_path}"
        except (FileNotFoundError, ValueError):
            plane, source = None, "NO MARKER and no calibration file - mm unavailable"
    result = Result(frame, plane, source)

    report("2/4  detecting parts (YOLO)")
    started = time.perf_counter()
    detections = detector.detect(frame)
    result.detect_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    for index, detection in enumerate(detections, start=1):
        report(f"3/4  outline + template fit, part {index}/{len(detections)} ({detection.name})")
        if plane is None:
            result.parts.append(PartView(detection))
        else:
            result.parts.append(analyse_part(frame, plane, detection))
    result.fit_ms = (time.perf_counter() - started) * 1000.0
    result.finished = time.monotonic()
    report(f"4/4  done: {len(detections)} part(s) - waiting for the next frame")
    return result


class Worker(threading.Thread):

    def __init__(self, detector: pipeline.Detector, calibration_path: str) -> None:
        super().__init__(name="analyse", daemon=True)
        self.detector = detector
        self.calibration_path = calibration_path
        self.step = "starting - loading the model"
        self.result: Result | None = None
        self.error: Exception | None = None
        self.busy = False
        self._pending: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def offer(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame

    def stop(self) -> None:
        self._stop.set()

    def _report(self, step: str) -> None:
        self.step = step

    def run(self) -> None:
        self.step = "waiting for a frame"
        while not self._stop.is_set():
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None:
                self._stop.wait(0.02)
                continue
            self.busy = True
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.result = analyse(frame, self.detector, self.calibration_path, self._report)
            except Exception as error:
                self.error = error
                self.step = f"error: {error}"
            finally:
                self.busy = False


def _to_px(plane: calibrate.PlaneCalibration, points_mm: np.ndarray) -> np.ndarray:
    # Millimetres to pixels uses the inverse calibration.
    inverse = np.linalg.inv(plane.matrix_3x3)
    return cv2.perspectiveTransform(np.asarray(points_mm, np.float64).reshape(-1, 1, 2),
                                    inverse).reshape(-1, 2)


def _poly(canvas: np.ndarray, points: np.ndarray, colour, thickness: int) -> None:
    cv2.polylines(canvas, [np.round(points).astype(np.int32).reshape(-1, 1, 2)],
                  True, colour, thickness, cv2.LINE_AA)


def _template_px(plane: calibrate.PlaneCalibration, part: PartView) -> np.ndarray | None:
    if part.fit is None:
        return None
    scale = min(max(part.fit.size_ratio, tf.SIZE_LIMITS[0]), tf.SIZE_LIMITS[1])
    outline = tf.template_mm(part.detection.name, part.fit.angle_deg, scale, part.fit.centre_mm)
    return _to_px(plane, outline)


def draw_main(frame: np.ndarray, result: Result | None) -> np.ndarray:
    canvas = frame.copy()
    if result is None:
        return canvas
    plane = result.plane
    if plane is not None:
        marker = _to_px(plane, calibrate.marker_mm_corners(plane.marker_mm))
        _poly(canvas, marker, MAGENTA, 4)
        # Draw the sheet x and y directions on the image.
        origin = np.array([[0.0, 0.0], [60.0, 0.0], [0.0, 60.0]])
        o, px, py = _to_px(plane, origin)
        cv2.arrowedLine(canvas, tuple(map(int, o)), tuple(map(int, px)), RED, 5, tipLength=0.2)
        cv2.arrowedLine(canvas, tuple(map(int, o)), tuple(map(int, py)), GREEN, 5, tipLength=0.2)
        cv2.putText(canvas, "+x", (int(px[0]) + 10, int(px[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.6, RED, 4)
        cv2.putText(canvas, "+y", (int(py[0]) + 10, int(py[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.6, GREEN, 4)
    for part in result.parts:
        detection = part.detection
        _poly(canvas, np.asarray(detection.corners), GREY, 2)
        if part.contour is not None:
            _poly(canvas, part.contour, ORANGE, 2)
        template = _template_px(plane, part) if plane is not None else None
        if template is not None:
            _poly(canvas, template, YELLOW if part.used == "template" else RED, 4)
        cx, cy = map(int, detection.center)
        cv2.circle(canvas, (cx, cy), 8, YELLOW, -1)
        label = f"{detection.name}"
        if part.sent is not None:
            label += f" {part.sent:+.1f}"
        lines = [label]
        if part.x_mm is not None:
            lines.append(f"({part.x_mm:.0f}, {part.y_mm:.0f}) mm")
        for row, text in enumerate(lines):
            position = (cx + 14, cy - 14 + row * 44)
            cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 9)
            cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.4, YELLOW, 3)
    return canvas


def add_header(view: np.ndarray, lines: list[str]) -> np.ndarray:
    height = 26 * len(lines) + 12
    header = np.zeros((height, view.shape[1], 3), np.uint8)
    for row, text in enumerate(lines):
        cv2.putText(header, text, (10, 24 + row * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    WHITE if row else YELLOW, 1, cv2.LINE_AA)
    return np.vstack([header, view])


def draw_parts(result: Result | None) -> np.ndarray:
    if result is None or not result.parts:
        empty = np.zeros((TILE + TILE_TEXT, TILE * 2, 3), np.uint8)
        cv2.putText(empty, "no parts yet", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        return empty
    tiles = []
    for part in result.parts:
        detection = part.detection
        corners = np.asarray(detection.corners)
        left, top = corners.min(axis=0)
        right, bottom = corners.max(axis=0)
        pad = 0.35 * max(right - left, bottom - top)
        x0, y0 = int(max(0, left - pad)), int(max(0, top - pad))
        x1 = int(min(result.frame.shape[1], right + pad))
        y1 = int(min(result.frame.shape[0], bottom + pad))
        crop = cv2.convertScaleAbs(result.frame[y0:y1, x0:x1], alpha=1.6, beta=10)
        shift = np.array([x0, y0], np.float64)
        _poly(crop, corners - shift, GREY, 1)
        if part.contour is not None:
            _poly(crop, part.contour - shift, ORANGE, 2)
        template = _template_px(result.plane, part) if result.plane is not None else None
        if template is not None:
            _poly(crop, template - shift, YELLOW if part.used == "template" else RED, 2)
        side = max(crop.shape[:2])
        square = np.zeros((side, side, 3), np.uint8)
        square[:crop.shape[0], :crop.shape[1]] = crop
        tile = cv2.resize(square, (TILE, TILE))
        text = np.zeros((TILE_TEXT, TILE, 3), np.uint8)
        current = "-" if part.current is None else f"{part.current:+.1f}"
        fitted = "-" if part.template is None else f"{part.template:+.1f}"
        score = "" if part.fit is None else f"  score {part.fit.score:.2f}  x{part.fit.size_ratio:.2f}"
        lines = [
            f"{detection.name}  conf {detection.confidence:.2f}",
            f"current {current}  template {fitted}",
            f"sent: {part.used}{score}",
        ]
        for row, line in enumerate(lines):
            colour = YELLOW if row == 0 else (GREEN if (row == 2 and part.used == "template") else WHITE)
            cv2.putText(text, line, (6, 22 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        tiles.append(np.vstack([tile, text]))
    while len(tiles) % TILES_PER_ROW and len(tiles) > TILES_PER_ROW:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + TILES_PER_ROW]) for i in range(0, len(tiles), TILES_PER_ROW)]
    return np.vstack(rows)


def profile_mm(shape: str) -> float:
    if shape == "square":
        return tf.SQUARE_SIDE_MM
    if shape == "hexagon":
        return tf.HEXAGON_FLATS_MM
    if shape == "triangle":
        return tf.TRIANGLE_SIDE_MM / math.sqrt(3.0)
    return 2.0 * tf.STAR_NOTCH_CHORD_MM / (2.0 * math.sin(math.pi / 5))


def load_mould() -> tuple[dict[str, float], str]:
    try:
        values = json.loads(MOULD_PATH.read_text(encoding="utf-8"))
        pockets = {shape: float(values["pockets_deg"][shape]) for shape in DEFAULT_POCKETS_DEG
                   if shape in values.get("pockets_deg", {})}
        merged = {**DEFAULT_POCKETS_DEG, **pockets}
        return merged, f"taught {values.get('taught', '?')} ({MOULD_PATH.name})"
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return dict(DEFAULT_POCKETS_DEG), "defaults from the 2026-10-30 mould frame (press M to teach)"


# Press M: save pocket angles and part photos.
def teach_mould(result: Result | None) -> list[str]:
    if result is None or result.plane is None:
        return []
    pockets, _source = load_mould()
    taught = []
    MOULD_DIR.mkdir(parents=True, exist_ok=True)
    for part in result.parts:
        shape = part.detection.name
        if shape not in DEFAULT_POCKETS_DEG or part.fit is None or part.used != "template":
            continue
        pockets[shape] = pipeline.fold_to_symmetry(shape, part.fit.angle_deg)
        cv2.imwrite(str(MOULD_DIR / f"{shape}.png"), _part_crop(result, part))
        taught.append(shape)
    if taught:
        MOULD_PATH.parent.mkdir(parents=True, exist_ok=True)
        MOULD_PATH.write_text(json.dumps({
            "taught": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "note": "pocket angles in the camera/wobjS frame, measured with the parts seated",
            "pockets_deg": pockets,
        }, indent=2) + "\n", encoding="utf-8")
    return taught


def _part_crop(result: Result, part: PartView) -> np.ndarray:
    corners = np.asarray(part.detection.corners)
    left, top = corners.min(axis=0)
    right, bottom = corners.max(axis=0)
    pad = 0.35 * max(right - left, bottom - top)
    x0, y0 = int(max(0, left - pad)), int(max(0, top - pad))
    x1 = int(min(result.frame.shape[1], right + pad))
    y1 = int(min(result.frame.shape[0], bottom + pad))
    crop = cv2.convertScaleAbs(result.frame[y0:y1, x0:x1], alpha=1.6, beta=10)
    side = max(crop.shape[:2])
    square = np.zeros((side, side, 3), np.uint8)
    square[:crop.shape[0], :crop.shape[1]] = crop
    return cv2.resize(square, (TILE, TILE))


def _mm_to_tile(points_mm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_mm, np.float64)
    return np.stack([TILE / 2 + points[:, 0] * MOULD_PX_PER_MM,
                     TILE / 2 - points[:, 1] * MOULD_PX_PER_MM], axis=1)


def _mould_photo(shape: str) -> np.ndarray:
    photo_path = MOULD_DIR / f"{shape}.png"
    photo = cv2.imread(str(photo_path)) if photo_path.is_file() else None
    if photo is None or photo.shape[:2] != (TILE, TILE):
        photo = np.zeros((TILE, TILE, 3), np.uint8)
        cv2.putText(photo, "no mould photo", (40, TILE // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1)
        cv2.putText(photo, "press M with the", (34, TILE // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)
        cv2.putText(photo, "mould in view", (50, TILE // 2 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)
    cv2.putText(photo, "mould", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
    return photo


def _taught_tile(part: PartView, zero_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Robot angle teach exists: the pocket comparison is replaced by what the
    # robot will really do. zero = camera angle of a part that needs no turn
    # (theta0); part.sent = pick_angle = the nAng sent; the green outline is
    # where that turn takes the part (lands on the zero pose mod the symmetry).
    shape = part.detection.name
    part_deg = float(part.raw)
    after = part_deg - pipeline.ANGLE_SIGN * float(part.sent)
    sketch = np.full((TILE, TILE, 3), 30, np.uint8)
    _poly(sketch, _mm_to_tile(tf.template_mm(shape, zero_deg)), WHITE, 2)
    _poly(sketch, _mm_to_tile(tf.template_mm(shape, part_deg)), RED, 2)
    _poly(sketch, _mm_to_tile(tf.template_mm(shape, after)), GREEN, 1)
    cv2.putText(sketch, "white=taught zero red=part green=after turn", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, WHITE, 1, cv2.LINE_AA)
    text = np.zeros((TILE_TEXT, 2 * TILE, 3), np.uint8)
    lines = [
        (f"{shape}: part {part_deg:+.1f}   taught zero theta0 {zero_deg:+.1f}   ({part.used})", YELLOW),
        (f"robot turns nAng {part.sent:+.1f} deg (sent value, sign {pipeline.ANGLE_SIGN:+.0f})", GREEN),
        ("robot angle teach in use - pocket check not needed", WHITE),
    ]
    for row, (line, colour) in enumerate(lines):
        cv2.putText(text, line, (6, 22 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.47, colour, 1, cv2.LINE_AA)
    return _mould_photo(shape), sketch, text


def draw_mould(result: Result | None, clearance_mm: float) -> np.ndarray:
    pockets, source = load_mould()
    header = np.zeros((34, 2 * TILE, 3), np.uint8)
    cv2.putText(header, f"pockets: {source}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
    parts = [] if result is None else [p for p in result.parts
                                       if p.detection.name in DEFAULT_POCKETS_DEG and p.sent is not None]
    if not parts:
        empty = np.zeros((TILE, 2 * TILE, 3), np.uint8)
        cv2.putText(empty, "no part with an angle yet", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        return np.vstack([header, empty])
    rows = []
    for part in parts:
        shape = part.detection.name
        taught_zero = pipeline.TAUGHT_ZERO_DEG.get(shape)
        if taught_zero is not None and part.raw is not None:
            photo, sketch, text = _taught_tile(part, taught_zero)
            rows.append(np.vstack([np.hstack([photo, sketch]), text]))
            continue
        part_deg = part.template if part.used == "template" else part.current
        pocket_deg = pockets[shape]
        turn = pipeline.fold_to_symmetry(shape, pocket_deg - part_deg)
        # More clearance means a wider accepted turn.
        window = WINDOW_1MM_DEG[shape] * clearance_mm

        photo = _mould_photo(shape)

        sketch = np.full((TILE, TILE, 3), 30, np.uint8)
        grow = 1.0 + 2.0 * clearance_mm / profile_mm(shape)
        pocket = _mm_to_tile(tf.template_mm(shape, pocket_deg, grow))
        cv2.fillPoly(sketch, [np.round(pocket).astype(np.int32)], (70, 70, 70), cv2.LINE_AA)
        _poly(sketch, pocket, WHITE, 2)
        _poly(sketch, _mm_to_tile(tf.template_mm(shape, part_deg)), RED, 2)
        _poly(sketch, _mm_to_tile(tf.template_mm(shape, part_deg + turn)), GREEN, 1)
        tip = _mm_to_tile(np.array([[0.0, 0.0], [32.0 * math.cos(math.radians(part_deg)),
                                                    32.0 * math.sin(math.radians(part_deg))]]))
        cv2.arrowedLine(sketch, tuple(map(int, tip[0])), tuple(map(int, tip[1])), RED, 1, tipLength=0.2)
        cv2.putText(sketch, "white=pocket red=part green=turned", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)

        fits = abs(turn) <= window
        text = np.zeros((TILE_TEXT, 2 * TILE, 3), np.uint8)
        lines = [
            (f"{shape}: part {part_deg:+.1f}   pocket {pocket_deg:+.1f}   ({part.used})", YELLOW),
            (f"turn to match pocket: {turn:+.1f} deg   pocket accepts +/-{window:.1f} deg "
             f"at {clearance_mm:.1f} mm clearance", WHITE),
            ("already within the pocket window" if fits else
             f"needs {abs(turn):.1f} deg of turn before it drops in", GREEN if fits else ORANGE),
        ]
        for row, (line, colour) in enumerate(lines):
            cv2.putText(text, line, (6, 22 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.47, colour, 1, cv2.LINE_AA)
        rows.append(np.vstack([np.hstack([photo, sketch]), text]))
    return np.vstack([header, *rows])


ANGLE_TEACH = "angle teach: not loaded"


def status_lines(worker_step: str, result: Result | None, fps: float, paused: bool,
                 manual: bool) -> list[str]:
    lines = [f"step: {worker_step}" + ("   [PAUSED]" if paused else "")]
    if result is None:
        lines.append("no result yet")
    else:
        age = time.monotonic() - result.finished
        lines.append(f"calibration: {result.plane_source}   {ANGLE_TEACH}")
        lines.append(f"detect {result.detect_ms:.0f} ms | outline+fit {result.fit_ms:.0f} ms | "
                     f"result is {age:.1f} s old | camera {fps:.1f} fps")
    keys = "SPACE analyse | " if manual else ""
    lines.append(keys + "P pause | S save | M teach mould | Q quit   grey=YOLO box  orange=outline  "
                 "yellow=template used  red=fell back")
    return lines


def save(frame: np.ndarray, annotated: np.ndarray, parts: np.ndarray) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{datetime.now():%Y%m%d_%H%M%S_%f}"
    cv2.imwrite(str(SAVE_DIR / f"{stem}.png"), frame)
    cv2.imwrite(str(SAVE_DIR / f"{stem}_live.png"), annotated)
    cv2.imwrite(str(SAVE_DIR / f"{stem}_parts.png"), parts)
    return SAVE_DIR / f"{stem}.png"


def render(frame: np.ndarray, result: Result | None, lines: list[str],
           clearance_mm: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    annotated = draw_main(frame, result)
    scale = VIEW_WIDTH / annotated.shape[1]
    view = cv2.resize(annotated, (VIEW_WIDTH, int(annotated.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)
    return add_header(view, lines), draw_parts(result), draw_mould(result, clearance_mm)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="live camera view with every step drawn on it")
    parser.add_argument("--image", help="show a saved frame instead of the Basler")
    parser.add_argument("--calibration", default=str(calibrate.CALIBRATION_PATH),
                        help="used only when the marker is not in view")
    parser.add_argument("--model", default=pipeline.MODEL_PATH, help="OBB weights")
    parser.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    parser.add_argument("--manual", action="store_true",
                        help="analyse only when SPACE is pressed (good on a slow CPU)")
    parser.add_argument("--clearance", type=float, default=1.0,
                        help="gap between part and pocket wall in mm (sets the accepted window)")
    parser.add_argument("--once", metavar="FOLDER",
                        help="no windows: analyse --image once, write the two pictures here, exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    global ANGLE_TEACH
    ANGLE_TEACH = pipeline.apply_angle_teach()
    print(ANGLE_TEACH)
    detector = pipeline.ShapeDetector(model_path=arguments.model, conf=arguments.conf)

    if arguments.once:
        if not arguments.image:
            print("--once needs --image", file=sys.stderr)
            return 2
        frame = cv2.imread(arguments.image, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"cannot read {arguments.image}", file=sys.stderr)
            return 2
        result = analyse(frame, detector, arguments.calibration, print)
        for part in result.parts:
            print(f"  {part.detection.name:9s} current {part.current}  template {part.template}  "
                  f"sent {part.sent}  ({part.used})")
        main_view, parts_view, mould_view = render(
            frame, result, status_lines("done", result, 0.0, False, False), arguments.clearance)
        folder = Path(arguments.once)
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "live.png"), main_view)
        cv2.imwrite(str(folder / "parts.png"), parts_view)
        cv2.imwrite(str(folder / "mould.png"), mould_view)
        print(f"wrote live.png, parts.png and mould.png in {folder}")
        return 0

    source = None
    still = None
    if arguments.image:
        still = cv2.imread(arguments.image, cv2.IMREAD_COLOR)
        if still is None:
            print(f"cannot read {arguments.image}", file=sys.stderr)
            return 2
    else:
        from camera import BaslerSource
        source = BaslerSource()
        source.open()

    # Analysis runs in a thread so pictures stay smooth.
    worker = Worker(detector, arguments.calibration)
    worker.start()
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PARTS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MOULD_WINDOW, cv2.WINDOW_NORMAL)
    print("Click a window first. SPACE analyse (with --manual), P pause, S save, "
          "M teach mould, Q/Esc quit.")

    paused = False
    frame = still
    fps, counted, started = 0.0, 0, time.perf_counter()
    if still is not None:
        worker.offer(still)
    try:
        while True:
            if source is not None and not paused:
                frame = source.read()
                if frame is None:
                    break
                counted += 1
                elapsed = time.perf_counter() - started
                if elapsed >= 1.0:
                    fps, counted, started = counted / elapsed, 0, time.perf_counter()
                if not arguments.manual and not worker.busy:
                    worker.offer(frame)
            result = worker.result
            lines = status_lines(worker.step, result, fps, paused, arguments.manual)
            main_view, parts_view, mould_view = render(frame, result, lines, arguments.clearance)
            cv2.imshow(MAIN_WINDOW, main_view)
            cv2.imshow(PARTS_WINDOW, parts_view)
            cv2.imshow(MOULD_WINDOW, mould_view)

            key = cv2.waitKey(1 if source is not None else 30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("p"), ord("P")):
                paused = not paused
            if key == ord(" ") and frame is not None:
                worker.offer(frame)
            if key in (ord("m"), ord("M")):
                taught = teach_mould(result)
                print("mould taught for: " + (", ".join(taught) if taught else
                      "nothing - seat the parts in the mould and wait for yellow outlines"))
            if key in (ord("s"), ord("S")) and frame is not None:
                print("saved", save(frame, draw_main(frame, result), parts_view))
            if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                   for name in (MAIN_WINDOW, PARTS_WINDOW, MOULD_WINDOW)):
                break
    finally:
        # Always stop the thread and camera on exit.
        worker.stop()
        if source is not None:
            source.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("stopped")
