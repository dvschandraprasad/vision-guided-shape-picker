from __future__ import annotations

import pipeline

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Real printed marker width in mm; the scale comes from it.
MARKER_MM = 100.0

MARKER_ID = 0
MARKER_DICT = cv2.aruco.DICT_5X5_100

CALIBRATION_PATH = pipeline.REPO_ROOT / "data" / "calibration.json"

SYMMETRY_DEG = pipeline.SYMMETRY_DEG
SENDS_ANGLE_DEG = pipeline.SENDS_ANGLE_DEG
ANGLE_OFFSET_DEG = pipeline.ANGLE_OFFSET_DEG

# How many camera frames we try before giving up.
LIVE_ATTEMPTS = 10

class MarkerNotFound(RuntimeError):
    ...

def make_detector() -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    # Sharper corners make the calibration steadier.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.polygonalApproxAccuracyRate = 0.05
    dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
    return cv2.aruco.ArucoDetector(dictionary, parameters)

def _grey(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def find_marker(
    image: np.ndarray,
    marker_id: int = MARKER_ID,
    detector: cv2.aruco.ArucoDetector | None = None,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise MarkerNotFound("empty frame; nothing to calibrate from")

    detector = detector or make_detector()
    grey = _grey(image)
    corners, ids, _rejected = detector.detectMarkers(grey)
    if ids is None or len(ids) == 0:
        # Second try with stronger contrast for dim frames.
        boosted = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(grey)
        corners, ids, _rejected = detector.detectMarkers(boosted)
    if ids is None or len(ids) == 0:
        raise MarkerNotFound(
            "no ArUco marker in this frame; check the DICT_5X5_100 print, focus and glare"
        )

    found = [int(value) for value in np.asarray(ids).reshape(-1)]
    for index, value in enumerate(found):
        if value == marker_id:
            return np.asarray(corners[index], np.float32).reshape(4, 2)

    raise MarkerNotFound(f"marker id {marker_id} not in this frame; saw {sorted(found)}")

def long_edge_angle(points: np.ndarray) -> float:
    corners = np.asarray(points, np.float32).reshape(-1, 2)
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    # The longest side gives the part's angle.
    start, end = max(edges, key=lambda edge: float(np.linalg.norm(edge[1] - edge[0])))
    delta = end - start
    return (math.degrees(math.atan2(float(delta[1]), float(delta[0]))) + 45.0) % 90.0 - 45.0

@dataclass(frozen=True)
class PlaneCalibration:

    matrix: tuple[float, ...]
    marker_mm: float
    centre_px: tuple[float, float]
    rotation_deg: float
    sides_px: tuple[float, float, float, float]
    frame_turns: int = 0

    # Reject impossible numbers so a bad file fails early.
    def __post_init__(self) -> None:
        if len(self.matrix) != 9:
            raise ValueError(f"a homography is 9 numbers, got {len(self.matrix)}")
        if not all(math.isfinite(value) for value in self.matrix):
            raise ValueError("homography contains a non-finite number")
        if self.matrix[8] == 0.0:
            raise ValueError("homography is singular at h33=0")
        if self.marker_mm <= 0.0:
            raise ValueError(f"marker_mm must be positive, got {self.marker_mm}")
        if len(self.sides_px) != 4 or min(self.sides_px) <= 0.0:
            raise ValueError(f"need four positive sides, got {self.sides_px}")
        if self.frame_turns not in (0, 1, 2, 3):
            raise ValueError(f"frame_turns must be 0-3, got {self.frame_turns}")

    @property
    def matrix_3x3(self) -> np.ndarray:
        return np.asarray(self.matrix, np.float64).reshape(3, 3)

    # Convert one pixel point into millimetres on the sheet.
    def to_mm(self, point_px: tuple[float, float]) -> tuple[float, float]:
        point = np.asarray([[point_px]], np.float64)
        mapped = cv2.perspectiveTransform(point, self.matrix_3x3)[0, 0]
        return (float(mapped[0]), float(mapped[1]))

    def corners_mm(self, corners_px: Any) -> np.ndarray:
        points = np.asarray(corners_px, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.matrix_3x3).reshape(-1, 2)

    def angle_deg(self, corners_px: Any) -> float:
        return long_edge_angle(self.corners_mm(corners_px))

    def direction_deg(self, point_px: tuple[float, float], heading_deg: float,
                      radius_px: float = 40.0) -> float:
        radians = math.radians(heading_deg)
        near = self.to_mm(point_px)
        far = self.to_mm((point_px[0] + radius_px * math.cos(radians),
                          point_px[1] + radius_px * math.sin(radians)))
        return math.degrees(math.atan2(far[1] - near[1], far[0] - near[0])) % 360.0

    def parts_mm(self, detection: pipeline.Detection,
                 image: np.ndarray | None = None) -> tuple[float, float, float]:
        return self.parts_mm_detail(detection, image)[:3]

    def parts_mm_detail(self, detection: pipeline.Detection,
                        image: np.ndarray | None = None) -> tuple[float, float, float, bool]:
        """parts_mm plus whether the angle came from the template fit (True)
        or from the outline/box fallback (False)."""
        # Use the configured angle method for this shape.
        source = pipeline.ANGLE_SOURCE[detection.name]
        if source is not None and image is not None:
            import template_fit
            if template_fit.ENABLED and detection.name in template_fit.TEMPLATES:
                fit = template_fit.fit_detection(image, detection, self)
                if template_fit.usable(fit, detection.name):
                    if detection.name in pipeline.CENTRALLY_SYMMETRIC:
                        x_mm, y_mm = self.to_mm(detection.center)
                    else:
                        x_mm, y_mm = fit.centre_mm
                    return x_mm, y_mm, fit.angle_deg, True
        by_box = detection.name in pipeline.CENTRALLY_SYMMETRIC
        centre, heading = detection.center, None
        if image is not None and (source == "outline" or not by_box):
            measured, heading = pipeline.refine(image, detection)
            centre = detection.center if by_box else measured
        x_mm, y_mm = self.to_mm(centre)
        if source == "outline" and heading is not None:
            return x_mm, y_mm, self.direction_deg(centre, heading), False
        return x_mm, y_mm, self.angle_deg(detection.corners), False

    def to_pick_command(
        self,
        detection: pipeline.Detection,
        angle_offset_deg: float = pipeline.ANGLE_OFFSET_DEG,
        image: np.ndarray | None = None,
    ) -> pipeline.PickCommand:
        x_mm, y_mm, raw = self.parts_mm(detection, image)
        return pipeline.PickCommand(
            detection.name, x_mm, y_mm, pick_angle(detection.name, raw, angle_offset_deg))

    def mm_per_px(self) -> float:
        return self.marker_mm / float(np.mean(self.sides_px))

    def as_scalar_calibration(self) -> pipeline.Calibration:
        return pipeline.Calibration(
            mm_per_px=self.mm_per_px(),
            origin_px=self.centre_px,
            rotation_deg=-self.rotation_deg,
            flip_y=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "homography",
            "marker_mm": self.marker_mm,
            "matrix": list(self.matrix),
            "centre_px": list(self.centre_px),
            "rotation_deg": self.rotation_deg,
            "sides_px": list(self.sides_px),
            "frame_upright": True,
            "frame_turns": self.frame_turns,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PlaneCalibration:
        if not isinstance(values, dict):
            raise ValueError("calibration JSON must be an object")
        if values.get("method", "homography") != "homography":
            raise ValueError(f"unknown calibration method {values.get('method')!r}")
        if values.get("frame_upright") is not True:
            raise ValueError(
                "calibration JSON has no \"frame_upright\": true; it may be a quarter "
                "turn out. Re-run calibrate.py on a frame with the marker visible."
            )
        try:
            return cls(
                matrix=tuple(float(value) for value in values["matrix"]),
                marker_mm=float(values["marker_mm"]),
                centre_px=(float(values["centre_px"][0]), float(values["centre_px"][1])),
                rotation_deg=float(values["rotation_deg"]),
                frame_turns=int(values.get("frame_turns", 0)),
                sides_px=(
                    float(values["sides_px"][0]),
                    float(values["sides_px"][1]),
                    float(values["sides_px"][2]),
                    float(values["sides_px"][3]),
                ),
            )
        except KeyError as error:
            raise ValueError(f"calibration JSON is missing {error.args[0]!r}") from error
        except (TypeError, IndexError) as error:
            raise ValueError(f"calibration JSON is malformed: {error}") from error

def marker_mm_corners(marker_mm: float = MARKER_MM) -> np.ndarray:
    half = float(marker_mm) / 2.0
    return np.array(
        [[-half, half], [half, half], [half, -half], [-half, -half]], np.float32
    )

def upright_turns(corners_px: Any, marker_mm: float = MARKER_MM) -> int:
    corners = np.asarray(corners_px, np.float64).reshape(4, 2)
    best_turns, best_score = 0, -math.inf
    for turns in range(4):
        plane = calibration_from_corners(np.roll(corners, turns, axis=0),
                                         marker_mm=marker_mm, upright=False)
        centre_x, centre_y = plane.centre_px
        right = plane.to_mm((centre_x + 100.0, centre_y))
        down = plane.to_mm((centre_x, centre_y + 100.0))
        # Best turn keeps +x to the right and +y up.
        score = right[0] - down[1]
        if score > best_score:
            best_turns, best_score = turns, score
    return best_turns

def upright_corners(corners_px: Any, marker_mm: float = MARKER_MM) -> np.ndarray:
    corners = np.asarray(corners_px, np.float64).reshape(4, 2)
    return np.roll(corners, upright_turns(corners, marker_mm), axis=0)

def calibration_from_corners(
    corners_px: Any, marker_mm: float = MARKER_MM, upright: bool = False
) -> PlaneCalibration:
    if marker_mm <= 0.0:
        raise ValueError(f"marker_mm must be positive, got {marker_mm}")

    corners = np.asarray(corners_px, np.float64).reshape(4, 2)
    turns = upright_turns(corners, marker_mm) if upright else 0
    if turns:
        corners = np.roll(corners, turns, axis=0)
    sides = (
        float(np.linalg.norm(corners[1] - corners[0])),
        float(np.linalg.norm(corners[2] - corners[1])),
        float(np.linalg.norm(corners[3] - corners[2])),
        float(np.linalg.norm(corners[0] - corners[3])),
    )
    if min(sides) <= 0.0:
        raise ValueError(f"degenerate marker corners, sides {sides}")

    # Map the four marker corners onto a millimetre square.
    matrix = cv2.getPerspectiveTransform(
        corners.astype(np.float32), marker_mm_corners(marker_mm)
    )
    return PlaneCalibration(
        matrix=tuple(float(value) for value in np.asarray(matrix).reshape(-1)),
        marker_mm=float(marker_mm),
        centre_px=(float(corners[:, 0].mean()), float(corners[:, 1].mean())),
        rotation_deg=long_edge_angle(corners),
        sides_px=sides,
        frame_turns=turns,
    )

def marker_scale(
    image: np.ndarray,
    marker_mm: float = MARKER_MM,
    marker_id: int = MARKER_ID,
    detector: cv2.aruco.ArucoDetector | None = None,
    upright: bool = True,
) -> PlaneCalibration:
    return calibration_from_corners(
        find_marker(image, marker_id=marker_id, detector=detector),
        marker_mm=marker_mm,
        upright=upright,
    )

def fold_to_symmetry(shape: str, angle_deg: float) -> float:
    return pipeline.fold_to_symmetry(shape, angle_deg)

def pick_angle(shape: str, angle_deg: float,
               offset_deg: float = pipeline.ANGLE_OFFSET_DEG) -> float:
    return pipeline.pick_angle(shape, angle_deg, offset_deg)

def save_calibration(calibration: PlaneCalibration, path: Path | str = CALIBRATION_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target

def load_calibration(path: Path | str = CALIBRATION_PATH) -> PlaneCalibration:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"no calibration at {source}; run calibrate.py on a frame")
    try:
        values = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} is not valid JSON: {error}") from error
    return PlaneCalibration.from_dict(values)

def _live_frame() -> np.ndarray:
    from pypylon import pylon

    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("no Basler camera detected")
    camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    # Open the camera before measuring the calibration sheet.
    camera.Open()
    try:
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(f"frame acquisition failed: {result.GetErrorDescription()}")
            return converter.Convert(result).GetArray().copy()
        finally:
            result.Release()
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()

def _live_frame_with_marker(marker_id: int, attempts: int = LIVE_ATTEMPTS) -> np.ndarray | None:
    image = None
    for attempt in range(1, attempts + 1):
        image = _live_frame()
        try:
            find_marker(image, marker_id=marker_id)
            if attempt > 1:
                print(f"marker found on attempt {attempt}")
            return image
        except MarkerNotFound:
            print(f"attempt {attempt}/{attempts}: no marker, retrying")
    if image is not None:
        # Save the last bad frame so the cause is visible.
        cv2.imwrite("calib_fail.png", image)
    return None

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ArUco marker -> pixel/mm calibration")
    parser.add_argument("image", nargs="?", help="saved frame (PNG/JPG); omit with --live")
    parser.add_argument("--live", action="store_true", help="grab one frame from the Basler")
    parser.add_argument("--marker-mm", type=float, default=MARKER_MM,
                        help=f"printed marker edge in mm (default {MARKER_MM:.0f}, measured)")
    parser.add_argument("--id", type=int, default=MARKER_ID, help="marker id to use")
    parser.add_argument("--out", default=str(CALIBRATION_PATH), help="where to write the JSON")
    arguments = parser.parse_args(argv)

    if arguments.live:
        image = _live_frame_with_marker(arguments.id)
        if image is None:
            print(f"no marker in {LIVE_ATTEMPTS} frames - last frame saved as calib_fail.png",
                  file=sys.stderr)
            print("check glare, focus, and that nothing covers the marker", file=sys.stderr)
            return 1
        source = "live frame"
    elif arguments.image:
        image = cv2.imread(arguments.image, cv2.IMREAD_COLOR)
        if image is None:
            resolved = Path(arguments.image).resolve()
            hint = (
                "it exists but OpenCV could not decode it - it may be an empty "
                "or cloud-only (OneDrive) file: open it once in an image viewer, "
                "or right-click it and choose 'Always keep on this device'"
                if resolved.is_file()
                else "no such file here - you are probably in the wrong folder"
            )
            parser.error(
                f"cannot read {arguments.image}\n"
                f"  folder    : {Path.cwd()}\n"
                f"  full path : {resolved}\n"
                f"  exists    : {resolved.is_file()}\n"
                f"  hint      : {hint}\n"
                f"  with the camera plugged in you want: calibrate.py --live\n"
                f"  the PNG is only for the offline check - run from the folder "
                f"that holds calibrate.py"
            )
        source = arguments.image
    else:
        parser.error("give an image path or --live")

    # One marker gives scale, rotation and millimetre mapping.
    calibration = marker_scale(image, marker_mm=arguments.marker_mm, marker_id=arguments.id)
    print(f"source        {source} ({image.shape[1]}x{image.shape[0]})")
    print(f"marker id     {arguments.id}, {calibration.marker_mm:.2f} mm")
    print(f"centre px     ({calibration.centre_px[0]:.2f}, {calibration.centre_px[1]:.2f})")
    print(f"sides px      {[round(side, 2) for side in calibration.sides_px]}")
    print(f"mm_per_px     {calibration.mm_per_px():.6f}  (scalar method, for comparison)")
    print(f"rotation      {calibration.rotation_deg:+.3f} deg")
    print(f"trapezoid     {100.0 * (max(calibration.sides_px) / min(calibration.sides_px) - 1.0):.1f}% "
          f"spread, removed by the homography")
    print(f"frame         {calibration.frame_turns} quarter turn(s) removed, "
          f"so +x = image right and +y = image up")
    written = save_calibration(calibration, arguments.out)
    print(f"written       {written}")
    return 0

MOVED_LIMIT_PX = 15.0


def marker_moved(plane: PlaneCalibration, image: np.ndarray) -> str | None:
    """Compare the marker in this frame with the saved calibration.

    Returns None when they agree (or the marker is not in view), or a message
    saying how far it moved, so a stale calibration.json is never used silently.
    """
    try:
        fresh = marker_scale(image, marker_mm=plane.marker_mm)
    except MarkerNotFound:
        return None
    dx = fresh.centre_px[0] - plane.centre_px[0]
    dy = fresh.centre_px[1] - plane.centre_px[1]
    moved = math.hypot(dx, dy)
    if moved <= MOVED_LIMIT_PX:
        return None
    return (f"the marker has moved {moved:.0f} px (about {moved * fresh.mm_per_px():.0f} mm) since "
            f"calibration.json was written: saved centre ({plane.centre_px[0]:.0f}, "
            f"{plane.centre_px[1]:.0f}) px, now ({fresh.centre_px[0]:.0f}, {fresh.centre_px[1]:.0f}) px.\n"
            "  re-run calibrate.py --live; if the sheet itself moved, also re-teach wobjS at the marker centre")


def load_plane(path: Path | str = CALIBRATION_PATH) -> PlaneCalibration | None:
    try:
        return load_calibration(path)
    except (FileNotFoundError, ValueError) as error:
        print(f"no usable calibration: {error}", file=sys.stderr)
        print(r"run: .venv\Scripts\python.exe calibrate.py --live"
              "   (marker flat, in view, nothing on top of it)", file=sys.stderr)
        return None

if __name__ == "__main__":
    raise SystemExit(main())
