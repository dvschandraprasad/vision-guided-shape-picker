from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import calibrate
import pipeline

Row = tuple[pipeline.Detection, pipeline.PickCommand, float, float]


def commands_for(
    detections: list[pipeline.Detection],
    plane: calibrate.PlaneCalibration,
    angle_offset_deg: float = pipeline.ANGLE_OFFSET_DEG,
    image: np.ndarray | None = None,
) -> list[Row]:
    # One pick command per part, nearest first.
    return [row for row, _used in commands_detail(detections, plane, angle_offset_deg, image)]


def commands_detail(
    detections: list[pipeline.Detection],
    plane: calibrate.PlaneCalibration,
    angle_offset_deg: float = pipeline.ANGLE_OFFSET_DEG,
    image: np.ndarray | None = None,
) -> list[tuple[Row, bool]]:
    """commands_for plus, per row, whether the angle came from the template fit."""
    rows: list[tuple[Row, bool]] = []
    for detection in detections:
        x_mm, y_mm, raw, used = plane.parts_mm_detail(detection, image)
        # Turn the raw angle into the robot pick angle.
        angle = calibrate.pick_angle(detection.name, raw, angle_offset_deg)
        command = pipeline.PickCommand(detection.name, x_mm, y_mm, angle)
        rows.append(((detection, command, raw, float(np.hypot(x_mm, y_mm))), used))
    return sorted(rows, key=lambda item: item[0][3])


def row_flags(row: Row, used_template: bool | None) -> list[str]:
    """Why this row must not be sent: UNTAUGHT (teach active, shape not taught)
    and/or FALLBACK (angled shape whose angle did not come from the template
    fit). used_template None = unknown (scalar path), not flagged."""
    shape = row[1].shape
    flags = []
    # Flag shapes the robot has not been taught.
    if pipeline.untaught(shape):
        flags.append("UNTAUGHT")
    if used_template is False and pipeline.SENDS_ANGLE_DEG.get(shape, False):
        flags.append("FALLBACK")
    return flags


def scalar_rows(
    detections: list[pipeline.Detection],
    plane: calibrate.PlaneCalibration,
    angle_offset_deg: float = pipeline.ANGLE_OFFSET_DEG,
) -> list[Row]:
    mapper = pipeline.CoordinateMapper(plane.as_scalar_calibration())
    rows: list[Row] = []
    for detection in detections:
        command = mapper.to_pick_command(detection, angle_offset_deg)
        distance = float(np.hypot(command.x_mm, command.y_mm))
        rows.append((detection, command, mapper.angle_in_frame(detection.angle_deg), distance))
    return sorted(rows, key=lambda row: row[3])


def axis_summary(plane: calibrate.PlaneCalibration) -> str:
    centre_x, centre_y = plane.centre_px
    # Show where image right and down point in millimetres.
    right = plane.to_mm((centre_x + 100.0, centre_y))
    down = plane.to_mm((centre_x, centre_y + 100.0))
    return (f"image right -> ({right[0]:+6.1f}, {right[1]:+6.1f}) mm   "
            f"image down  -> ({down[0]:+6.1f}, {down[1]:+6.1f}) mm")


def format_row(row: Row, flags: list[str] | None = None) -> str:
    detection, command, raw, distance = row
    mark = f"  [{' '.join(flags)}]" if flags else ""
    return (f"{command.shape:<8} {detection.confidence:4.2f}  "
            f"({command.x_mm:8.1f}, {command.y_mm:8.1f}) mm  "
            f"r {distance:7.1f}  angle {raw:7.2f} -> {command.angle_deg:7.2f}  "
            f"|  {pipeline.encode_command(command)}{mark}")


def send_refusal(flagged: list[tuple[Row, list[str]]], allow_fallback: bool) -> str | None:
    """Why --send must not send these rows, or None."""
    problems = []
    for row, flags in flagged:
        shape = row[1].shape
        if "UNTAUGHT" in flags:
            problems.append(pipeline.pick_blocked(shape) or f"{shape} untaught")
        if "FALLBACK" in flags and not allow_fallback:
            problems.append(f"REFUSED {shape}: angle from the outline/box FALLBACK, not the "
                            "template fit (better light, or --allow-fallback to accept it)")
    return "\n".join(problems) or None


# Draw part outlines and pick values on the picture.
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


def grab_frame() -> np.ndarray:
    from pypylon import pylon

    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("no Basler camera detected")
    # Use the first Basler camera found.
    camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
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


# Load saved calibration, or explain how to make one.
def load_plane(path: str | Path) -> calibrate.PlaneCalibration | None:
    try:
        return calibrate.load_calibration(path)
    except (FileNotFoundError, ValueError) as error:
        print(f"no usable calibration: {error}", file=sys.stderr)
        print("run: .venv\\Scripts\\python.exe calibrate.py --live"
              "   (marker flat, in view, nothing on top of it)", file=sys.stderr)
        return None


def read_image(path: str | None, live_label: str) -> tuple[np.ndarray, str] | None:
    if path:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None if image is None else (image, str(path))
    return grab_frame(), live_label


def write_overlay(image: np.ndarray, rows: list[Row], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), draw_overlay(image, rows))
    return path


def send_rows(rows: list[Row], host: str, port: int) -> int:
    with pipeline.TcpTransport(host=host, port=port, server=True) as transport:
        transport.open()
        print(f"listening    {host}:{transport.port}"
              "  (nothing is sent until the robot connects in)")
        try:
            # Wait for the robot to connect before sending.
            transport.accept()
        except pipeline.TransportError as error:
            print(f"not sent     {error}", file=sys.stderr)
            print("             the robot must run a RAPID socket client that"
                  " connects to this PC and a free port", file=sys.stderr)
            print("             if the address was rejected, it is not this"
                  " machine's address: run 'hostname -I' on the Pi and pass"
                  " THIS device's address, never the robot's (192.168.125.2)",
                  file=sys.stderr)
            return 1
        for _detection, command, _raw, _distance in rows:
            transport.send_command(command)
            print(f"sent         {pipeline.encode_command(command)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="one frame -> millimetre pick commands")
    parser.add_argument("--image", help="saved frame (PNG/JPG); omit to grab the Basler once")
    parser.add_argument("--calibration", default=str(calibrate.CALIBRATION_PATH))
    parser.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    parser.add_argument("--model", default=pipeline.MODEL_PATH, help="OBB weights")
    parser.add_argument("--scalar", action="store_true",
                        help="use the one-number mapper instead of the homography")
    parser.add_argument("--angle-offset", type=float, default=pipeline.ANGLE_OFFSET_DEG,
                        help="degrees added before the symmetry fold (decision #17)")
    parser.add_argument("--overlay", help="where to write the annotated image")
    parser.add_argument("--no-overlay", action="store_true", help="write no image")
    parser.add_argument("--send", action="store_true", help="send the lines to the robot")
    parser.add_argument("--host", default=None, help="this PC's address; needed with --send")
    parser.add_argument("--port", type=int, default=5000, help="port to listen on")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="with --send: also send angled parts whose template fit failed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    teach = pipeline.apply_angle_teach()
    # Simple mapper skips angle checks, so refuse this.
    if arguments.send and arguments.scalar:
        print("--scalar cannot be used with --send: the one-number mapper ignores the "
              "template fit and the angle teach checks", file=sys.stderr)
        return 2
    if arguments.send:
        blocked = pipeline.send_blocked()
        if blocked:
            print(blocked, file=sys.stderr)
            return 2
    if arguments.send and not arguments.host:
        print("--send needs --host <this PC's address on the robot network>", file=sys.stderr)
        print("             ipconfig: use the adapter on the robot's 192.168.125.x subnet,"
              " never the robot's own address", file=sys.stderr)
        return 2

    plane = load_plane(arguments.calibration)
    if plane is None:
        return 2
    loaded = read_image(arguments.image, "Basler live")
    if loaded is None:
        print(f"cannot read {arguments.image}", file=sys.stderr)
        return 2
    image, source = loaded
    # Check whether the marker moved since calibration.
    moved = calibrate.marker_moved(plane, image)
    if moved:
        print(f"STOP         {moved}", file=sys.stderr)
        return 2

    trapezoid = 100.0 * (max(plane.sides_px) / min(plane.sides_px) - 1.0)
    print(f"frame        {source} ({image.shape[1]}x{image.shape[0]})")
    print(f"calibration  {arguments.calibration}  method {'scalar' if arguments.scalar else 'homography'}")
    print(f"marker       centre ({plane.centre_px[0]:.2f}, {plane.centre_px[1]:.2f}) px, "
          f"mm_per_px {plane.mm_per_px():.6f}, trapezoid {trapezoid:.1f}%, "
          f"rotation {plane.rotation_deg:+.3f} deg")
    print(f"frame axes   {axis_summary(plane)}  (+x right, +y up is the documented frame)")
    print(f"angle        offset {arguments.angle_offset:+.1f} deg   "
          f"turning: {pipeline.sending_shapes()}   (every other shape sends 0)")
    print(f"             {teach}")

    detector = pipeline.ShapeDetector(model_path=arguments.model, conf=arguments.conf)
    # Detect visible parts in this frame.
    detections = detector.detect(image)
    if not detections:
        print("no shapes found - move the camera or lower --conf (a tight crop finds nothing)")
        return 1

    if arguments.scalar:
        rows = scalar_rows(detections, plane, arguments.angle_offset)
        flagged = [(row, row_flags(row, None)) for row in rows]
    else:
        detail = commands_detail(detections, plane, arguments.angle_offset, image)
        rows = [row for row, _used in detail]
        flagged = [(row, row_flags(row, used)) for row, used in detail]
    for row, flags in flagged:
        print(format_row(row, flags))

    if not arguments.no_overlay:
        name = Path(source).stem if arguments.image else "live"
        target = Path(arguments.overlay or Path("data/captured_frames") / f"overlay_{name}.png")
        print(f"overlay      {write_overlay(image, rows, target)}")

    if arguments.send:
        refusal = send_refusal(flagged, arguments.allow_fallback)
        if refusal:
            print(refusal, file=sys.stderr)
            print("nothing sent", file=sys.stderr)
            return 1
        return send_rows(rows, arguments.host, arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
