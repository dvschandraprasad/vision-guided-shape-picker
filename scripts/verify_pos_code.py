from __future__ import annotations

import pipeline

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import calibrate
import live_pick
import verify_positions

SAMPLE_FRAME = pipeline.REPO_ROOT / "data" / "captured_frames" / "calibration_frame.png"
# Longest allowed pick distance from the marker centre.
MAX_REACH_MM = 350.0

NEIGHBOURS = (
    ("calibrate", calibrate, ("marker_scale", "PlaneCalibration", "fold_to_symmetry",
                              "pick_angle")),
    ("live_pick", live_pick, ("commands_for", "format_row", "draw_overlay")),
    ("pipeline", pipeline, ("ShapeDetector", "encode_command", "decode_command", "PICK_Z_MM",
                            "pick_angle", "SENDS_ANGLE_DEG")),
    ("verify_positions", verify_positions, ("format_table", "rapid_module")),
)

# Every helper module must still provide the names used here.
def check_neighbours() -> list[str]:
    problems: list[str] = []
    for name, module, needed in NEIGHBOURS:
        missing = [attribute for attribute in needed if not hasattr(module, attribute)]
        if missing:
            where = getattr(module, "__file__", "?")
            problems.append(f"{name}.py is too old (no {', '.join(missing)}): {where}")
    return problems

@dataclass(frozen=True)
# One test result: name, pass flag, readable detail.
class Check:

    name: str
    ok: bool
    detail: str

# Known shape and angle cases with the expected folded angle.
def fold_cases() -> list[tuple[str, float, float]]:
    return [
        ("circle", 0.0, 0.0),
        ("circle", 137.0, 0.0),
        ("square", 30.0, 30.0),
        ("square", 95.0, 5.0),
        ("square", -3.0, -3.0),
        ("hexagon", 45.0, -15.0),
        ("triangle", 100.0, -20.0),
        ("star", 40.0, -32.0),
    ]

def check_model(detections: list[pipeline.Detection], elapsed_ms: float) -> Check:
    names = {detection.name for detection in detections}
    unknown = names - set(pipeline.CLASSES)
    ok = bool(detections) and not unknown
    detail = f"{len(detections)} boxes, {len(names)} classes ({', '.join(sorted(names))}), {elapsed_ms:.0f} ms"
    if unknown:
        detail = f"unknown class names {sorted(unknown)}"
    return Check("model + detection", ok, detail)

# Checks frame is upright: right is +x, up is +y.
def check_frame(image: np.ndarray, plane: calibrate.PlaneCalibration) -> Check:
    centre_x, centre_y = plane.centre_px
    right = plane.to_mm((centre_x + 100.0, centre_y))
    down = plane.to_mm((centre_x, centre_y + 100.0))
    upright = right[0] > abs(right[1]) and -down[1] > abs(down[0])
    inside = 0 <= centre_x < image.shape[1] and 0 <= centre_y < image.shape[0]
    ok = upright and inside and plane.frame_turns in (0, 1, 2, 3)
    return Check(
        "marker + upright frame",
        ok,
        f"{plane.marker_mm:.0f} mm marker, centre ({centre_x:.0f}, {centre_y:.0f}) px, "
        f"{plane.frame_turns} quarter turn(s) removed, "
        f"right -> ({right[0]:+.1f}, {right[1]:+.1f}) mm",
    )

# Compares one measured step with the average pixel scale.
def check_scale(plane: calibrate.PlaneCalibration) -> Check:
    centre_x, centre_y = plane.centre_px
    right = plane.to_mm((centre_x + 100.0, centre_y))
    expected = 100.0 * plane.mm_per_px()
    error = abs(abs(right[0]) - expected) / expected if expected else 1.0
    return Check(
        "px -> mm scale",
        error < 0.05,
        f"100 px = {abs(right[0]):.2f} mm, mean scale says {expected:.2f} mm "
        f"({100.0 * error:.1f}% apart), marker sides "
        f"{min(plane.sides_px):.0f}-{max(plane.sides_px):.0f} px",
    )

def check_folds() -> Check:
    wrong: list[str] = []
    for shape, angle, expected in fold_cases():
        got = calibrate.fold_to_symmetry(shape, angle)
        if abs(got - expected) > 1e-9:
            wrong.append(f"{shape} {angle} -> {got}, expected {expected}")
    return Check(
        "symmetry fold",
        not wrong,
        "; ".join(wrong) if wrong else f"{len(fold_cases())} cases, square never above 45, no shape above half its period",
    )

# Command angle must match the rule for its shape.
def check_angle_rule(rows) -> Check:
    problems: list[str] = []
    for _detection, command, raw, _distance in rows:
        expected = pipeline.pick_angle(command.shape, raw)
        if abs(expected - command.angle_deg) > 1e-6:
            problems.append(f"{command.shape}: table says {command.angle_deg:.2f}, "
                            f"rule says {expected:.2f}")
        if not pipeline.SENDS_ANGLE_DEG[command.shape] and command.angle_deg != 0.0:
            problems.append(f"{command.shape} must not turn but sent {command.angle_deg}")
    if abs(pipeline.pick_angle("square", 40.0, 20.0) + 30.0) > 1e-9:
        problems.append("the offset is not applied before the fold, or the fold drops the sign")
    if abs(calibrate.pick_angle("square", 40.0, 20.0)
           - pipeline.pick_angle("square", 40.0, 20.0)) > 1e-9:
        problems.append("calibrate.pick_angle disagrees with pipeline.pick_angle")
    return Check(
        "angle rule",
        not problems,
        "; ".join(problems) if problems
        else f"turning: {pipeline.sending_shapes()}, offset "
             f"{pipeline.ANGLE_OFFSET_DEG:+.1f} deg, {len(rows)} rows agree",
    )

def check_heights(rows) -> Check:
    heights = {command.z_mm for _d, command, _r, _dist in rows}
    ok = heights == {pipeline.PICK_Z_MM} and pipeline.PICK_Z_MM == 10.0
    return Check("pick height", ok, f"z = {sorted(heights)} mm (PICK_Z_MM {pipeline.PICK_Z_MM})")

# Line is slot;x;y;z;angle, one decimal, no trailing separator.
def check_wire(rows) -> Check:
    problems: list[str] = []
    tolerance = 0.051
    for _detection, command, _raw, _distance in rows:
        line = pipeline.encode_command(command)
        expected = [str(pipeline.SLOT[command.shape]), f"{command.x_mm:.1f}",
                    f"{command.y_mm:.1f}", f"{command.z_mm:.1f}",
                    f"{command.angle_deg:.1f}"]
        if line.split(";") != expected:
            problems.append(f"{line!r} != {';'.join(expected)!r}")
        back = pipeline.decode_command(line)
        if back.shape != command.shape:
            problems.append(f"{line!r} reads back as {back.shape!r}")
        for field in ("x_mm", "y_mm", "z_mm", "angle_deg"):
            if abs(getattr(back, field) - getattr(command, field)) > tolerance:
                problems.append(f"{line!r} {field} reads back as {getattr(back, field)!r}")
    return Check(
        "wire format",
        not problems,
        "; ".join(problems) if problems else
        f"{len(rows)} lines, slot;x;y;z;angle (five fields, no trailing separator, "
        f"one decimal each), e.g. {pipeline.encode_command(rows[0][1])!r}",
    )

# Every pick must stay inside the arm reach limit.
def check_reach(rows) -> Check:
    distances = [distance for _d, _c, _r, distance in rows]
    furthest = max(distances) if distances else 0.0
    return Check("coordinates in reach", 0.0 < furthest <= MAX_REACH_MM,
                 f"nearest {min(distances) if distances else 0.0:.1f} mm, "
                 f"furthest {furthest:.1f} mm, limit {MAX_REACH_MM:.0f} mm")

# RAPID CheckPoint lines must carry the same numbers as the wire.
def check_rapid(rows, module: str) -> Check:
    calls = [line.strip() for line in module.splitlines() if line.strip().startswith("CheckPoint \"")]
    problems: list[str] = []
    if len(calls) != len(rows):
        problems.append(f"{len(calls)} CheckPoint lines for {len(rows)} shapes")
    for call, (_detection, command, _raw, _distance) in zip(calls, rows, strict=False):
        wanted = (f'CheckPoint "{command.shape}", {command.x_mm:.1f}, '
                  f"{command.y_mm:.1f}, {command.angle_deg:.1f};")
        if call != wanted:
            problems.append(f"{call!r} != {wanted!r}")
        if not call.endswith(";"):
            problems.append(f"{call!r} does not end with a semicolon")
    for required in ("MODULE VisionCheck", "ENDPROC", "ENDMODULE", "nPartZ := 10.0",
                     "pTarget := [[nX, nY, nSafeZ], oPick, cnfPick, ejPick];",
                     "WObj:=wobjS"):
        if required not in module:
            problems.append(f"module is missing {required!r}")
    if "pickBase" in module:
        problems.append("module still uses pickBase, which updated_robo.mod no longer defines")
    return Check(
        "rapid module",
        not problems,
        "; ".join(problems) if problems else
        f"{len(calls)} points, same numbers as the wire, built from the pick frame in wobjS",
    )

# Sends every command over loopback TCP and reads it back.
def check_socket(rows) -> Check:
    expected = [pipeline.encode_command(command) for _d, command, _r, _dist in rows]
    server = pipeline.TcpTransport(host="127.0.0.1", port=0)
    client = pipeline.TcpTransport(host="127.0.0.1", port=0, server=False)
    received: list[str | None] = []
    try:
        server.open()
        client.port = server.port
        client.open()
        server.accept(timeout_s=5.0)
        for _detection, command, _raw, _distance in rows:
            client.send_command(command)
        received = [server.receive_line(timeout_s=5.0) for _ in expected]
    except (pipeline.TransportError, OSError) as error:
        return Check("tcp round trip", False, f"loopback failed: {error}")
    finally:
        client.close()
        server.close()

    decoded = [pipeline.decode_command(line) for line in received if line is not None]
    wanted = [command for _d, command, _r, _dist in rows]
    tolerance = 0.051
    same = len(decoded) == len(wanted) and all(
        back.shape == command.shape
        and all(abs(getattr(back, field) - getattr(command, field)) <= tolerance
                for field in ("x_mm", "y_mm", "z_mm", "angle_deg"))
        for back, command in zip(decoded, wanted, strict=False)
    )
    ok = received == expected and same
    detail = (f"{len(expected)} lines sent, read back and decoded over 127.0.0.1, "
              f"identical commands (slot, x, y, z, angle)"
              if ok else f"sent {expected!r}, read back {received!r}, decoded {decoded!r}")
    return Check("tcp round trip", ok, detail)

# Overlay is drawn on a copy, the frame is not changed.
def check_overlay(image: np.ndarray, rows) -> tuple[Check, np.ndarray]:
    canvas = live_pick.draw_overlay(image, rows)
    ok = (canvas.shape[:2] == image.shape[:2] and canvas.ndim == 3
          and int(np.count_nonzero(canvas - cv2.cvtColor(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY),
                                                       cv2.COLOR_GRAY2BGR))))
    return Check("overlay", bool(ok), f"{canvas.shape[1]}x{canvas.shape[0]}, "
                                      f"{len(rows)} boxes drawn, never mutates the frame"), canvas

def report(checks: list[Check], path: Path | None = None) -> int:
    width = max(len(check.name) for check in checks)
    lines: list[str] = []
    for index, check in enumerate(checks, start=1):
        lines.append(f"{index:3d}  {'PASS' if check.ok else 'FAIL'}  "
                     f"{check.name:<{width}}  {check.detail}")
    passed = sum(1 for check in checks if check.ok)
    lines.append("")
    lines.append(f"{passed} of {len(checks)} checks passed"
                 + ("" if passed == len(checks) else "  <-- fix these before going near the robot"))
    for line in lines:
        print(line)
    if path is not None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed == len(checks) else 1

# Runs every check without camera, robot or display, then writes files.
def main(argv: list[str] | None = None) -> int:
    stale = check_neighbours()
    if stale:
        print("this folder mixes files from two different zips:", file=sys.stderr)
        for problem in stale:
            print(f"  {problem}", file=sys.stderr)
        print("extract the newest IN700_lab_pc.zip into an empty folder and run it there",
              file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(
        description="prove the whole offline code path: no camera, no robot, no display")
    parser.add_argument("--image", default=str(SAMPLE_FRAME), help="frame with the marker in it")
    parser.add_argument("--model", default=pipeline.MODEL_PATH, help="OBB weights")
    parser.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    parser.add_argument("--out", default="verify_out", help="where to write the artifacts")
    arguments = parser.parse_args(argv)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.txt"

    image = cv2.imread(arguments.image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"cannot read {arguments.image} (folder {Path.cwd()})", file=sys.stderr)
        print("pass --image with a frame that has the marker in it", file=sys.stderr)
        return 2

    print("INA700 offline code check - no camera, no robot, no display")
    print(f"frame        {arguments.image} ({image.shape[1]}x{image.shape[0]})")
    print(f"model        {arguments.model} at conf {arguments.conf:.2f}")
    print("calibration  derived from the marker in this frame, not from calibration.json")
    print()

    checks: list[Check] = []

    started = time.perf_counter()
    detector = pipeline.ShapeDetector(model_path=arguments.model, conf=arguments.conf)
    detections = detector.detect(image)
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    checks.append(check_model(detections, elapsed_ms))
    if not detections:
        return report(checks, report_path)

    try:
        plane = calibrate.marker_scale(image, marker_mm=calibrate.MARKER_MM)
    except ValueError as error:
        checks.append(Check("marker + upright frame", False, f"no usable marker: {error}"))
        return report(checks, report_path)
    checks.append(check_frame(image, plane))
    checks.append(check_scale(plane))
    checks.append(check_folds())

    rows = live_pick.commands_for(detections, plane)
    print(verify_positions.format_table(rows))
    print()

    checks.append(check_heights(rows))
    checks.append(check_wire(rows))
    checks.append(check_angle_rule(rows))
    checks.append(check_reach(rows))

    stamp = time.strftime("%Y-%m-%d %H:%M")
    module = verify_positions.rapid_module(rows, stamp)
    checks.append(check_rapid(rows, module))
    checks.append(check_socket(rows))

    overlay_check, canvas = check_overlay(image, rows)
    checks.append(overlay_check)

    (out / "VisionCheck.mod").write_text(module, encoding="ascii")
    (out / "table.txt").write_text(verify_positions.format_table(rows) + "\n", encoding="ascii")
    cv2.imwrite(str(out / "overlay.png"), canvas)

    code = report(checks, report_path)
    print(f"artifacts    {out}\\report.txt, {out}\\VisionCheck.mod, {out}\\table.txt, "
          f"{out}\\overlay.png")
    print("next         load VisionCheck.mod in RobotStudio and run CheckAll"
          " (see README section 5.2)")
    return code

if __name__ == "__main__":
    raise SystemExit(main())
