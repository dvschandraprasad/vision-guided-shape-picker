from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import calibrate
import pipeline
from live_pick import axis_summary, commands_detail, row_flags

# Hover height in mm above the sheet.
DEFAULT_SAFE_Z_MM = 50.0

TEMPLATE = r'''MODULE VisionCheck
  ! Written by verify_positions.py on {stamp}.
  ! These points are measured, never taught: offsets from the marker centre
  ! in wobjS, the same numbers the pick message carries. Needs wobjS, toolss
  ! and the pick frame (oPick, cnfPick, ejPick) from updated_robo.mod.
  !
  ! CheckPoint goes to the hover height, drops to the part height, waits two
  ! seconds and lifts. Watch the cup: at the part height it should touch the
  ! top of that part. The vacuum is never set, so nothing is picked.
  ! If the gripper cannot rotate, delete the RelTool line - the point is the
  ! same, only the gripper stays where it is.

  LOCAL CONST num nSafeZ := {safe_z};
  LOCAL CONST num nPartZ := {part_z};

  PROC CheckPoint(string sName, num nX, num nY, num nAng)
    VAR robtarget pTarget;
    pTarget := [[nX, nY, nSafeZ], oPick, cnfPick, ejPick];
    MoveJ pTarget, v300, z20, toolss\WObj:=wobjS;
    pTarget := RelTool([[nX, nY, nPartZ], oPick, cnfPick, ejPick],
      0, 0, 0 \Rz:=nAng);
    MoveL pTarget, v50, fine, toolss\WObj:=wobjS;
    TPWrite "check " + sName;
    WaitTime 2;
    pTarget := [[nX, nY, nSafeZ], oPick, cnfPick, ejPick];
    MoveL pTarget, v500, z20, toolss\WObj:=wobjS;
  ENDPROC

  PROC CheckAll()
{calls}
  ENDPROC
ENDMODULE
'''


# Prints one text line for each pick point.
def format_table(rows: list[pipeline.Row],
                 flags: list[list[str]] | None = None) -> str:
    lines = ["  #  shape     slot      x_mm      y_mm    z_mm   angle   conf      r_mm  wire"]
    for index, (detection, command, _raw, distance) in enumerate(rows, start=1):
        mark = flags[index - 1] if flags else []
        lines.append(
            f"{index:3d}  {command.shape:<8} {pipeline.SLOT[command.shape]:4d} "
            f"{command.x_mm:9.1f} {command.y_mm:9.1f} {command.z_mm:7.1f} "
            f"{command.angle_deg:7.1f} {detection.confidence:6.2f} {distance:9.1f}  "
            f"{pipeline.encode_command(command)}" + (f"  [{' '.join(mark)}]" if mark else ""))
    return "\n".join(lines)


# Format decimal numbers for the generated RAPID.
def rapid_number(value: float) -> str:
    return f"{float(value):.1f}"


# Builds the RAPID check module text.
def rapid_module(rows: list[pipeline.Row], stamp: str,
                 safe_z: float = DEFAULT_SAFE_Z_MM,
                 part_z: float | None = None) -> str:
    # Uses the first point height when none given.
    part = rows[0][1].z_mm if part_z is None else float(part_z)
    calls = "\n".join(
        f'    CheckPoint "{command.shape}", {rapid_number(command.x_mm)}, '
        f"{rapid_number(command.y_mm)}, {rapid_number(command.angle_deg)};"
        for _detection, command, _raw, _distance in rows)
    return TEMPLATE.format(stamp=stamp, safe_z=rapid_number(safe_z),
                           part_z=rapid_number(part), calls=calls)


# Reads the command line options.
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pick coordinates in mm + a RAPID module that checks them")
    parser.add_argument("--image", help="saved frame (PNG/JPG); omit to grab the Basler once")
    parser.add_argument("--calibration", default=str(calibrate.CALIBRATION_PATH))
    parser.add_argument("--model", default=pipeline.MODEL_PATH, help="OBB weights")
    parser.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    parser.add_argument("--safe-z", type=float, default=DEFAULT_SAFE_Z_MM,
                        help="hover height above the sheet, used by the RAPID module")
    parser.add_argument("--rapid", default="VisionCheck.mod", help="where to write the module")
    parser.add_argument("--no-rapid", action="store_true", help="write no RAPID module")
    parser.add_argument("--angle-offset", type=float, default=pipeline.ANGLE_OFFSET_DEG,
                        help="degrees added before the symmetry fold (decision #17)")
    parser.add_argument("--overlay", help="where to write the annotated image")
    parser.add_argument("--no-overlay", action="store_true", help="write no image")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    print(pipeline.apply_angle_teach())

    # Read the saved marker-plane calibration.
    plane = calibrate.load_plane(arguments.calibration)
    if plane is None:
        return 2
    loaded = pipeline.read_image(arguments.image, "live frame")
    if loaded is None:
        print(f"cannot read {arguments.image} (folder {Path.cwd()})", file=sys.stderr)
        return 2
    image, source = loaded
    # Check whether the marker moved since calibration.
    moved = calibrate.marker_moved(plane, image)
    if moved:
        print(f"STOP         {moved}", file=sys.stderr)
        return 2

    # Prepare the OBB detector for this frame.
    detector = pipeline.ShapeDetector(model_path=arguments.model, conf=arguments.conf)
    detections = detector.detect(image)
    if not detections:
        print("no shapes found - lower --conf (a tight crop finds nothing)")
        return 1

    # Turns detections into millimetre pick commands.
    detail = commands_detail(detections, plane, arguments.angle_offset, image)
    rows = [row for row, _used in detail]
    flags = [row_flags(row, used) for row, used in detail]
    print(f"source       {source} ({image.shape[1]}x{image.shape[0]})")
    print(f"frame axes   {axis_summary(plane)}"
          "   (+x right, +y up is the documented frame)")
    print("origin       marker centre = wobjS zero, z = height above the sheet")
    print(f"angle        offset {arguments.angle_offset:+.1f} deg   "
          f"turning: {pipeline.sending_shapes()}   (every other shape sends 0)")
    print()
    print(format_table(rows, flags))
    if any(flags):
        print("             UNTAUGHT = angle teach active but this shape not taught; "
              "FALLBACK = no template fit. Do not trust the angle of these rows.")

    # Writes the RAPID module unless turned off.
    if not arguments.no_rapid:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        target = Path(arguments.rapid)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rapid_module(rows, stamp, safe_z=arguments.safe_z),
                          encoding="ascii")
        print()
        print(f"rapid        {target}   ({len(rows)} points)")
        print("             load it in the same RobotStudio task as MainModule, then run")
        print("             CheckAll - it hovers 50 mm above each point, drops to the part")
        print("             height, waits 2 s and lifts. Nothing is picked, nothing is placed.")

    if not arguments.no_overlay:
        name = Path(source).stem if arguments.image else "live"
        target = Path(arguments.overlay or Path("data/captured_frames") / f"check_{name}.png")
        print(f"overlay      {pipeline.write_overlay(image, rows, target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
