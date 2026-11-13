from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import calibrate
import pipeline

# Reverse teach, run together with AngleTeach.mod:
#   the robot drops a part taken out of its pocket with nAng = 0 -> theta0,
#   then with nAng = T -> theta1 = theta0 + k*T.
#   For a part seen at theta: nAng = k * fold(theta - theta0),
#   i.e. ANGLE_SIGN = k and SHAPE_OFFSET_DEG[shape] = -theta0.

# Largest error allowed between the two drop angles.
MAX_RESIDUAL_DEG = 3.0
TEACH_TURN_DEG = 20.0
FRESH_THETA0 = timedelta(minutes=15)   # a --turn 20 must follow its --turn 0 within this
MAX_DROP_DIST_MM = 15.0               # measured part centre vs the AngleTeach.mod drop point
DROP_X_MM = -120.0                    # = nTeachX in AngleTeach.mod
DROP_Y_MM = 60.0                      # = nTeachY in AngleTeach.mod


# One error type for every refused teach step.
class TeachError(RuntimeError):
    pass


def empty() -> dict[str, Any]:
    return {"version": 1, "turn_deg": TEACH_TURN_DEG, "shapes": {}, "sign": None}


# Read the teach file; a missing file means empty.
def load(path: str | Path = pipeline.ANGLE_TEACH_PATH) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return empty()
    try:
        values = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TeachError(f"{source} is not readable JSON ({error}); fix it, or move it away "
                         "and teach again") from None
    if (not isinstance(values, dict) or values.get("version") != 1
            or not isinstance(values.get("shapes"), dict)):
        raise TeachError(f"{source} is not a version 1 angle teach file")
    return values


# Save the teach values to disk safely.
def save(values: dict[str, Any], path: str | Path = pipeline.ANGLE_TEACH_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # atomic: a crash mid-write never leaves a half file for live_pick to read
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=target.name + ".", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(json.dumps(values, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


# Compare both drops to find the angle sign.
def solve(shape: str, theta0: float, theta1: float, turn_deg: float) -> tuple[int, float, float]:
    d = pipeline.fold_to_symmetry(shape, theta1 - theta0)
    k = (1 if d >= 0.0 else -1) * (1 if turn_deg >= 0.0 else -1)
    return k, abs(abs(d) - abs(turn_deg)), d


def global_sign(values: dict[str, Any]) -> tuple[int | None, list[str]]:
    # The sign is a property of the robot/camera pair, so every taught shape must
    # agree on it. On disagreement the sign is stored as null and pipeline ignores
    # the whole file (defaults stay in use) until the odd shape is re-taught.
    signs = {shape: entry["k"] for shape, entry in values["shapes"].items()
             if entry.get("k") is not None}
    if not signs:
        return None, []
    if len(set(signs.values())) > 1:
        detail = ", ".join(f"{shape} {k:+d}" for shape, k in sorted(signs.items()))
        bar = "!" * 72
        return None, [bar,
                      f"!! SHAPES DISAGREE ON THE SIGN ({detail})",
                      "!! sign set to null: live_pick / dashboard IGNORE the whole teach file",
                      "!! and run with the defaults until you re-teach the odd shape",
                      "!! (or --forget it). Check the part did not slip between the drops.",
                      bar]
    return next(iter(signs.values())), []


def invalidate(values: dict[str, Any], shape: str) -> None:
    """Drop a shape's teach (start of every --turn 0), so a refused or aborted
    turn-0 measurement leaves nothing usable behind."""
    values["shapes"].pop(shape, None)
    values["sign"], _ = global_sign(values)


def check_fresh(values: dict[str, Any], shape: str, now: datetime | None = None) -> None:
    """A nonzero turn needs a theta0 from a --turn 0 that is unused (k None)
    and less than FRESH_THETA0 old - otherwise it may belong to another drop."""
    entry = values["shapes"].get(shape, {})
    if entry.get("theta0") is None:
        raise TeachError(f"no theta0 for {shape}: run --turn 0 first (with the nAng 0 drop)")
    if entry.get("k") is not None:
        raise TeachError(f"the {shape} theta0 was already used by a finished teach; "
                         "rerun TeachShape and start again with --turn 0")
    try:
        measured = datetime.fromisoformat(str(entry.get("measured")))
    except ValueError:
        measured = None
    age = None if measured is None else (now or datetime.now()) - measured
    if age is None or age > FRESH_THETA0 or age < -timedelta(minutes=1):
        raise TeachError(f"the {shape} theta0 is stale (measured {entry.get('measured')}, "
                         f"limit {FRESH_THETA0.seconds // 60} min); rerun TeachShape and "
                         "start again with --turn 0")


def record(values: dict[str, Any], shape: str, turn_deg: float, theta: float,
           now: datetime | None = None) -> list[str]:
    if not pipeline.SENDS_ANGLE_DEG.get(shape, False):
        raise TeachError(f"{shape} sends no angle; nothing to teach")
    clock = now or datetime.now()
    stamp = clock.isoformat(timespec="seconds")
    lines: list[str] = []
    if turn_deg == 0.0:
        invalidate(values, shape)
        entry = values["shapes"].setdefault(shape, {})
        entry.update(theta0=pipeline.fold_to_symmetry(shape, theta), theta1=None, k=None,
                     residual_deg=None, taught=None, measured=stamp)
        lines.append(f"{shape}: theta0 = {entry['theta0']:+.2f} deg stored "
                     f"(raw {theta:+.2f}). Now press PLAY, then run --turn {TEACH_TURN_DEG:g}")
        return lines
    half = pipeline.SYMMETRY_DEG[shape] / 2.0
    if abs(turn_deg) + MAX_RESIDUAL_DEG >= half:
        raise TeachError(f"--turn {turn_deg:g} + residual limit {MAX_RESIDUAL_DEG:g} is not "
                         f"below half the {shape} period ({half:g} deg): the sign would be "
                         "ambiguous")
    check_fresh(values, shape, clock)
    entry = values["shapes"][shape]
    k, residual, d = solve(shape, entry["theta0"], theta, turn_deg)
    if residual > MAX_RESIDUAL_DEG:
        raise TeachError(
            f"{shape}: the part turned {d:+.2f} deg between the drops, expected "
            f"+-{abs(turn_deg):g} (residual {residual:.1f} > {MAX_RESIDUAL_DEG:g} deg). "
            "Not stored. Check the part did not slip, the right part was measured, "
            "and that both drops used the same point; then repeat --turn 0 and this step")
    # Both drops estimate the zero: theta0 directly, theta1 - k*T indirectly.
    # Store their circular mean (mod the period) to halve the measurement error.
    first = entry["theta0"]
    back = theta - k * turn_deg
    theta0 = pipeline.fold_to_symmetry(
        shape, first + pipeline.fold_to_symmetry(shape, back - first) / 2.0)
    entry.update(theta0=theta0, theta0_drop=first, theta1=pipeline.fold_to_symmetry(shape, theta),
                 k=k, residual_deg=round(residual, 3), taught=stamp)
    values["turn_deg"] = abs(turn_deg)
    values["sign"], warnings = global_sign(values)
    lines.append(f"{shape}: turned {d:+.2f} deg for nAng {turn_deg:+g} -> k {k:+d}, "
                 f"residual {residual:.2f} deg, zero {theta0:+.2f} (mean of both drops)")
    if values["sign"] is not None:
        lines.append(f"  equivalent: ANGLE_SIGN = {values['sign']:+d}, "
                     f"SHAPE_OFFSET_DEG['{shape}'] = {-entry['theta0']:+.2f}")
    return lines + warnings


# Angle the robot would be sent for this part.
def sent_for(values: dict[str, Any], shape: str, theta: float) -> float | None:
    entry = values["shapes"].get(shape, {})
    if values.get("sign") is None or entry.get("theta0") is None:
        return None
    return values["sign"] * pipeline.fold_to_symmetry(shape, theta - entry["theta0"])


MAX_SPREAD_DEG = 2.0
# Number of camera frames used for the median angle.
LIVE_FRAMES = 5


def circular_median(shape: str, angles: list[float]) -> tuple[float, float]:
    """Median of angles on the shape's symmetry circle and the largest deviation.

    Angles are compared modulo the shape period (a star at +35 and -36 is the
    same pose), so the result does not jump at the fold. The median is the
    sample with the smallest summed folded distance to all others.
    """
    if not angles:
        raise TeachError("no angle measured")
    folded = [pipeline.fold_to_symmetry(shape, a) for a in angles]
    median = min(folded, key=lambda c: (sum(abs(pipeline.fold_to_symmetry(shape, a - c))
                                            for a in folded), abs(c)))
    spread = max(abs(pipeline.fold_to_symmetry(shape, a - median)) for a in folded)
    return median, spread


# Get frames from image files or the live camera.
def grab_frames(arguments: argparse.Namespace) -> list[tuple[Any, str]]:
    if arguments.image:
        frames = []
        for path in arguments.image:
            loaded = pipeline.read_image(path, path)
            if loaded is None:
                raise TeachError(f"cannot read {path}")
            frames.append(loaded)
        return frames
    count = arguments.frames or LIVE_FRAMES
    try:
        from camera import BaslerSource
        source = BaslerSource()
        try:
            source.open()
            frames = []
            for index in range(count):
                frame = source.read()
                if frame is None:
                    raise TeachError("the camera returned no frame")
                frames.append((frame, f"Basler live {index + 1}/{count}"))
            return frames
        finally:
            source.close()
    except TeachError:
        raise
    except (RuntimeError, ImportError) as error:
        raise TeachError(f"camera: {error} (close Pylon Viewer / live_view, check the cable)") \
            from None


def measure_frame(image: Any, plane: Any, detector: Any,
                  shape: str | None) -> tuple[pipeline.Detection, float, float, float, bool]:
    detections = [d for d in detector.detect(image) if pipeline.SENDS_ANGLE_DEG[d.name]
                  and (shape is None or d.name == shape)]
    wanted = shape or "angled part"
    if len(detections) != 1:
        found = ", ".join(d.name for d in detections) or "none"
        raise TeachError(f"need exactly one {wanted} in view, found {len(detections)} ({found}); "
                         "clear the sheet or pass --shape")
    detection = detections[0]
    # exactly the call live_pick.commands_for makes (parts_mm is parts_mm_detail
    # without the flag); raw is before pick_angle's offset, sign and fold
    x_mm, y_mm, raw, used = plane.parts_mm_detail(detection, image)
    return detection, x_mm, y_mm, raw, used


# Measure the part angle; refuse doubtful results.
def measure(arguments: argparse.Namespace) -> tuple[str, float, str]:
    # Same plane as live_pick / dashboard / verify_positions: the saved
    # calibration, refused if the marker in the frame has moved.
    plane = calibrate.load_plane(arguments.calibration)
    if plane is None:
        raise TeachError(f"no usable calibration file {arguments.calibration}")
    frames = grab_frames(arguments)
    detector = pipeline.ShapeDetector(model_path=arguments.model, conf=arguments.conf)
    shape = arguments.shape
    raws: list[float] = []
    lines: list[str] = [f"calibration {arguments.calibration}",
                        f"drop point  ({arguments.drop_x:g}, {arguments.drop_y:g}) mm from "
                        "--drop-x/--drop-y: must match nTeachX/nTeachY in AngleTeach.mod"]
    fallbacks = 0
    for image, label in frames:
        moved = calibrate.marker_moved(plane, image)
        if moved:
            raise TeachError(moved)
        detection, x_mm, y_mm, raw, used = measure_frame(image, plane, detector, shape)
        if shape is None:
            shape = detection.name
        elif detection.name != shape:
            raise TeachError(f"{label}: saw {detection.name}, earlier frames saw {shape}")
        # Distance from the part to the taught drop point.
        off = ((x_mm - arguments.drop_x) ** 2 + (y_mm - arguments.drop_y) ** 2) ** 0.5
        if off > MAX_DROP_DIST_MM:
            raise TeachError(
                f"{label}: the {detection.name} is at ({x_mm:.1f}, {y_mm:.1f}) mm, {off:.0f} mm from "
                f"the drop point ({arguments.drop_x:g}, {arguments.drop_y:g}) (limit "
                f"{MAX_DROP_DIST_MM:g}). Wrong part, or --drop-x/--drop-y do not match "
                "nTeachX/nTeachY in AngleTeach.mod")
        # Count frames where the template fit was not used.
        fallbacks += not used
        raws.append(raw)
        lines.append(f"  {label}: {detection.name} {detection.confidence:.2f} at "
                     f"({x_mm:.1f}, {y_mm:.1f}) mm, raw {raw:+.2f} deg via "
                     f"{'template fit' if used else 'FALLBACK (no good template fit)'}")
    assert shape is not None
    # One angle for all frames, plus their spread.
    median, spread = circular_median(shape, raws)
    lines.append(f"  median {median:+.2f} deg (mod {pipeline.SYMMETRY_DEG[shape]:g}) over "
                 f"{len(raws)} frame(s), spread +-{spread:.2f} deg")
    info = "\n".join(lines)
    if fallbacks and not arguments.allow_fallback:
        raise TeachError(f"{info}\n{fallbacks} of {len(raws)} frame(s) used the outline/box "
                         "fallback, not the template fit. Not stored. Improve the light or "
                         "the drop point, or pass --allow-fallback if you accept it")
    if spread > MAX_SPREAD_DEG:
        raise TeachError(f"{info}\nangle spread {spread:.2f} > {MAX_SPREAD_DEG:g} deg between "
                         "frames. Not stored. Is the arm/a hand in view or the part moving?")
    if fallbacks:
        info += f"\n  WARNING: {fallbacks} frame(s) used the fallback (--allow-fallback)"
    return shape, median, info


# Print the stored teach in a short readable table.
def show(values: dict[str, Any]) -> str:
    lines = [f"sign {values.get('sign')}   turn {values.get('turn_deg')} deg"]
    for shape, entry in sorted(values["shapes"].items()):
        state = "taught" if entry.get("k") is not None else "theta0 only (run --turn 20)"
        theta0 = entry.get("theta0")
        offset = f"{-theta0:+.2f}" if theta0 is not None else "-"
        lines.append(f"  {shape:<9} theta0 {theta0}  theta1 {entry.get('theta1')}  "
                     f"k {entry.get('k')}  residual {entry.get('residual_deg')}  "
                     f"offset {offset}  {state}  {entry.get('taught') or entry.get('measured')}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="measure ANGLE_SIGN and per-shape zero "
                                     "from the AngleTeach.mod drops")
    parser.add_argument("--turn", type=float, help="nAng the robot used for this drop (0 or 20)")
    parser.add_argument("--shape", choices=[s for s, sends in pipeline.SENDS_ANGLE_DEG.items() if sends])
    parser.add_argument("--image", nargs="+", help="saved frame(s); omit to grab the Basler")
    parser.add_argument("--frames", type=int, default=None,
                        help=f"live frames to grab (default {LIVE_FRAMES}); the median is used")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="accept frames where the template fit was not used")
    parser.add_argument("--drop-x", type=float, default=DROP_X_MM,
                        help="drop point x in wobjS mm; must equal nTeachX in AngleTeach.mod")
    parser.add_argument("--drop-y", type=float, default=DROP_Y_MM,
                        help="drop point y in wobjS mm; must equal nTeachY in AngleTeach.mod")
    parser.add_argument("--model", default=pipeline.MODEL_PATH, help="OBB weights")
    parser.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    parser.add_argument("--calibration", default=str(calibrate.CALIBRATION_PATH))
    parser.add_argument("--file", default=str(pipeline.ANGLE_TEACH_PATH), help=argparse.SUPPRESS)
    parser.add_argument("--show", action="store_true", help="print the stored teach and exit")
    parser.add_argument("--forget", metavar="SHAPE", help="delete one shape from the file")
    return parser.parse_args(argv)


# Run one teach step: measure, check and save.
def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.frames is not None and arguments.frames < 1:
        print("--frames must be 1 or more", file=sys.stderr)
        return 2
    if arguments.image and arguments.frames:
        print("--frames is for the live camera; with --image every given file is one frame",
              file=sys.stderr)
        return 2
    try:
        values = load(arguments.file)
        if arguments.show:
            print(f"file   {arguments.file}")
            print(show(values))
            for line in global_sign(values)[1]:
                print(line)
            return 0
        if arguments.forget:
            if values["shapes"].pop(arguments.forget, None) is None:
                print(f"{arguments.forget} is not in {arguments.file}")
                return 1
            values["sign"], warnings = global_sign(values)
            save(values, arguments.file)
            print("\n".join([f"forgot {arguments.forget}", *warnings]))
            return 0
        if arguments.turn is None:
            print("give --turn 0 or --turn 20 (or --show / --forget SHAPE)", file=sys.stderr)
            return 2
        if arguments.shape is None:
            print("give --shape (the teach pendant prints the full command)", file=sys.stderr)
            return 2
        if arguments.turn == 0.0:
            # forget the old teach first: if this measurement is refused,
            # nothing usable is left for this shape
            invalidate(values, arguments.shape)
            save(values, arguments.file)
        else:
            check_fresh(values, arguments.shape)
        shape, raw, info = measure(arguments)
        print(info)
        lines = record(values, shape, arguments.turn, raw)
    except TeachError as error:
        print(f"REFUSED  {error}", file=sys.stderr)
        return 1
    save(values, arguments.file)
    print("\n".join(lines))
    sent = sent_for(values, shape, raw)
    if values.get("sign") is None and values["shapes"][shape].get("k") is not None:
        print("NOT USABLE YET: the sign is null because the shapes disagree (see above)")
    if sent is not None:
        print(f"this part would now be sent nAng {sent:+.2f} deg "
              f"(expected {arguments.turn:+g}: the robot picks it back with that turn)")
    print(f"saved  {arguments.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
