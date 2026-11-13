# Tests for teaching turn sign and angle offsets.
from __future__ import annotations

import json

import pytest

import pipeline
import teach_angle

ANGLED = ("square", "hexagon", "triangle", "star")


@pytest.fixture(autouse=True)
def keep_defaults(monkeypatch):
    monkeypatch.setattr(pipeline, "ANGLE_SIGN", pipeline.ANGLE_SIGN)
    monkeypatch.setattr(pipeline, "SHAPE_OFFSET_DEG", dict(pipeline.SHAPE_OFFSET_DEG))
    monkeypatch.setattr(pipeline, "TAUGHT_ZERO_DEG", {})


def same_mod(shape, a, b):
    return abs(pipeline.fold_to_symmetry(shape, a - b)) < 1e-6


def drop(theta0, k, n_ang):
    # camera angle of a part dropped with turn n_ang (robot turns it k*n_ang)
    # Camera angle of the part after the robot turns it.
    return theta0 + k * n_ang


@pytest.mark.parametrize("shape", ANGLED)
@pytest.mark.parametrize("k", (1, -1))
# solve() must recover the turn sign and fit.
@pytest.mark.parametrize("theta0", (-179.0, -47.3, 0.0, 12.5, 29.99, 59.0, 143.0))
def test_math_undoes_rotation(shape, k, theta0):
    period = pipeline.SYMMETRY_DEG[shape]
    theta1 = drop(theta0, k, 20.0)
    got_k, residual, _ = teach_angle.solve(shape, theta0, theta1, 20.0)
    assert got_k == k and residual < 1e-6
    for delta in (-period / 2 + 1e-3, -period / 2 + 0.5, -7.0, 0.0, 3.0, period / 2 - 0.5,
                  period / 2 - 1e-3, period / 2 + 0.2, 2 * period + 11.0):
        theta = theta0 + delta + period * 3
        n_ang = k * pipeline.fold_to_symmetry(shape, theta - theta0)
        assert abs(n_ang) <= period / 2 + 1e-9
        # picked with n_ang, placed with the taught pocket pose: the part's angle
        # relative to the nAng-0 pose must be a whole symmetry step
        assert same_mod(shape, theta - k * n_ang, theta0)


@pytest.mark.parametrize("shape", ANGLED)
@pytest.mark.parametrize("k", (1, -1))
def test_pick_angle_with_taught_globals(shape, k, tmp_path):
    theta0 = 17.0
    values = teach_angle.empty()
    teach_angle.record(values, shape, 0.0, theta0)
    teach_angle.record(values, shape, 20.0, drop(theta0, k, 20.0))
    path = teach_angle.save(values, tmp_path / "teach.json")
    text = pipeline.apply_angle_teach(path)
    assert text.startswith(f"angle teach: sign {k:+d}, taught {shape}; UNTAUGHT")
    assert shape not in text.split("UNTAUGHT")[1]
    assert pipeline.ANGLE_TEACH_STATE == "applied"
    assert pipeline.ANGLE_SIGN == k
    assert pipeline.SHAPE_OFFSET_DEG[shape] == pytest.approx(-theta0)
    for n_ang in (-11.0, 0.0, 9.5):
        theta = drop(theta0, k, n_ang)
        assert pipeline.pick_angle(shape, theta) == pytest.approx(n_ang)
    assert pipeline.pick_angle(shape, theta0 + 20.0 * k) == pytest.approx(20.0)


def test_triangle_example_minus_sign():
    # cell example: triangle seen at +25.9 deg, pocket zero at 0, sign -1 -> send -25.9
    values = teach_angle.empty()
    teach_angle.record(values, "triangle", 0.0, 0.0)
    teach_angle.record(values, "triangle", 20.0, -20.0)
    assert values["sign"] == -1
    assert teach_angle.sent_for(values, "triangle", 25.9) == pytest.approx(-25.9)


def test_json_round_trip(tmp_path):
    values = teach_angle.empty()
    teach_angle.record(values, "star", 0.0, 50.0)
    teach_angle.record(values, "star", 20.0, 30.5)
    path = teach_angle.save(values, tmp_path / "data" / "angle_teach.json")
    again = teach_angle.load(path)
    assert again == json.loads(json.dumps(values))
    assert again["version"] == 1 and again["turn_deg"] == 20.0 and again["sign"] == -1
    entry = again["shapes"]["star"]
    # stored zero = mean of the turn-0 drop (50) and theta1 - k*T (50.5), mod 72
    assert entry["theta0"] == pytest.approx(pipeline.fold_to_symmetry("star", 50.25))
    assert entry["theta0_drop"] == pytest.approx(pipeline.fold_to_symmetry("star", 50.0))
    assert entry["k"] == -1 and entry["residual_deg"] == pytest.approx(0.5)
    assert entry["taught"] is not None


def test_refuses_bad_residual_and_missing_theta0():
    values = teach_angle.empty()
    with pytest.raises(teach_angle.TeachError, match="theta0"):
        teach_angle.record(values, "square", 20.0, 5.0)
    teach_angle.record(values, "square", 0.0, 5.0)
    with pytest.raises(teach_angle.TeachError, match="residual"):
        teach_angle.record(values, "square", 20.0, 5.0 + 12.0)
    assert values["shapes"]["square"]["k"] is None and values["sign"] is None
    with pytest.raises(teach_angle.TeachError):
        teach_angle.record(values, "hexagon", 35.0, 0.0)
    with pytest.raises(teach_angle.TeachError):
        teach_angle.record(values, "circle", 0.0, 0.0)


def test_disagreeing_signs_null_and_loud():
    values = teach_angle.empty()
    teach_angle.record(values, "square", 0.0, 0.0)
    teach_angle.record(values, "square", 20.0, 20.0)
    assert values["sign"] == 1
    teach_angle.record(values, "star", 0.0, 0.0)
    lines = teach_angle.record(values, "star", 20.0, -20.0)
    assert any("DISAGREE" in line for line in lines)
    assert values["sign"] is None
    assert values["shapes"]["square"]["k"] == 1 and values["shapes"]["star"]["k"] == -1
    assert teach_angle.sent_for(values, "star", 5.0) is None
    # forgetting the odd shape restores agreement
    values["shapes"].pop("star")
    assert teach_angle.global_sign(values) == (1, [])


def test_apply_ignores_disagreeing_signs(tmp_path):
    values = teach_angle.empty()
    for shape, k in (("square", 1), ("triangle", -1)):
        teach_angle.record(values, shape, 0.0, 5.0)
        teach_angle.record(values, shape, 20.0, 5.0 + 20.0 * k)
    path = teach_angle.save(values, tmp_path / "teach.json")
    text = pipeline.apply_angle_teach(path)
    assert "IGNORED" in text and "DISAGREE" in text
    assert pipeline.ANGLE_SIGN == 1.0
    assert all(value == 0.0 for value in pipeline.SHAPE_OFFSET_DEG.values())
    assert pipeline.TAUGHT_ZERO_DEG == {}
    # a hand-edited sign that hides the disagreement is still refused
    raw = json.loads(path.read_text())
    raw["sign"] = 1
    path.write_text(json.dumps(raw))
    assert "IGNORED" in pipeline.apply_angle_teach(path)
    assert pipeline.SHAPE_OFFSET_DEG["triangle"] == 0.0


def test_retaught_theta0_keeps_sign_of_others():
    values = teach_angle.empty()
    teach_angle.record(values, "square", 0.0, 0.0)
    teach_angle.record(values, "square", 20.0, -20.0)
    teach_angle.record(values, "square", 0.0, 3.0)   # restart square
    assert values["sign"] is None   # no complete shape any more


@pytest.mark.parametrize("shape", ANGLED)
def test_circular_median_wraps_at_the_fold(shape):
    half = pipeline.SYMMETRY_DEG[shape] / 2.0
    # samples straddle the fold: +half-0.4 and -half+0.3 are 0.7 deg apart
    samples = [half - 0.4, -half + 0.3, half - 0.1, -half + 0.6, half - 0.2]
    median, spread = teach_angle.circular_median(shape, samples)
    # a linear mean would land near 0 and a linear spread near the half period
    assert abs(pipeline.fold_to_symmetry(shape, median - half)) <= 0.4
    assert spread <= 0.7 + 1e-9
    # same answer when the raw values carry whole periods
    period = pipeline.SYMMETRY_DEG[shape]
    shifted = [a + period * n for a, n in zip(samples, (3, -2, 0, 7, -1))]
    again, spread2 = teach_angle.circular_median(shape, shifted)
    assert abs(pipeline.fold_to_symmetry(shape, again - median)) < 1e-9
    assert spread2 == pytest.approx(spread)


def test_circular_median_plain_and_outlier():
    median, spread = teach_angle.circular_median("square", [10.1, 10.0, 9.9, 10.2, 30.0])
    assert median == pytest.approx(10.1) and spread == pytest.approx(19.9)
    median, spread = teach_angle.circular_median("star", [3.0])
    assert median == 3.0 and spread == 0.0
    with pytest.raises(teach_angle.TeachError):
        teach_angle.circular_median("star", [])


# --- measure(): the real path (calibration file, read_image, parts_mm, template
# fit) on parts painted into the sample frame; only the YOLO detector is stubbed.

# Real sample photo used by the measure tests.
FRAME = pipeline.Path(pipeline.__file__).resolve().parent / "frame_20261030_171238_018578.png"
needs_frame = pytest.mark.skipif(not FRAME.is_file(), reason="sample frame missing")


@pytest.fixture(scope="module")
def cell():
    import cv2
    import calibrate
    frame = cv2.imread(str(FRAME))
    return frame, calibrate.marker_scale(frame)


# Draws a fake part into the photo at a known angle.
def paint(cell, shape, angle, centre=(-120.0, 60.0), blob=False):
    import cv2
    import numpy as np
    import live_view
    import template_fit
    frame, plane = cell
    image = frame.copy()
    px = live_view._to_px(plane, template_fit.template_mm(shape, angle, 1.0, centre))
    if blob:   # a round smudge: YOLO box but no template fit
        c = px.mean(axis=0)
        cv2.circle(image, tuple(map(int, c)), int(np.linalg.norm(px - c, axis=1).max() * 0.8),
                   (60, 60, 60), -1)
    else:
        cv2.fillPoly(image, [np.round(px).astype(np.int32)], (60, 60, 60), cv2.LINE_AA)
    rect = cv2.minAreaRect(px.astype(np.float32))
    detection = pipeline.Detection(pipeline.CLASSES.index(shape), 0.9,
                                   tuple(map(float, rect[0])), float(rect[2]),
                                   tuple(map(tuple, cv2.boxPoints(rect).tolist())))
    return image, detection


def run_measure(tmp_path, monkeypatch, cell, frames, extra=()):
    import cv2
    import calibrate
    calibration = calibrate.save_calibration(cell[1], tmp_path / "calibration.json")
    by_frame = {}
    paths = []
    for index, (image, detection) in enumerate(frames):
        path = tmp_path / f"f{index}.png"
        cv2.imwrite(str(path), image)
        paths.append(str(path))
        by_frame[index] = detection
    calls = iter(range(len(frames)))

    class Stub:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, image):
            return [by_frame[next(calls)]]

    monkeypatch.setattr(pipeline, "ShapeDetector", Stub)
    arguments = teach_angle.parse_args(["--turn", "0", "--image", *paths,
                                        "--calibration", str(calibration), *extra])
    return teach_angle.measure(arguments)


@needs_frame
def test_measure_uses_live_pick_raw_angle(tmp_path, monkeypatch, cell):
    import live_pick
    image, detection = paint(cell, "triangle", 7.0)
    shape, raw, info = run_measure(tmp_path, monkeypatch, cell, [(image, detection)])
    assert shape == "triangle" and "template fit" in info
    # identical to what live_pick feeds into pick_angle
    row = live_pick.commands_for([detection], cell[1], 0.0, image)[0]
    assert row[2] == pytest.approx(raw)
    assert row[1].angle_deg == pytest.approx(pipeline.pick_angle("triangle", raw))
    assert raw == pytest.approx(7.0, abs=0.3)


@needs_frame
def test_measure_median_over_frames_and_spread_refusal(tmp_path, monkeypatch, cell):
    frames = [paint(cell, "square", a) for a in (7.0, 7.3, 6.8)]
    shape, raw, info = run_measure(tmp_path, monkeypatch, cell, frames)
    assert shape == "square" and raw == pytest.approx(7.0, abs=0.4) and "3 frame(s)" in info
    frames = [paint(cell, "square", a) for a in (7.0, 7.1, 12.0)]
    with pytest.raises(teach_angle.TeachError, match="spread"):
        run_measure(tmp_path, monkeypatch, cell, frames)


@needs_frame
def test_measure_refuses_fallback_unless_allowed(tmp_path, monkeypatch, cell):
    frames = [paint(cell, "triangle", 0.0, blob=True)]
    with pytest.raises(teach_angle.TeachError, match="fallback"):
        run_measure(tmp_path, monkeypatch, cell, frames)
    shape, _raw, info = run_measure(tmp_path, monkeypatch, cell, frames, ["--allow-fallback"])
    assert shape == "triangle" and "WARNING" in info


def test_frames_argument_checks(capsys):
    assert teach_angle.main(["--turn", "0", "--frames", "0"]) == 2
    assert teach_angle.main(["--turn", "0", "--image", "a.png", "--frames", "3"]) == 2
    assert teach_angle.parse_args(["--turn", "0", "--image", "a.png", "b.png"]).image == \
        ["a.png", "b.png"]


@needs_frame
def test_mould_window_shows_taught_turn(tmp_path, cell, monkeypatch):
    import live_view
    image, detection = paint(cell, "triangle", 7.0)
    values = teach_angle.empty()
    teach_angle.record(values, "triangle", 0.0, 30.0)
    teach_angle.record(values, "triangle", 20.0, 10.0)
    monkeypatch.setattr(pipeline, "TAUGHT_ZERO_DEG", {})
    assert "sign -1" in pipeline.apply_angle_teach(teach_angle.save(values, tmp_path / "t.json"))
    part = live_view.analyse_part(image, cell[1], detection)
    assert part.used == "template" and part.raw == pytest.approx(7.0, abs=0.5)
    assert part.sent == pytest.approx(-pipeline.fold_to_symmetry("triangle", part.raw - 30.0))
    texts = []
    monkeypatch.setattr(live_view.cv2, "putText",
                        lambda img, text, *a, **k: texts.append(text) or img)
    view = live_view.draw_mould(live_view.Result(image, cell[1], "test", [part]), 1.0)
    assert view.ndim == 3 and view.shape[1] == 2 * live_view.TILE
    assert any("taught zero theta0 +30.0" in t for t in texts)
    assert any(f"nAng {part.sent:+.1f}" in t for t in texts)
    # not taught -> old pocket comparison
    monkeypatch.setattr(pipeline, "TAUGHT_ZERO_DEG", {})
    texts.clear()
    live_view.draw_mould(live_view.Result(image, cell[1], "test", [part]), 1.0)
    assert any("turn to match pocket" in t for t in texts)


def test_apply_without_file_keeps_defaults(tmp_path):
    text = pipeline.apply_angle_teach(tmp_path / "missing.json")
    assert text == "angle teach: none (defaults: sign +1, offsets 0)"
    assert pipeline.ANGLE_SIGN == 1.0
    assert all(value == 0.0 for value in pipeline.SHAPE_OFFSET_DEG.values())
    assert pipeline.pick_angle("triangle", 25.9) == pytest.approx(25.9)


def test_apply_ignores_broken_or_incomplete(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert "IGNORED" in pipeline.apply_angle_teach(bad)
    values = teach_angle.empty()
    teach_angle.record(values, "star", 0.0, 10.0)
    half = teach_angle.save(values, tmp_path / "half.json")
    assert "IGNORED" in pipeline.apply_angle_teach(half)
    assert pipeline.ANGLE_SIGN == 1.0 and pipeline.SHAPE_OFFSET_DEG["star"] == 0.0


def test_cli_show_forget_and_refusal(tmp_path, monkeypatch, capsys):
    path = tmp_path / "teach.json"
    readings = iter([("hexagon", 4.0, "info"), ("hexagon", 4.0 + 13.0, "info")])
    monkeypatch.setattr(teach_angle, "measure", lambda arguments: next(readings))
    assert teach_angle.main(["--turn", "0", "--shape", "hexagon", "--file", str(path)]) == 0
    assert teach_angle.main(["--turn", "20", "--shape", "hexagon", "--file", str(path)]) == 1
    assert "REFUSED" in capsys.readouterr().err
    assert teach_angle.main(["--show", "--file", str(path)]) == 0
    assert "hexagon" in capsys.readouterr().out
    assert teach_angle.main(["--forget", "hexagon", "--file", str(path)]) == 0
    assert teach_angle.load(path)["shapes"] == {}


# Importing pipeline must not apply a teach file.
def test_not_applied_at_import():
    import subprocess
    import sys
    code = "import pipeline; print(pipeline.ANGLE_SIGN, pipeline.SHAPE_OFFSET_DEG['triangle'])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=pipeline.Path(pipeline.__file__).resolve().parent, check=True).stdout
    assert out.split() == ["1.0", "0.0"]


# ---------------------------------------------------------------- review fixes
from datetime import datetime, timedelta  # noqa: E402


@pytest.fixture
def teach_state(monkeypatch):
    monkeypatch.setattr(pipeline, "ANGLE_TEACH_STATE", "none")
    monkeypatch.setattr(pipeline, "ANGLE_TEACH_DETAIL", "")


def test_stale_theta0_refused():
    t0 = datetime(2026, 10, 30, 12, 0, 0)
    values = teach_angle.empty()
    teach_angle.record(values, "square", 0.0, 5.0, now=t0)
    with pytest.raises(teach_angle.TeachError, match="stale"):
        teach_angle.record(values, "square", 20.0, 25.0, now=t0 + timedelta(minutes=16))
    teach_angle.record(values, "square", 20.0, 25.0, now=t0 + timedelta(minutes=14))
    # theta0 now used: a second --turn 20 must not reuse it
    with pytest.raises(teach_angle.TeachError, match="already used"):
        teach_angle.record(values, "square", 20.0, 25.0, now=t0 + timedelta(minutes=14))


def test_turn0_invalidates_before_measuring(tmp_path, monkeypatch, capsys):
    path = tmp_path / "teach.json"
    values = teach_angle.empty()
    teach_angle.record(values, "star", 0.0, 10.0)
    teach_angle.record(values, "star", 20.0, 30.0)
    teach_angle.save(values, path)
    assert teach_angle.load(path)["sign"] == 1

    def refused(arguments):
        raise teach_angle.TeachError("camera: no Basler")
    monkeypatch.setattr(teach_angle, "measure", refused)
    assert teach_angle.main(["--turn", "0", "--shape", "star", "--file", str(path)]) == 1
    left = teach_angle.load(path)
    assert "star" not in left["shapes"] and left["sign"] is None
    assert "camera: no Basler" in capsys.readouterr().err
    # and the stale nonzero turn is refused before the camera is touched
    assert teach_angle.main(["--turn", "20", "--shape", "star", "--file", str(path)]) == 1


@pytest.mark.parametrize("shape", ANGLED)
def test_turn_guard_includes_residual(shape):
    half = pipeline.SYMMETRY_DEG[shape] / 2.0
    values = teach_angle.empty()
    teach_angle.record(values, shape, 0.0, 0.0)
    with pytest.raises(teach_angle.TeachError, match="ambiguous"):
        teach_angle.record(values, shape, half - teach_angle.MAX_RESIDUAL_DEG, 0.0)
    assert teach_angle.MAX_RESIDUAL_DEG == 3.0
    # 20 deg is fine for all, hexagon included (20 + 3 < 30)
    teach_angle.record(values, shape, 20.0, 20.0)


def test_theta0_is_circular_mean_across_fold():
    values = teach_angle.empty()
    # square period 90: theta0 +44.5, theta1 - 20 = -44.5 (= +45.5): mean +45 / -45
    teach_angle.record(values, "square", 0.0, 44.5)
    teach_angle.record(values, "square", 20.0, 44.5 + 21.0)
    theta0 = values["shapes"]["square"]["theta0"]
    assert abs(pipeline.fold_to_symmetry("square", theta0 - 45.0)) < 1e-9


def test_broken_json_is_a_clean_refusal(tmp_path, capsys):
    path = tmp_path / "teach.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(teach_angle.TeachError, match="not readable JSON"):
        teach_angle.load(path)
    assert teach_angle.main(["--show", "--file", str(path)]) == 1
    assert "REFUSED" in capsys.readouterr().err
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(teach_angle.TeachError):
        teach_angle.load(path)


def test_save_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "teach.json"
    teach_angle.save(teach_angle.empty(), path)

    def boom(*args):
        raise OSError("disk full")
    monkeypatch.setattr(teach_angle.os, "replace", boom)
    values = teach_angle.empty()
    values["turn_deg"] = 7.0
    with pytest.raises(OSError):
        teach_angle.save(values, path)
    assert teach_angle.load(path)["turn_deg"] == teach_angle.TEACH_TURN_DEG
    assert [p.name for p in tmp_path.iterdir()] == ["teach.json"]


def test_camera_error_becomes_teach_error(monkeypatch):
    import types
    import sys as system

    class Broken:
        def open(self):
            raise RuntimeError("No Basler camera detected.")

        def close(self):
            pass
    monkeypatch.setitem(system.modules, "camera", types.SimpleNamespace(BaslerSource=Broken))
    arguments = teach_angle.parse_args(["--turn", "0", "--shape", "star"])
    with pytest.raises(teach_angle.TeachError, match="camera: No Basler"):
        teach_angle.grab_frames(arguments)


@needs_frame
def test_measure_refuses_part_away_from_drop_point(tmp_path, monkeypatch, cell):
    frames = [paint(cell, "square", 7.0, centre=(-120.0 + 20.0, 60.0))]
    with pytest.raises(teach_angle.TeachError, match="from the drop point"):
        run_measure(tmp_path, monkeypatch, cell, frames)
    shape, _raw, info = run_measure(tmp_path, monkeypatch, cell, frames,
                                    ["--drop-x", "-100", "--drop-y", "60"])
    assert shape == "square" and "must match nTeachX/nTeachY" in info


def full_teach(tmp_path, shapes_k):
    values = teach_angle.empty()
    for shape, k in shapes_k:
        teach_angle.record(values, shape, 0.0, 5.0)
        teach_angle.record(values, shape, 20.0, 5.0 + 20.0 * k)
    return teach_angle.save(values, tmp_path / "teach.json")


def test_status_for_ignored_and_applied(tmp_path, teach_state):
    assert pipeline.apply_angle_teach(tmp_path / "none.json").startswith("angle teach: none")
    assert pipeline.ANGLE_TEACH_STATE == "none" and pipeline.send_blocked() is None
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    pipeline.apply_angle_teach(bad)
    assert pipeline.ANGLE_TEACH_STATE == "ignored" and "REFUSED" in pipeline.send_blocked()
    pipeline.apply_angle_teach(full_teach(tmp_path, [("square", 1), ("star", -1)]))
    assert pipeline.ANGLE_TEACH_STATE == "ignored" and "disagree" in pipeline.send_blocked()
    text = pipeline.apply_angle_teach(full_teach(tmp_path, [("square", 1)]))
    assert pipeline.ANGLE_TEACH_STATE == "applied" and pipeline.send_blocked() is None
    assert "UNTAUGHT" in text and "star" in text and "triangle" in text


def test_untaught_shape_under_active_sign(tmp_path, teach_state):
    pipeline.apply_angle_teach(full_teach(tmp_path, [("triangle", -1)]))
    assert pipeline.ANGLE_SIGN == -1.0
    assert pipeline.pick_blocked("triangle") is None
    assert pipeline.untaught("star") and "not taught" in pipeline.pick_blocked("star")
    assert pipeline.pick_blocked("circle") is None   # no angle, nothing to teach
    import live_pick
    star = (None, pipeline.PickCommand("star", 0.0, 0.0, 3.0), 0.0, 0.0)
    tri = (None, pipeline.PickCommand("triangle", 0.0, 0.0, 3.0), 0.0, 0.0)
    assert live_pick.row_flags(star, True) == ["UNTAUGHT"]
    assert live_pick.row_flags(tri, True) == []
    assert "not taught" in live_pick.send_refusal([(star, ["UNTAUGHT"])], True)
    # without a teach nothing is untaught
    pipeline.apply_angle_teach(tmp_path / "missing.json")
    assert not pipeline.untaught("star")


def test_fallback_flag_and_refusal(teach_state):
    import live_pick
    star = (None, pipeline.PickCommand("star", 0.0, 0.0, 3.0), 0.0, 0.0)
    circle = (None, pipeline.PickCommand("circle", 0.0, 0.0, 0.0), 0.0, 0.0)
    assert live_pick.row_flags(star, False) == ["FALLBACK"]
    assert live_pick.row_flags(circle, False) == []
    assert live_pick.row_flags(star, None) == []
    assert "FALLBACK" in live_pick.send_refusal([(star, ["FALLBACK"])], False)
    assert live_pick.send_refusal([(star, ["FALLBACK"])], True) is None
    assert "[FALLBACK]" in live_pick.format_row(
        (pipeline.Detection(3, 0.9, (0.0, 0.0), 0.0, ((0, 0),) * 4), star[1], 0.0, 0.0),
        ["FALLBACK"])


@needs_frame
def test_parts_mm_detail_matches_parts_mm(cell):
    image, detection = paint(cell, "triangle", 7.0)
    detail = cell[1].parts_mm_detail(detection, image)
    assert detail[:3] == cell[1].parts_mm(detection, image) and detail[3] is True
    blob, blob_detection = paint(cell, "triangle", 0.0, blob=True)
    assert cell[1].parts_mm_detail(blob_detection, blob)[3] is False


def test_live_pick_send_refusals(tmp_path, monkeypatch, capsys, teach_state):
    import live_pick
    assert live_pick.main(["--send", "--scalar", "--host", "1.2.3.4"]) == 2
    assert "--scalar cannot be used with --send" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    monkeypatch.setattr(pipeline.apply_angle_teach, "__defaults__", (bad,))
    assert live_pick.main(["--send", "--host", "1.2.3.4"]) == 2
    assert "REFUSED to send" in capsys.readouterr().err


@needs_frame
def test_live_pick_send_refuses_untaught_row(tmp_path, monkeypatch, capsys, cell, teach_state):
    import calibrate
    import cv2
    import live_pick
    image, detection = paint(cell, "square", 7.0)
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), image)
    calibration = calibrate.save_calibration(cell[1], tmp_path / "calibration.json")
    monkeypatch.setattr(pipeline.apply_angle_teach, "__defaults__",
                        (full_teach(tmp_path, [("star", 1)]),))

    class Stub:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, img):
            return [detection]
    monkeypatch.setattr(pipeline, "ShapeDetector", Stub)
    sent = []
    monkeypatch.setattr(live_pick, "send_rows", lambda *a: sent.append(a) or 0)
    common = ["--image", str(frame), "--calibration", str(calibration), "--no-overlay"]
    assert live_pick.main(common) == 0
    assert "[UNTAUGHT]" in capsys.readouterr().out
    assert live_pick.main([*common, "--send", "--host", "1.2.3.4"]) == 1
    assert "square is not taught" in capsys.readouterr().err and not sent
