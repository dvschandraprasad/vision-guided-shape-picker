"""Tests for calibrate.py: marker -> millimetres, no camera and no display.

Every frame here is synthetic: a real ArUco marker is generated with OpenCV,
optionally perspective-warped, and pasted into a blank image. So the tests prove
the detection-to-homography chain on data a real camera can produce, without a
Basler attached.

Run:
    .venv/Scripts/python.exe -m pytest test_calibrate.py -q
"""

from __future__ import annotations

import dataclasses
import json
import math

import cv2
import numpy as np
import pytest

import calibrate
import pipeline


def marker_patch(px: int = 400) -> np.ndarray:
    """A generated DICT_5X5_100 id-0 marker with the white quiet zone it needs."""
    dictionary = cv2.aruco.getPredefinedDictionary(calibrate.MARKER_DICT)
    marker = cv2.aruco.generateImageMarker(dictionary, calibrate.MARKER_ID, px)
    border = px // 8
    return cv2.copyMakeBorder(marker, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=255)


def marker_frame(quad: np.ndarray | None = None, size: int = 900,
                 turns: int = 0) -> np.ndarray:
    """Paste the marker at `quad` (4 pixel corners) into a blank white frame.

    `turns` rotates the printed pattern before it is pasted, which is what a marker
    stuck down a quarter turn looks like to the camera.
    """
    patch = np.ascontiguousarray(np.rot90(marker_patch(), turns))
    height, width = patch.shape[:2]
    source = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    if quad is None:
        half, centre = 150.0, size / 2.0
        quad = np.array([[centre - half, centre - half], [centre + half, centre - half],
                         [centre + half, centre + half], [centre - half, centre + half]],
                        np.float32)
    quad = np.asarray(quad, np.float32).reshape(4, 2)
    frame = np.full((size, size), 255, np.uint8)
    warped = cv2.warpPerspective(
        patch, cv2.getPerspectiveTransform(source, quad), (size, size),
        flags=cv2.INTER_NEAREST, borderValue=255,
    )
    frame = np.minimum(frame, warped)
    # Three colour channels, as the camera gives.
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


# The quad is where the printed *patch* lands; ArUco reports the black border
# inside it, so detected corners sit a few percent inwards. Tests therefore check
# the centre and the self-consistent mapping, never absolute corner coordinates.
SQUARE_QUAD = np.array([[300, 300], [600, 300], [600, 600], [300, 600]], np.float32)
SKEWED_QUAD = np.array([[330, 300], [610, 320], [570, 620], [300, 585]], np.float32)
CENTRE = (450.0, 450.0)


# --- detection ---------------------------------------------------------------
# Finding the marker must give four pixel corners.
def test_find_marker_returns_four_corners_of_the_generated_marker():
    corners = calibrate.find_marker(marker_frame(SQUARE_QUAD))

    assert corners.shape == (4, 2)
    centre = corners.mean(axis=0)
    assert math.dist((float(centre[0]), float(centre[1])), CENTRE) < 4.0
    assert all(0.0 <= x <= 900.0 and 0.0 <= y <= 900.0 for x, y in corners)


def test_a_blank_frame_raises_instead_of_returning_none():
    with pytest.raises(calibrate.MarkerNotFound):
        calibrate.find_marker(np.full((400, 400, 3), 255, np.uint8))


def test_an_empty_array_raises():
    with pytest.raises(calibrate.MarkerNotFound):
        calibrate.find_marker(np.zeros((0, 0, 3), np.uint8))


def test_wrong_marker_id_raises_and_names_what_it_saw():
    with pytest.raises(calibrate.MarkerNotFound, match=r"marker id 7 not in this frame"):
        calibrate.find_marker(marker_frame(SQUARE_QUAD), marker_id=7)


def test_greyscale_input_is_accepted():
    frame = marker_frame(SQUARE_QUAD)
    corners = calibrate.find_marker(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    centre = corners.mean(axis=0)
    assert math.dist((float(centre[0]), float(centre[1])), CENTRE) < 4.0


# --- the homography ----------------------------------------------------------
def test_corners_round_trip_onto_the_marker_square():
    calibration = calibrate.calibration_from_corners(SQUARE_QUAD)
    mapped = calibration.corners_mm(SQUARE_QUAD)

    half = calibrate.MARKER_MM / 2.0
    expected = np.array([[half, half], [-half, half], [-half, -half], [half, -half]])
    # Order above is y-down image order (TL,TR,BR,BL) mapped with y up, so the
    # expected square is the same quad read clockwise: compare as a set.
    assert np.allclose(sorted(mapped.round(3).tolist()), sorted(expected.round(3).tolist()),
                       atol=0.01)


def test_the_default_marker_is_the_measured_print_not_the_nominal_100():
    # 2026-10-28: a steel rule across the fresh sheet's black square, all four
    # edges. The old print scaled to 97 mm; the number is measured, not assumed.
    assert calibrate.MARKER_MM == 100.0
    calibration = calibrate.calibration_from_corners(SQUARE_QUAD)

    assert calibration.marker_mm == 100.0


def test_a_skewed_marker_still_maps_its_corners_onto_a_square():
    calibration = calibrate.calibration_from_corners(SKEWED_QUAD)
    mapped = calibration.corners_mm(SKEWED_QUAD)
    sides_mm = [float(np.linalg.norm(mapped[(index + 1) % 4] - mapped[index])) for index in range(4)]

    assert max(sides_mm) - min(sides_mm) < 0.05, f"sides still trapezoidal: {sides_mm}"


def test_the_scalar_ratio_and_the_homography_agree_on_an_unskewed_marker():
    calibration = calibrate.calibration_from_corners(SQUARE_QUAD)
    scalar = calibration.as_scalar_calibration()
    mapper = pipeline.CoordinateMapper(scalar)

    for point in [(450.0, 450.0), (500.0, 430.0), (400.0, 470.0)]:
        homography_mm = calibration.to_mm(point)
        scalar_mm = mapper.to_mm(point)
        assert math.dist(homography_mm, scalar_mm) < 0.5, f"{point}: {homography_mm} vs {scalar_mm}"


def test_the_scalar_ratio_and_the_homography_disagree_on_a_skewed_marker():
    calibration = calibrate.calibration_from_corners(SKEWED_QUAD)
    mapper = pipeline.CoordinateMapper(calibration.as_scalar_calibration())

    # A corner of the frame is where perspective bites hardest.
    point = (700.0, 700.0)
    homography_mm = calibration.to_mm(point)
    scalar_mm = mapper.to_mm(point)
    assert math.dist(homography_mm, scalar_mm) > 1.0, (
        "if these ever agree on a trapezoid, the homography is not doing its job"
    )


# One frame and the marker size give the scale.
def test_marker_scale_needs_only_a_frame():
    calibration = calibrate.marker_scale(marker_frame(SKEWED_QUAD))

    assert isinstance(calibration, calibrate.PlaneCalibration)
    # The marker is drawn about 230 px wide, so 100 mm / ~230 px must land here.
    assert 0.2 < calibration.mm_per_px() < 0.6
    assert calibration.mm_per_px() == pytest.approx(
        calibrate.MARKER_MM / float(np.mean(calibration.sides_px))
    )


def test_a_zero_marker_size_is_rejected():
    with pytest.raises(ValueError, match="marker_mm must be positive"):
        calibrate.calibration_from_corners(SQUARE_QUAD, marker_mm=0.0)


def test_degenerate_corners_are_rejected():
    collapsed = np.array([[10, 10], [10, 10], [10, 10], [10, 10]], np.float32)
    with pytest.raises(ValueError, match="degenerate"):
        calibrate.calibration_from_corners(collapsed)


def test_the_matrix_is_exactly_nine_numbers_and_rebuilds_to_3x3():
    calibration = calibrate.calibration_from_corners(SQUARE_QUAD)

    assert len(calibration.matrix) == 9
    assert all(isinstance(value, float) for value in calibration.matrix)
    assert calibration.matrix_3x3.shape == (3, 3)


def test_a_bad_matrix_is_rejected_at_construction():
    with pytest.raises(ValueError, match="9 numbers"):
        calibrate.PlaneCalibration((1.0, 2.0), 97.0, (0.0, 0.0), 0.0, (1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="singular"):
        calibrate.PlaneCalibration((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                                   97.0, (0.0, 0.0), 0.0, (1.0, 1.0, 1.0, 1.0))


def test_the_record_cannot_be_edited():
    calibration = calibrate.calibration_from_corners(SQUARE_QUAD)

    with pytest.raises(dataclasses.FrozenInstanceError):
        calibration.marker_mm = 100.0  # type: ignore[misc]


# --- angles ------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, angle, expected",
    [
        ("square", 95.0, 5.0),
        ("hexagon", 65.0, 5.0),
        ("triangle", 125.0, 5.0),
        ("star", 77.0, 5.0),
        ("square", 5.0, 5.0),
        ("hexagon", 115.0, -5.0),
        ("triangle", 235.0, -5.0),
        ("star", 149.0, 5.0),
        ("circle", 33.0, 0.0),
        ("circle", 180.0, 0.0),
    ],
)
def test_folds_follow_the_symmetry_table(shape, angle, expected):
    assert calibrate.fold_to_symmetry(shape, angle) == pytest.approx(expected, abs=1e-9)


def test_a_180_degree_triangle_is_not_folded_away():
    # 180 is not a multiple of 120, so this must NOT become 0: the ambiguity is real.
    assert calibrate.fold_to_symmetry("triangle", 180.0) == pytest.approx(-60.0)


def test_an_unknown_shape_is_rejected():
    with pytest.raises(ValueError, match="unknown shape"):
        calibrate.fold_to_symmetry("octagon", 10.0)


def box_px(calibration, angle_deg, centre_mm=(10.0, 10.0), half=12.0):
    """A square box built in *millimetres* at `angle_deg`, pushed back to pixels.

    Building it in mm space is the point: the pixel box is deliberately not
    axis-aligned, so only a mapper that removes the skew can recover the angle.
    """
    theta = math.radians(angle_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    offsets = [(-half, -half), (half, -half), (half, half), (-half, half)]
    mm = np.array(
        [[centre_mm[0] + x * cos - y * sin, centre_mm[1] + x * sin + y * cos] for x, y in offsets],
        np.float64,
    )
    inverse = np.linalg.inv(calibration.matrix_3x3)
    return cv2.perspectiveTransform(mm.reshape(-1, 1, 2), inverse).reshape(-1, 2)


@pytest.mark.parametrize("angle", [0.0, 5.0, -20.0, 30.0])
def test_the_angle_is_measured_after_the_skew_is_removed(angle):
    calibration = calibrate.calibration_from_corners(SKEWED_QUAD)

    assert calibration.angle_deg(box_px(calibration, angle)) == pytest.approx(angle, abs=1.0)


def test_a_box_aligned_in_millimetres_is_not_tilted_by_the_camera():
    calibration = calibrate.calibration_from_corners(SKEWED_QUAD)
    box = box_px(calibration, 0.0)

    # Corner order is TL,TR,BR,BL, so the pixel box is visibly tilted...
    top = box[1] - box[0]
    assert abs(math.degrees(math.atan2(float(top[1]), float(top[0])))) > 1.0
    # ...and the homography still reports 0.
    assert calibration.angle_deg(box) == pytest.approx(0.0, abs=0.5)


# --- the frame is self-correcting -------------------------------------------
def axes_mm(calibration, step_px: float = 100.0):
    """Where the image's right and down directions point, in millimetres."""
    centre_x, centre_y = calibration.centre_px
    return (calibration.to_mm((centre_x + step_px, centre_y)),
            calibration.to_mm((centre_x, centre_y + step_px)))


def is_upright(calibration) -> bool:
    """+x is the right of the image and +y is up, within 45 degrees."""
    right, down = axes_mm(calibration)
    return (right[0] > abs(right[1])) and (-down[1] > abs(down[0]))


def test_the_documented_mount_needs_no_correction():
    calibration = calibrate.calibration_from_corners(SQUARE_QUAD, upright=True)

    assert calibration.frame_turns == 0
    assert is_upright(calibration)


@pytest.mark.parametrize("turns", [1, 2, 3])
# A marker taped a quarter turn still gives the right frame.
def test_a_marker_taped_a_quarter_turn_is_corrected(turns):
    frame = marker_frame(SQUARE_QUAD, turns=turns)
    literal = calibrate.calibration_from_corners(calibrate.find_marker(frame))

    assert not is_upright(literal), "the tape should visibly turn the literal frame"
    corrected = calibrate.marker_scale(frame)
    assert is_upright(corrected), "marker_scale must hand back the documented frame"
    assert corrected.frame_turns != 0


def test_a_marker_placed_upside_down_needs_exactly_two_turns():
    corrected = calibrate.marker_scale(marker_frame(SQUARE_QUAD, turns=2))

    assert corrected.frame_turns == 2
    assert is_upright(corrected)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_the_correction_is_idempotent(turns):
    corners = calibrate.find_marker(marker_frame(SQUARE_QUAD, turns=turns))

    assert calibrate.upright_turns(calibrate.upright_corners(corners)) == 0


def test_a_slight_camera_skew_survives_the_correction():
    """Only quarter turns go; a real tilt must stay or the mapping would lie."""
    calibration = calibrate.marker_scale(marker_frame(SKEWED_QUAD))

    assert calibration.frame_turns == 0
    assert is_upright(calibration)


# --- files -------------------------------------------------------------------
# Saving then loading must give the same calibration back.
def test_calibration_round_trips_through_json(tmp_path):
    original = calibrate.calibration_from_corners(SKEWED_QUAD)
    path = calibrate.save_calibration(original, tmp_path / "calibration.json")

    loaded = calibrate.load_calibration(path)

    assert loaded == original
    assert loaded.to_mm((500.0, 500.0)) == pytest.approx(original.to_mm((500.0, 500.0)))


# Missing calibration must never invent a scale.
def test_a_missing_file_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="no calibration at"):
        calibrate.load_calibration(tmp_path / "absent.json")


def test_broken_json_fails_closed(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        calibrate.load_calibration(path)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"method": "homography"},
        {"method": "magic", "marker_mm": 97, "matrix": [1] * 9,
         "centre_px": [0, 0], "rotation_deg": 0, "sides_px": [1, 1, 1, 1],
         "frame_upright": True},
        {"method": "homography", "marker_mm": 97, "matrix": [0] * 9,
         "centre_px": [0, 0], "rotation_deg": 0, "sides_px": [1, 1, 1, 1],
         "frame_upright": True},
        {"method": "homography", "marker_mm": 0, "matrix": [1] * 9,
         "centre_px": [0, 0], "rotation_deg": 0, "sides_px": [1, 1, 1, 1],
         "frame_upright": True},
        {"method": "homography", "marker_mm": 97, "matrix": [1] * 9,
         "centre_px": [0, 0], "rotation_deg": 0, "sides_px": [0, 1, 1, 1],
         "frame_upright": True},
    ],
)
def test_malformed_calibration_json_fails_closed(tmp_path, payload):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        calibrate.load_calibration(path)


def test_the_saved_json_is_readable_and_names_the_method(tmp_path):
    path = calibrate.save_calibration(
        calibrate.calibration_from_corners(SQUARE_QUAD), tmp_path / "calibration.json"
    )
    values = json.loads(path.read_text(encoding="utf-8"))

    assert values["method"] == "homography"
    assert values["marker_mm"] == calibrate.MARKER_MM
    assert len(values["matrix"]) == 9
    assert values["frame_upright"] is True


def test_the_saved_json_records_the_quarter_turns_it_removed(tmp_path):
    calibration = calibrate.marker_scale(marker_frame(SQUARE_QUAD, turns=1))
    path = calibrate.save_calibration(calibration, tmp_path / "calibration.json")
    values = json.loads(path.read_text(encoding="utf-8"))

    assert values["frame_turns"] == calibration.frame_turns
    assert calibrate.load_calibration(path) == calibration


def test_a_calibration_file_without_the_upright_flag_fails_closed(tmp_path):
    """Old files may be a quarter turn out, so they are refused, not trusted."""
    payload = calibrate.calibration_from_corners(SQUARE_QUAD).to_dict()
    del payload["frame_upright"]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Re-run calibrate.py"):
        calibrate.load_calibration(path)


def test_frame_turns_outside_the_four_quarter_turns_is_rejected():
    with pytest.raises(ValueError, match="frame_turns"):
        dataclasses.replace(calibrate.calibration_from_corners(SQUARE_QUAD), frame_turns=4)


# --- command line ------------------------------------------------------------
# The command line turns one saved frame into a file.
def test_the_cli_writes_a_calibration_from_a_saved_frame(tmp_path):
    frame_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(frame_path), marker_frame(SKEWED_QUAD))
    out = tmp_path / "calibration.json"

    assert calibrate.main([str(frame_path), "--out", str(out)]) == 0

    calibration = calibrate.load_calibration(out)
    assert calibration.marker_mm == calibrate.MARKER_MM


def test_the_cli_names_the_folder_and_the_full_path_when_the_file_is_missing(
    tmp_path, capsys, monkeypatch
):
    """A bare "cannot read x.png" cost a lab session; the message must locate the file."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as raised:
        calibrate.main([str(tmp_path / "not_here.png")])

    assert raised.value.code == 2
    message = capsys.readouterr().err
    assert "folder    :" in message
    assert str(tmp_path) in message  # the folder we actually looked in
    assert "exists    : False" in message
    assert "no such file here" in message
    assert "--live" in message  # the command that works with the camera connected


def test_the_cli_says_so_when_the_file_is_there_but_cannot_be_decoded(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    broken = tmp_path / "empty.png"
    broken.write_bytes(b"")

    with pytest.raises(SystemExit):
        calibrate.main([str(broken)])

    message = capsys.readouterr().err
    assert "exists    : True" in message
    assert "OneDrive" in message


def test_the_cli_defaults_to_the_measured_marker_size(tmp_path, capsys):
    frame_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(frame_path), marker_frame(SQUARE_QUAD))
    out = tmp_path / "calibration.json"

    calibrate.main([str(frame_path), "--out", str(out)])

    printed = capsys.readouterr().out
    assert "100.00 mm" in printed
    assert "0.178" not in printed  # no camera-specific number is hardcoded


# --- The pick point from the part's own outline ----------------------------
# The box centre is exact for square, hexagon and circle because they are
# centrally symmetric. A triangle's is not: measured on these synthetic masks it
# sits about 5.8 mm from the true centre at the 100 mm marker scale - more than
# any pocket clearance we have measured.
PX_PER_MM = 0.219081      # the laptop frame's scale; only the ratios matter here


def flat_plane() -> calibrate.PlaneCalibration:
    """A pure scale plane: 0.219 mm per pixel, y up, origin at (500, 400) px."""
    matrix = (PX_PER_MM, 0.0, 0.0,
              0.0, -PX_PER_MM, 0.0,
              0.0, 0.0, 1.0)
    return calibrate.PlaneCalibration(matrix, 100.0, (500.0, 400.0), 0.0,
                                      (1.0, 1.0, 1.0, 1.0))


def synthetic_part(vertices, size=(800, 900)):
    """A filled dark part on white paper, plus its min-area box and its centroid."""
    image = np.full((size[0], size[1], 3), 255, np.uint8)
    polygon = np.asarray(np.round(vertices), np.int32)
    cv2.fillPoly(image, [polygon], (120, 60, 20))     # a saturated, dark BGR part
    (box_x, box_y), (width, height), theta = cv2.minAreaRect(polygon)
    corners = cv2.boxPoints(((box_x, box_y), (width, height), theta))
    moments = cv2.moments(polygon)
    centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    return image, (box_x, box_y), tuple(map(tuple, corners)), centroid


# Three corners on a circle, 120 degrees apart.
def triangle_vertices(turn_deg: float = 0.0, radius: float = 90.0,
                      centre=(500.0, 400.0)):
    return [(centre[0] + radius * math.cos(math.radians(turn_deg + 90 + 120 * k)),
             centre[1] + radius * math.sin(math.radians(turn_deg + 90 + 120 * k)))
            for k in range(3)]


# Wrap a box as a detection the pipeline can refine.
def triangle_detection(box_centre, corners) -> pipeline.Detection:
    return pipeline.Detection(pipeline.CLASSES.index("triangle"), 0.9, box_centre,
                              0.0, corners)


# The box centre is off for a triangle; outline fixes it.
def test_the_box_centre_sits_five_millimetres_off_a_triangle():
    plane = flat_plane()
    image, box_centre, corners, centroid = synthetic_part(triangle_vertices())
    detection = triangle_detection(box_centre, corners)

    assert math.dist(box_centre, centroid) * PX_PER_MM > 4.0   # the defect is real
    centre, _tip = pipeline.refine(image, detection)
    assert math.dist(centre, centroid) < 1.0                   # a pixel, not five mm

    x_mm, y_mm, _raw = plane.parts_mm(detection, image)
    wanted = plane.to_mm(centroid)
    assert math.dist((x_mm, y_mm), wanted) < 1.0
    box_mm = plane.parts_mm(detection)                          # no pixels, no fix
    assert math.dist(box_mm[:2], wanted) > 4.0


def test_the_tip_heading_turns_with_the_triangle():
    first = None
    for turn in (0.0, 25.0, 50.0):
        image, box_centre, corners, _centroid = synthetic_part(triangle_vertices(turn))
        _centre, tip = pipeline.refine(image, triangle_detection(box_centre, corners))
        assert tip is not None
        if first is None:
            first = tip
            continue
        # a tip and its neighbours are 120 degrees apart, so the advance is mod 120
        assert abs(((tip - first) % 120.0) - turn % 120.0) < 2.0


def test_a_square_pick_point_is_already_the_outline_centre():
    half = 91.0
    vertices = [(500 + x, 400 + y)
                for x, y in ((-half, -half), (half, -half), (half, half), (-half, half))]
    image, box_centre, corners, centroid = synthetic_part(vertices)
    centre, tip = pipeline.refine(image, pipeline.Detection(
        pipeline.CLASSES.index("square"), 0.9, box_centre, 0.0, corners))
    assert math.dist(centre, centroid) < 1.0
    assert math.dist(centre, box_centre) < 1.0
    assert tip is not None


def test_refine_falls_back_when_the_crop_is_bare_paper():
    plane = flat_plane()
    blank = np.full((300, 300, 3), 255, np.uint8)
    detection = pipeline.Detection(3, 0.9, (150.0, 150.0), 0.0,
                                   ((100.0, 100.0), (200.0, 100.0),
                                    (200.0, 200.0), (100.0, 200.0)))
    assert pipeline.refine(blank, detection) == ((150.0, 150.0), None)
    x_mm, y_mm, raw = plane.parts_mm(detection, blank)
    assert (x_mm, y_mm) == plane.to_mm((150.0, 150.0))
    assert math.isfinite(raw)


def shifted_refine(_image, _detection):
    """Pretend the outline measured a point 60 px to the right."""
    return (560.0, 400.0), 12.0


# A fixed square detection for the pick point tests.
def square_detection() -> pipeline.Detection:
    return pipeline.Detection(pipeline.CLASSES.index("square"), 0.9, (500.0, 400.0), 0.0,
                              ((409.0, 309.0), (591.0, 309.0),
                               (591.0, 491.0), (409.0, 491.0)))


def test_a_symmetric_shape_keeps_the_box_centre(monkeypatch):
    # On a real frame the outline moved a square by up to 1.8 mm, because a
    # highlight splits the mask and half a part has the wrong centroid. Its box
    # centre is exact, so the pixels are ignored for square, hexagon and circle.
    plane = flat_plane()
    monkeypatch.setattr(pipeline, "refine", shifted_refine)
    frame = np.zeros((800, 900, 3), np.uint8)
    x_mm, y_mm, _raw = plane.parts_mm(square_detection(), frame)
    assert (x_mm, y_mm) == plane.to_mm((500.0, 400.0))
    # the scalar path has its own scale (a plane with unit sides says nothing
    # about millimetres), so compare against its own to_mm, not the plane's
    mapper = pipeline.CoordinateMapper(plane.as_scalar_calibration())
    command = mapper.to_pick_command(square_detection(), image=frame)
    centre_mm = mapper.to_mm(square_detection().center)
    assert command.x_mm == pytest.approx(centre_mm[0], abs=0.06)
    assert command.y_mm == pytest.approx(centre_mm[1], abs=0.06)


def test_an_asymmetric_shape_follows_the_outline(monkeypatch):
    plane = flat_plane()
    monkeypatch.setattr(pipeline, "refine", shifted_refine)
    frame = np.zeros((800, 900, 3), np.uint8)
    image, box_centre, corners, _centroid = synthetic_part(triangle_vertices())
    detection = triangle_detection(box_centre, corners)
    x_mm, y_mm, _raw = plane.parts_mm(detection, frame)
    assert (x_mm, y_mm) == plane.to_mm((560.0, 400.0))
    assert image.shape[0] == 800   # the mask is unused here, only the centre matters


# Ten corners, long and short radius in turn.
def star_vertices(turn_deg: float = 0.0, radius: float = 90.0, centre=(500.0, 400.0)):
    return [(centre[0] + (radius if k % 2 == 0 else 0.45 * radius)
             * math.cos(math.radians(turn_deg + 36 * k)),
             centre[1] + (radius if k % 2 == 0 else 0.45 * radius)
             * math.sin(math.radians(turn_deg + 36 * k))) for k in range(10)]


@pytest.mark.parametrize("shape", ["triangle", "star"])
def test_a_three_or_five_fold_heading_survives_the_whole_turn(shape):
    # direction_deg used to wrap by 180. For a triangle that is half its 120 degree
    # period and for a star half its 72, so on part of every turn the robot was
    # sent the worst possible angle - the reason those two missed the mould while
    # square and hexagon (180 = 2 x 90 = 3 x 60) fit. Sweep the full circle.
    plane = flat_plane()
    draw = triangle_vertices if shape == "triangle" else star_vertices
    phase = 90.0 if shape == "triangle" else 0.0     # where vertex 0 sits at turn 0
    worst = 0.0
    for turn in range(0, 360, 15):
        image, box_centre, corners, _centroid = synthetic_part(draw(float(turn)))
        detection = pipeline.Detection(pipeline.CLASSES.index(shape), 0.9, box_centre,
                                       0.0, corners)
        sent = plane.to_pick_command(detection, image=image).angle_deg
        truth = pipeline.pick_angle(shape, -(turn + phase))   # image y is flipped
        worst = max(worst, pipeline.fold_to_symmetry(shape, sent - truth))
    assert worst < 2.0


def test_a_star_is_measured_even_when_the_border_is_not_sheet(monkeypatch):
    # On the real frame both stars had a neighbour lying on the crop's 3 px border,
    # so the "background" was not the sheet and the first mask came out as one blob
    # touching the border - unusable. The second try estimates the background
    # outside the box instead. Force the first try to fail and the star must still
    # be measured, or those parts fall back to a box angle that was never usable.
    image, box_centre, corners, _centroid = synthetic_part(star_vertices(10.0))
    detection = pipeline.Detection(pipeline.CLASSES.index("star"), 0.9, box_centre,
                                   0.0, corners)
    # A border ring and nothing else: unusable however it is clipped to the box,
    # which is what the real frame produced. (A full-crop mask no longer works as
    # a double - clipped to the box it becomes the box rectangle, which passes the
    # fill gate, so the second try was never reached and the test proved nothing.)
    ring = np.zeros(image.shape[:2], np.uint8)
    ring[:3] = ring[-3:] = 1
    ring[:, :3] = ring[:, -3:] = 1
    monkeypatch.setattr(pipeline, "foreground_mask", lambda *_args: ring)
    centre, heading = pipeline.refine(image, detection)
    assert heading is not None
    assert math.dist(centre, (500.0, 400.0)) < 5.0
