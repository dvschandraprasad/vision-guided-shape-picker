"""Tests for the lab tool: no camera, no model, no display.

The calibration is synthetic - four pixel corners of a known square - so
the expected millimetres can be worked out by hand. The point of these tests is
the wiring: shape -> folded angle -> slot -> the exact line the robot receives.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

import calibrate
import live_pick
import pipeline

SIDE_PX = 1000.0          # the marker is 1000 px on a side, whatever it measures
HALF = SIDE_PX / 2.0
# The fake marker centre, in image pixels.
CENTRE = (2000.0, 1500.0)
TEN_MM_PX = calibrate.MARKER_MM / 10.0 / (calibrate.MARKER_MM / SIDE_PX)  # 100 px = a tenth of the marker


def square_corners(centre=CENTRE, half=HALF):
    """ArUco order for a marker whose printed top edge faces image-up."""
    x, y = centre
    return np.array(
        [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]],
        np.float64,
    )


@pytest.fixture
def plane() -> calibrate.PlaneCalibration:
    """The documented mount: +x is image right, +y is image up."""
    return calibrate.calibration_from_corners(square_corners())


@pytest.fixture
def small_plane() -> calibrate.PlaneCalibration:
    """A calibration small enough that its centre sits inside a 300x200 frame."""
    return calibrate.calibration_from_corners(square_corners((150.0, 100.0), 100.0))


def box(centre, angle_deg: float = 0.0, half: float = 100.0):
    """Four pixel corners of a square turned clockwise by `angle_deg` in the image."""
    x, y = centre
    theta = math.radians(angle_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    corners: tuple[tuple[float, float], ...] = tuple(
        (x + dx * cos - dy * sin, y + dx * sin + dy * cos)
        for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))
    )
    return corners


def detection(shape: str, centre, angle_deg: float = 0.0, conf: float = 0.9):
    """A detection whose corners are a square turned by `angle_deg` in the image."""
    return pipeline.Detection(
        class_id=pipeline.CLASSES.index(shape),
        confidence=conf,
        center=centre,
        angle_deg=angle_deg,
        corners=box(centre, angle_deg),
    )


def axes(plane: calibrate.PlaneCalibration):
    """The two mm vectors axis_summary prints, as numbers."""
    # Reads the two millimetre axes back from the printed text.
    numbers = [float(value) for value in re.findall(r"[-+]?\d+\.\d+", live_pick.axis_summary(plane))]
    assert len(numbers) == 4, live_pick.axis_summary(plane)
    return (numbers[0], numbers[1]), (numbers[2], numbers[3])


def test_a_point_one_marker_to_the_right_reads_one_marker_width(plane):
    assert plane.to_mm((CENTRE[0] + SIDE_PX, CENTRE[1])) == pytest.approx(
        (calibrate.MARKER_MM, 0.0), abs=1e-6)


def test_axis_summary_shows_the_documented_frame_for_a_square_mount(plane):
    """Marker top toward image-up: image right is +x, image down is -y."""
    right, down = axes(plane)
    hundred_px = calibrate.MARKER_MM / 10.0
    assert right == pytest.approx((hundred_px, 0.0), abs=0.05)
    assert down == pytest.approx((0.0, -hundred_px), abs=0.05)


def test_axis_summary_exposes_a_marker_taped_a_quarter_turn():
    """A rotated mount swaps the axes and the millimetres still look plausible."""
    turned = np.roll(square_corners(), 1, axis=0)
    right, down = axes(calibrate.calibration_from_corners(turned))
    hundred_px = calibrate.MARKER_MM / 10.0
    assert right == pytest.approx((0.0, -hundred_px), abs=0.05)
    assert down == pytest.approx((-hundred_px, 0.0), abs=0.05)


def test_commands_for_maps_centre_to_mm_and_folds_the_angle(plane):
    rows = live_pick.commands_for(
        [detection("square", (CENTRE[0] + SIDE_PX, CENTRE[1]), 30.0)], plane)
    # Each row is shape, command, raw angle, distance.
    _, command, raw, distance = rows[0]
    assert (command.x_mm, command.y_mm) == pytest.approx((calibrate.MARKER_MM, 0.0), abs=1e-6)
    assert command.z_mm == pipeline.PICK_Z_MM
    assert distance == pytest.approx(calibrate.MARKER_MM, abs=1e-6)
    assert command.angle_deg == pytest.approx(-30.0, abs=1e-4)
    assert raw == pytest.approx(-30.0, abs=1e-4), "image-clockwise reads negative in the y-up frame"


def test_a_circle_is_never_told_to_turn(plane):
    rows = live_pick.commands_for(
        [detection("circle", (CENTRE[0], CENTRE[1] - SIDE_PX), 37.0)], plane)
    assert rows[0][1].angle_deg == 0.0


def test_a_quarter_turned_square_is_asked_for_an_eighth_turn(plane):
    """A square turned 22.5 degrees in the image must be told -22.5, not 67.5."""
    rows = live_pick.commands_for([detection("square", CENTRE, 22.5)], plane)
    assert rows[0][1].angle_deg == pytest.approx(-22.5, abs=1e-6)


def test_a_triangle_sends_its_folded_heading(plane):
    """The outline heading replaced the box angle for 3, 5 and 6-fold shapes.

    In this fixture the box's raw heading is 90 degrees in the millimetre frame
    (y up, so it runs the other way from the image's 30), and 90 folds to 30 for a
    triangle's 120-degree symmetry.
    """
    rows = live_pick.commands_for([detection("triangle", CENTRE, 30.0)], plane)
    assert rows[0][1].angle_deg == pytest.approx(-30.0)


def test_the_taught_offset_shifts_the_square_before_the_fold(plane):
    """The offset corrects the measurement chain, so it goes on the raw angle.

    In this fixture the box's raw heading is 67.5 degrees: the millimetre frame is
    y up, so it runs the other way from the image's 22.5. The offset is added to
    that raw number and then folded, which is why -10 sends -32.5. The sign and
    size the robot needs are taught on the arm, never predicted (decision #17).
    """
    plain = live_pick.commands_for([detection("square", CENTRE, 22.5)], plane)
    shifted = live_pick.commands_for([detection("square", CENTRE, 22.5)], plane,
                                     angle_offset_deg=-10.0)
    # 1e-4 deg is the float32 homography's own round-off (0.0002 mm at 100 mm) and
    # the tolerance its sibling angle tests already use.
    assert plain[0][1].angle_deg == pytest.approx(-22.5, abs=1e-4)
    assert shifted[0][1].angle_deg == pytest.approx(-32.5, abs=1e-4)


def test_an_offset_past_the_half_period_folds_back(plane):
    """67.5 + 60 = 127.5, which is 127.5 - 90 = 37.5 for a square."""
    rows = live_pick.commands_for([detection("square", CENTRE, 22.5)], plane,
                                  angle_offset_deg=60.0)
    assert rows[0][1].angle_deg == pytest.approx(37.5, abs=1e-4)


def test_rows_are_sorted_by_distance_from_the_marker(plane):
    near = detection("square", (CENTRE[0] + 200.0, CENTRE[1]))
    far = detection("square", (CENTRE[0] + SIDE_PX, CENTRE[1]))
    rows = live_pick.commands_for([far, near], plane)
    # Rows come nearest first, by the distance field.
    assert [row[3] for row in rows] == sorted(row[3] for row in rows)


def test_format_row_carries_the_exact_wire_line(plane):
    rows = live_pick.commands_for([detection("star", (CENTRE[0] + SIDE_PX, CENTRE[1]))], plane)
    row = live_pick.format_row(rows[0])
    assert pipeline.encode_command(rows[0][1]) in row
    assert row.startswith("star")
    assert ";10.0;" in row


def test_draw_overlay_keeps_the_frame_size_and_does_not_touch_the_input(small_plane):
    image = np.zeros((200, 300, 3), np.uint8)
    rows = live_pick.commands_for([detection("hexagon", (150.0, 100.0))], small_plane)
    # Drawing returns a new frame and leaves the input alone.
    overlay = live_pick.draw_overlay(image, rows)
    assert overlay.shape == image.shape
    assert overlay.any()
    assert not image.any()


def test_a_gray_frame_is_widened_to_colour_for_the_overlay(small_plane):
    gray = np.zeros((200, 300), np.uint8)
    rows = live_pick.commands_for([detection("square", (150.0, 100.0))], small_plane)
    assert live_pick.draw_overlay(gray, rows).shape == (200, 300, 3)


def test_an_empty_detection_list_draws_nothing(small_plane):
    image = np.zeros((200, 300, 3), np.uint8)
    assert not live_pick.draw_overlay(image, live_pick.commands_for([], small_plane)).any()


# --- failing loudly on a fresh PC ---------------------------------------------
def test_send_without_a_host_names_the_address_to_pass(capsys):
    # With no host, the tool must stop and explain.
    code = live_pick.main(["--send", "--image", "frame.png"])

    printed = capsys.readouterr().err
    assert code == 2
    assert "--host" in printed
    assert "this PC" in printed
    assert "192.168.125.x" in printed


def test_a_missing_calibration_says_which_command_to_run(tmp_path, capsys):
    code = live_pick.main(["--image", "frame.png",
                           "--calibration", str(tmp_path / "absent.json")])

    printed = capsys.readouterr().err
    assert code == 2
    assert "no usable calibration" in printed
    assert "calibrate.py" in printed


def test_a_calibration_from_before_the_upright_fix_is_refused_not_used(tmp_path, capsys):
    payload = calibrate.calibration_from_corners(square_corners()).to_dict()
    # An old calibration file without this field is refused.
    del payload["frame_upright"]
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(payload), encoding="utf-8")

    code = live_pick.main(["--image", "frame.png", "--calibration", str(stale)])

    printed = capsys.readouterr().err
    assert code == 2
    assert "frame_upright" in printed
