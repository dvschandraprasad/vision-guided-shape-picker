"""Tests for the coordinate check tool: no camera, no model, no robot.

The point of these tests is the two things a wrong answer would ruin: the table a
person reads and the RAPID module the robot runs. Both are text, so both are
checked as text - no inference, no motion.
"""

from __future__ import annotations

import json
import re

import pytest

import calibrate
import live_pick
import test_live_pick as lab
import verify_pos_code
import verify_positions


@pytest.fixture
def plane() -> calibrate.PlaneCalibration:
    """The documented mount: +x is image right, +y is image up."""
    return calibrate.calibration_from_corners(lab.square_corners())


@pytest.fixture
def rows(plane):
    """Three shapes: one on the marker, one right and up, one straight up."""
    detections = [
        lab.detection("circle", lab.CENTRE, 0.0, 0.90),
        lab.detection("square", (lab.CENTRE[0] + 1000.0, lab.CENTRE[1] - 1000.0), 0.0, 0.80),
        lab.detection("triangle", (lab.CENTRE[0], lab.CENTRE[1] - 2000.0), 30.0, 0.70),
    ]
    # Turn the shapes into table rows using the real picker.
    return live_pick.commands_for(detections, plane)


# Pull out only the CheckPoint lines from the RAPID text.
def checkpoint_lines(module: str) -> list[str]:
    return [line.strip() for line in module.splitlines() if "CheckPoint \"" in line]


def test_the_table_names_shape_slot_millimetres_and_the_wire_line(rows):
    table = verify_positions.format_table(rows)

    assert "shape" in table.splitlines()[0]
    assert "circle" in table
    # the square sits right of and above the marker: +x is right, +y is up
    assert "2;100.0;100.0;10.0;0.0" in table
    # the triangle is not allowed to turn the robot (decision #27): 0.0 in the angle
    # field, while its slot stays 4 - shape and heading are separate fields.
    assert "4;0.0;200.0;10.0;-30.0" in table   # a triangle sends its signed folded heading


def test_the_table_is_ordered_nearest_to_the_marker_first(rows):
    table = verify_positions.format_table(rows)
    # Read the shape name column to check table order.
    order = [line.split()[1] for line in table.splitlines()[1:]]

    assert order == ["circle", "square", "triangle"]


def test_the_rapid_module_is_a_module_with_one_line_per_shape(rows):
    module = verify_positions.rapid_module(rows, "23-10-2026 20:00")

    assert module.startswith("MODULE VisionCheck\n")
    assert module.rstrip().endswith("ENDMODULE")
    assert "PROC CheckPoint(string sName, num nX, num nY, num nAng)" in module
    assert "PROC CheckAll()" in module
    assert len(checkpoint_lines(module)) == len(rows)


def test_the_rapid_module_builds_the_point_from_the_numbers_not_pickbase(rows):
    """updated_robo.mod keeps no pickBase, so the check must not need one."""
    module = verify_positions.rapid_module(rows, "now")

    assert "pickBase" not in module
    assert "pTarget := [[nX, nY, nSafeZ], oPick, cnfPick, ejPick];" in module
    assert "pTarget := RelTool([[nX, nY, nPartZ], oPick, cnfPick, ejPick]," in module
    assert module.count("toolss\\WObj:=wobjS") == 3
    assert "nPartZ := 10.0" in module  # the part height travels in the message
    assert "nSafeZ := 50.0" in module


def test_every_number_in_a_call_carries_a_decimal_point(rows):
    """RAPID needs 84.4, not 84 - and a missing dot is a syntax error in the cell."""
    module = verify_positions.rapid_module(rows, "now")
    pattern = r'^CheckPoint "\w+", -?\d+\.\d, -?\d+\.\d, -?\d+\.\d;$'

    for line in checkpoint_lines(module):
        assert re.match(pattern, line), line


def test_the_calls_name_the_shape_and_the_angle_it_sends(rows):
    module = verify_positions.rapid_module(rows, "now")

    checkpoints = checkpoint_lines(module)
    assert checkpoints[0] == 'CheckPoint "circle", 0.0, 0.0, 0.0;'
    assert checkpoints[2] == 'CheckPoint "triangle", 0.0, 200.0, -30.0;'


def test_a_square_keeps_its_angle_in_the_module(plane):
    """The check module must carry the same angle a pick would (decision #27)."""
    rows = live_pick.commands_for([lab.detection("square", lab.CENTRE, 30.0, 0.9)], plane)
    module = verify_positions.rapid_module(rows, "now")

    assert checkpoint_lines(module)[0].endswith(", -30.0;")


def test_a_square_cannot_be_sent_turning_more_than_45_degrees(plane):
    """The folded angle, not the raw one, is what the robot is told to rotate to."""
    detections = [
        lab.detection("square", (lab.CENTRE[0] + 500.0, lab.CENTRE[1]), 0.0, 0.9),
        # Pixel y grows downward, so a minus y is up.
        lab.detection("hexagon", (lab.CENTRE[0], lab.CENTRE[1] - 500.0), 20.0, 0.9),
    ]
    module = verify_positions.rapid_module(live_pick.commands_for(detections, plane), "now")

    # Read the angle number at the end of each call.
    angles = [float(line.split(",")[-1].strip(" ;")) for line in checkpoint_lines(module)]
    assert all(-30.0 <= angle <= 30.0 for angle in angles), angles


def test_the_safe_height_can_be_moved_out_of_the_way(rows):
    module = verify_positions.rapid_module(rows, "now", safe_z=120.0)

    assert "nSafeZ := 120.0" in module


def test_the_module_says_where_it_came_from_and_that_nothing_is_picked(rows):
    module = verify_positions.rapid_module(rows, "23-10-2026 20:00")

    assert "Written by verify_positions.py on 23-10-2026 20:00" in module
    assert "measured, never taught" in module
    assert "vacuum is never set" in module


def test_a_missing_calibration_says_which_command_to_run(tmp_path, capsys):
    code = verify_positions.main(["--calibration", str(tmp_path / "none.json"), "--no-rapid"])

    printed = capsys.readouterr().err
    assert code == 2
    assert "no usable calibration" in printed
    assert "calibrate.py --live" in printed


def test_a_calibration_from_before_the_upright_fix_is_refused(tmp_path, capsys):
    legacy = {
        "method": "homography",
        "marker_mm": 97.0,
        "centre_px": [0.0, 0.0],
        "rotation_deg": 0.0,
        "sides_px": [97.0, 97.0, 97.0, 97.0],
        "matrix": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(legacy))

    assert verify_positions.main(["--calibration", str(path), "--no-rapid"]) == 2
    assert "frame_upright" in capsys.readouterr().err


def test_the_report_is_kept_in_the_output_folder(tmp_path, capsys):
    checks = [verify_pos_code.Check("model", True, "8 boxes"),
              verify_pos_code.Check("wire", False, "line was wrong")]

    code = verify_pos_code.report(checks, tmp_path / "report.txt")

    printed = capsys.readouterr().out
    kept = (tmp_path / "report.txt").read_text()
    assert code == 1
    assert "1 of 2 checks passed" in printed
    assert "1 of 2 checks passed" in kept
    assert "FAIL  wire" in kept
    assert "fix these before going near the robot" in kept


def test_a_frame_that_cannot_be_read_is_exit_2(tmp_path, capsys):
    code = verify_pos_code.main(["--image", str(tmp_path / "nope.png"),
                                 "--out", str(tmp_path / "out")])

    assert code == 2
    assert "cannot read" in capsys.readouterr().err


def test_a_folder_that_mixes_two_zips_is_named_not_crashed(tmp_path, capsys, monkeypatch):
    class Old:
        __file__ = str(tmp_path / "verify_positions.py")

    monkeypatch.setattr(verify_pos_code, "NEIGHBOURS", (("verify_positions", Old(), ("format_table",)),))

    assert verify_pos_code.main(["--image", str(tmp_path / "nope.png"), "--out", str(tmp_path)]) == 2
    printed = capsys.readouterr().err
    assert "mixes files from two different zips" in printed
    assert "verify_positions.py is too old (no format_table)" in printed
    assert "extract the newest IN700_lab_pc.zip" in printed


def test_every_shipped_module_is_the_same_version():
    # Check that helper modules provide the expected interfaces.
    assert verify_pos_code.check_neighbours() == []
