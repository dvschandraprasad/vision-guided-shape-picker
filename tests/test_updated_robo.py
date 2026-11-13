"""The RAPID module that ships beside MainModule_trail_14.mod.

It is a text artifact that no compiler in CI touches, so these tests pin what
the lab kept getting wrong and what the assignment actually asks for: the
robtarget budget (docs/project instructions.md:3), a hidden work object behind
Offs(), a base-frame safe-area window, a lift direction that followed the tool
axis, the required WaitDI, and a module name that matches its file name.
"""

from __future__ import annotations

import re
from pathlib import Path

# The RAPID file these tests inspect.
MODULE = Path(__file__).with_name("updated_robo.mod")
if not MODULE.is_file():  # the packaged copy keeps the modules in rapid/ beside scripts/
    MODULE = Path(__file__).resolve().parent.parent / "rapid" / "updated_robo.mod"


def read_lines() -> list[str]:
    return MODULE.read_text(encoding="utf-8").splitlines()


def code_lines() -> list[str]:
    """Every line that is not a comment, so comments cannot satisfy a check."""
    return [line for line in read_lines() if not line.lstrip().startswith("!")]


# Check the stored module and file names agree.
def test_the_module_name_matches_the_file_name():
    assert any(f"MODULE {MODULE.stem}" in line for line in code_lines())


def test_declares_seven_robtargets_five_positions_an_approach_and_a_home():
    """docs/project instructions.md:3 - five pick/place positions, one
    approach point, one home point, and no more."""
    pattern = re.compile(r"\s*(?:CONST|VAR|PERS)\s+robtarget\s+(\w+)\s*:=")
    names: set[str] = set()
    for line in code_lines():
        match = pattern.match(line)
        if match is not None:
            names.add(match.group(1))

    # Five shape positions plus approach and home make seven.
    assert len(names) == 7, f"the assignment allows 5 + approach + home, found {names}"
    assert names == {"start", "approach", "pCircle", "pSquare", "pStar", "pHex", "pTriangle"}


def test_only_the_pick_move_carries_the_turn():
    """The turn belongs at the pick, once; the place must not repeat it.

    The part keeps a fixed offset from the wrist once gripped, so arriving at the
    pocket's taught rotation already delivers the heading the vision measured.
    Repeating nAng at the mould turned every part a second time - measured in the
    cell on 2026-10-30, no shape came out right until it was reverted. The pick
    line is the one that must keep the rotation.
    """
    text = "\n".join(code_lines())

    assert "PROC PlaceAt(robtarget pTarget)" in text
    assert "pTurned" not in text, "the place must not rotate the pocket target"
    assert "PlaceAt GetPlaceTarget(nNr);" in text
    assert "Rz:=nAng" in text, "the pick still has to turn the cup"
    assert "oPick" in text


def test_the_pick_frame_is_orientation_only_so_it_costs_no_robtarget():
    text = "\n".join(code_lines())

    assert "pickBase" not in text
    assert "oPick" in text
    assert "cnfPick" in text
    assert "ejPick" in text


def test_offs_is_used_once_after_a_move_in_the_same_work_object():
    lines = read_lines()
    calls = [
        (index, line)
        for index, line in enumerate(lines)
        if not line.lstrip().startswith("!") and re.search(r"\bOffs\s*\(", line)
    ]

    assert len(calls) == 1, "one Offs(), the documented pick lift"
    index, call = calls[0]
    assert "pickBase" not in call
    previous_moves = [
        line for line in lines[:index] if "Move" in line and not line.lstrip().startswith("!")
    ]
    assert "wobjS" in previous_moves[-1], "Offs displaces along the work object of the previous move"


# Check that lift height is added to Z.
def test_lifts_add_to_z_instead_of_displacing_along_the_tool_axis():
    lifts = [line for line in code_lines() if "Move" in line and "Above(" in line]

    assert len(lifts) == 8, "one in PickAt, two in PlaceAt, five in CheckTargets"
    assert not [line for line in code_lines() if re.search(r"RelTool\([^)]*nLift", line)], (
        "RelTool moves along the tool axis; its sign depends on how the cup is mounted"
    )


def test_the_safe_area_is_sheet_millimetres_from_the_marker_centre():
    # Check the expected sheet-edge limit.
    bounds = [line for line in code_lines() if "ELSEIF nX" in line]

    assert len(bounds) == 1
    assert "nX > 200" in bounds[0]
    assert "800" not in bounds[0]


# Every move must name the work object it uses.
def test_every_pick_and_place_move_names_its_work_object():
    text = "\n".join(code_lines())

    assert "toolss\\WObj:=wobjS" in text
    assert "toolss\\WObj:=wobjD" in text


def test_the_yellow_button_is_waited_for_before_ready():
    """The assignment requires WaitDI; one press must mean one command."""
    lines = read_lines()
    waited = [index for index, line in enumerate(lines) if "WaitDI diYellowBtn, 1" in line]
    sent = [index for index, line in enumerate(lines) if 'RobotClienSendMessage "READY"' in line]

    assert len(waited) == 1
    assert len(sent) == 1
    assert waited[0] < sent[0], "the button is pressed before READY goes out"
    assert any("WaitDI diYellowBtn, 0" in line for line in lines), "released first"
