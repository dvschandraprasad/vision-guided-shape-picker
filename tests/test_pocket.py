"""Tests for the pocket acceptance-angle model: limits, monotonicity, no magic."""

from __future__ import annotations

import pytest

import pocket

POLYGONS = ("square", "hexagon", "triangle", "star")


# Test helper: angle for one shape, never None.
def window(shape: str, gap_mm: float) -> float:
    """The acceptance angle, with the circle's `None` (any angle) ruled out."""
    angle = pocket.insert_angle_deg(shape, gap_mm)
    assert angle is not None
    return angle


def test_a_circle_has_no_heading_to_absorb():
    assert pocket.insert_angle_deg("circle", 1.0) is None
    assert pocket.fits("circle", 0.0, 179.0)


def test_no_clearance_accepts_no_rotation():
    for shape in POLYGONS:
        assert window(shape, 0.0) == 0.0


# More gap must never give a smaller angle.
def test_more_clearance_accepts_more_rotation():
    for shape in POLYGONS:
        angles = [window(shape, gap) for gap in (0.5, 1.0, 2.0, 3.0)]
        assert angles == sorted(angles), f"{shape}: {angles}"
        assert angles[-1] > angles[0]


def test_the_window_never_crosses_the_half_symmetry_limit():
    for shape in POLYGONS:
        assert window(shape, 25.0) <= pocket.HALF_PERIOD_DEG[shape]


def test_a_sharp_vertex_absorbs_less_than_a_flat_side():
    """A triangle corner sweeps a long radius, so it reaches the wall first."""
    assert window("triangle", 1.0) < window("hexagon", 1.0)
    assert window("star", 1.0) < window("hexagon", 1.0)


def test_a_measured_gap_lands_in_a_believable_window():
    """1 mm of clearance on a 50 mm part is a few degrees, not zero and not tens."""
    for shape in POLYGONS:
        angle = window(shape, 1.0)
        assert 0.0 < angle < 15.0, f"{shape}: {angle}"


def test_fits_says_no_well_past_the_window():
    accepted = window("hexagon", 1.0)
    assert pocket.fits("hexagon", 1.0, 0.0)
    assert not pocket.fits("hexagon", 1.0, accepted + 5.0)


def test_an_unknown_shape_is_a_loud_error():
    with pytest.raises(ValueError, match="unknown shape"):
        pocket.outline("octagon", 50.0)


def test_a_negative_gap_is_refused():
    # A negative gap is refused, not silently used.
    with pytest.raises(ValueError, match="gap"):
        pocket.insert_angle_deg("square", -1.0)


def test_the_table_covers_every_shape_and_gap():
    rows = pocket.table((0.5, 1.0), 50.0)
    assert [name for name, _ in rows] == list(pocket.SHAPES)
    assert all(len(angles) == 2 for _, angles in rows)


# Test helper: mouth angle for one shape.
def mouth(shape: str, clearance_mm: float, chamfer_mm: float) -> float:
    """The entry angle of a chamfered mouth, with the circle's `None` ruled out."""
    angle = pocket.mouth_angle_deg(shape, clearance_mm, chamfer_mm)
    assert angle is not None
    return angle


def test_a_chamfer_widens_the_mouth_monotonically():
    for shape in POLYGONS:
        angles = [mouth(shape, 0.5, chamfer) for chamfer in (0.0, 1.0, 2.0, 4.0)]
        assert angles == sorted(angles), f"{shape}: {angles}"


# Zero chamfer must match the plain wall angle.
def test_no_chamfer_is_the_straight_wall_again():
    for shape in POLYGONS:
        assert pocket.mouth_angle_deg(shape, 0.5, 0.0) == pocket.insert_angle_deg(shape, 0.5)


# Closes the loop: needed chamfer must admit the angle.
def test_the_requested_chamfer_really_admits_that_angle():
    for shape in POLYGONS:
        wanted = pocket.MEASURED_P95_DEG[shape]
        needed = pocket.chamfer_needed_mm(shape, min(wanted, 5.0), 0.5)
        assert needed is not None, shape
        admitted = mouth(shape, 0.5, needed)
        assert admitted >= min(wanted, 5.0) - 0.2, f"{shape}: {needed} mm admits {admitted}"


def test_a_shape_is_never_asked_for_more_than_its_symmetry():
    assert pocket.chamfer_needed_mm("square", 45.0) is None
    assert pocket.chamfer_needed_mm("circle", 30.0) == 0.0


def test_a_negative_chamfer_is_refused():
    with pytest.raises(ValueError, match="chamfer"):
        pocket.mouth_angle_deg("hexagon", 0.5, -1.0)
