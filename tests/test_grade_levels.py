"""Tests for the grade-level dashboards: no camera, no model, no robot.

    python -m pytest tests/test_grade_levels.py -q
"""
from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import grade_common as gc
import dashboard_web
from grade1_web import Grade1Controller
from grade2_web import Grade2Controller
from grade3_web import Grade3Controller
from pipeline import CLASSES, Detection, PickCommand

GRADES = (Grade1Controller, Grade2Controller, Grade3Controller)


# Stand-in mapper that returns fixed pick coordinates.
class FakeMapper:
    def to_pick_command(self, detection, angle_offset_deg=0.0, image=None):
        return PickCommand(detection.name, 10.0, 20.0, 0.0)


# Stand-in sender that records commands and moves nothing.
class FakeSender:
    error = None
    broken = False

    def __init__(self):
        self.sent: list[PickCommand] = []

    def offer(self, command):
        self.sent.append(command)


class FakeDetector:
    last_inference_ms = 812.0


class FakeWorker:
    detector = FakeDetector()


def detection(name: str, confidence: float = 0.9) -> Detection:
    square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    return Detection(CLASSES.index(name), confidence, (5.0, 5.0), 0.0, square)


def frame_with(controller, *names):
    """Simulate a new camera result: these parts are in view."""
    model = dashboard_web.FastModel(FakeMapper())
    model.refresh([detection(n, 0.9 - i * 0.1) for i, n in enumerate(names)])
    model.selected = controller.model.selected
    model.selected_colour = controller.model.selected_colour
    controller.model = model


# Patch the dashboard before each test, then undo.
@pytest.fixture(autouse=True)
def patched():
    previous = gc.install(Grade1Controller)
    yield
    gc.restore(previous)


# Build one controller using the fake parts.
def make(cls):
    sender = FakeSender()
    controller = cls(dashboard_web.FastModel(FakeMapper()), FakeWorker(), sender, None)
    return controller, sender


ALL = gc.ASSIGNED_SHAPES  # ('circle', 'hexagon', 'triangle')


# --- detector ------------------------------------------------------------------
def test_assigned_shapes_are_circle_hexagon_triangle():
    assert ALL == ("circle", "hexagon", "triangle")


def test_unassigned_shapes_are_dropped():
    found = [detection(name) for name in CLASSES]
    assert [d.name for d in gc.keep_assigned(found)] == list(ALL)


def test_detector_filters_and_records_time(monkeypatch):
    every = [detection(name) for name in CLASSES]
    monkeypatch.setattr(gc.AssignedShapeDetector.__mro__[1], "detect", lambda self, image: every)
    det = gc.AssignedShapeDetector.__new__(gc.AssignedShapeDetector)
    det.detect(None)                                  # warm-up: no time recorded
    assert det.last_inference_ms is None
    assert {d.name for d in det.detect(None)} == set(ALL)
    assert det.last_inference_ms is not None


def test_install_limits_shape_list_and_restores():
    assert dashboard_web.SHAPE_CHOICES == ("any", *ALL)
    assert dashboard_web.Controller is Grade1Controller
    previous = gc.install(Grade3Controller)
    gc.restore(previous)
    assert dashboard_web.Controller is Grade1Controller


# --- page ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", GRADES)
def test_page_builds_for_every_grade(cls):
    page = gc.build_page(cls)
    assert f"Grade level {cls.LEVEL}" in page
    assert "gradePoll(d);" in page
    assert ("Start next cycle" in page) == cls.MULTI_CYCLE
    shape_hidden = '<span style="display:none">Pick by shape' in page
    colour_hidden = '<span style="display:none">colour' in page
    assert shape_hidden == (not cls.SELECT_SHAPE)
    assert colour_hidden == (not cls.SELECT_COLOUR)


# --- Grade I -------------------------------------------------------------------
def test_grade1_one_cycle_all_three():
    c, sender = make(Grade1Controller)
    for _ in ALL:
        frame_with(c, *[s for s in ALL if s not in c.attempts])
        assert "picked" in c.pick()
    assert {cmd.shape for cmd in sender.sent} == set(ALL)
    assert c.cycle_complete
    assert "run complete" in c.state()["grade"]["headline"]
    assert "single cycle" in c.next_cycle()


def test_grade1_ignores_filters():
    c, _ = make(Grade1Controller)
    c.set_filters("triangle", "red")
    assert c.model.selected is None and c.model.selected_colour is None


def test_grade1_repick_after_a_miss_is_allowed():
    c, sender = make(Grade1Controller)
    frame_with(c, "circle")
    c.pick()
    frame_with(c, "circle")       # the robot missed: still in view
    assert "picked circle" in c.pick()
    assert len(sender.sent) == 2


# --- Grade II ------------------------------------------------------------------
def test_grade2_blocks_until_next_cycle():
    c, sender = make(Grade2Controller)
    for _ in ALL:
        frame_with(c, *[s for s in ALL if s not in c.attempts])
        c.pick()
    frame_with(c, *ALL)
    assert "press 'Start next cycle'" in c.pick()
    assert len(sender.sent) == 3
    assert "cycle 2 started" in c.next_cycle()
    assert c.attempts == {} and c.cycle == 2
    assert "picked" in c.pick()


def test_grade2_colour_allowed_shape_ignored():
    c, _ = make(Grade2Controller)
    c.set_filters("triangle", None)
    assert c.model.selected is None
    c.set_filters(None, "red")
    assert c.model.selected_colour == "red"


def test_grade2_shows_inference_time():
    c, _ = make(Grade2Controller)
    g = c.state()["grade"]
    assert g["inference_ms"] == 812 and g["limit_ms"] == 3000


# --- Grade III -----------------------------------------------------------------
def test_grade3_needs_a_shape():
    c, sender = make(Grade3Controller)
    frame_with(c, *ALL)
    assert "choose which shape" in c.pick()
    assert sender.sent == []


def test_grade3_picks_the_chosen_shape():
    c, sender = make(Grade3Controller)
    frame_with(c, *ALL)
    c.set_filters("hexagon", None)
    assert "attempt 1 of 3" in c.pick()
    assert sender.sent[-1].shape == "hexagon"


def test_grade3_three_attempts_then_refused():
    c, sender = make(Grade3Controller)
    c.set_filters("triangle", None)
    for attempt in (1, 2, 3):
        frame_with(c, *ALL)
        assert f"attempt {attempt} of 3" in c.pick()
    frame_with(c, *ALL)
    assert "all 3 attempts used" in c.pick()
    assert len(sender.sent) == 3
    c.set_filters("circle", None)       # other shapes still allowed
    assert "picked circle" in c.pick()


def test_grade3_retry_allowed_after_cycle_complete():
    c, _ = make(Grade3Controller)
    for shape in ALL:
        frame_with(c, *ALL)
        c.set_filters(shape, None)
        c.pick()
    assert c.cycle_complete
    frame_with(c, *ALL)
    assert "attempt 2 of 3" in c.pick()


def test_grade3_next_cycle_resets_attempts_and_choice():
    c, _ = make(Grade3Controller)
    frame_with(c, *ALL)
    c.set_filters("triangle", None)
    c.pick()
    c.next_cycle()
    assert c.cycle == 2 and c.attempts == {} and c.model.selected is None


def test_grade3_rejects_unassigned_shape():
    c, _ = make(Grade3Controller)
    with pytest.raises(ValueError):
        c.set_filters("star", None)


def test_link_down_counts_nothing():
    c, sender = make(Grade3Controller)
    sender.error, sender.broken = RuntimeError("gone"), True
    frame_with(c, *ALL)
    c.set_filters("circle", None)
    assert "link is down" in c.pick()
    assert c.attempts == {}


# --- HTTP route ----------------------------------------------------------------
def test_next_cycle_route_over_http():
    c, _ = make(Grade2Controller)
    dashboard_web.CONTROLLER = c
    # Start a temporary web server on a free port.
    server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard_web.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/next_cycle", method="POST")
        assert json.load(urllib.request.urlopen(req))["status"] == "cycle 2 started"
        state = json.load(urllib.request.urlopen(base + "/state"))
        assert state["grade"]["cycle"] == 2
        page = urllib.request.urlopen(base + "/").read().decode()
        assert '"triangle"' in page and '"star"' not in page
    finally:
        server.shutdown()
        server.server_close()
