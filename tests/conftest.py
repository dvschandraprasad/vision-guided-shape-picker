"""Test set-up only (added when the project was packaged for GitHub; no program logic).

This file does two things:

1. Test isolation.
   pipeline.apply_angle_teach() changes module-level values in pipeline.py
   (ANGLE_SIGN, SHAPE_OFFSET_DEG, ...). Some tests call live_pick.main(), which calls
   apply_angle_teach() and reads the real data/angle_teach.json. Without the fixture
   below, the taught values stay active for the tests that run later, and those tests
   fail only because of the run order. Every test file passes when it is run on its own.
   The fixture saves those values before each test and puts them back after it.

2. Known-missing files.
   A few source files were not in the hand-in (see README "Known limitations").
   The tests that need them are not run while the file is missing, and pytest says so
   at the top and the bottom of the run. As soon as you copy a missing file into its
   place, its tests run again - nothing here has to change.

     file / package needed                       tests
     scripts/dashboard_new.py                    tests/test_grade_levels.py (not collected)
     scripts/dashboard.py + tkinter              tests/test_dashboard.py    (not collected)
     rapid/updated_robo.mod                      tests/test_updated_robo.py (skipped)
     ultralytics (needs torch)                   5 tests in tests/test_pipeline.py (skipped)

   The 7 skips in tests/test_angle_teach.py come from the test file itself: it needs
   scripts/frame_20261030_171238_018578.png.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
from pathlib import Path

import pytest

import pipeline

# ---------------------------------------------------------------------------
# 1. test isolation
# ---------------------------------------------------------------------------
_STATE_NAMES = (
    "ANGLE_SIGN",
    "SHAPE_OFFSET_DEG",
    "TAUGHT_ZERO_DEG",
    "ANGLE_TEACH_STATE",
    "ANGLE_TEACH_DETAIL",
)


@pytest.fixture(autouse=True)
def _restore_pipeline_angle_state():
    saved = {name: copy.deepcopy(getattr(pipeline, name)) for name in _STATE_NAMES}
    yield
    for name, value in saved.items():
        current = getattr(pipeline, name)
        if isinstance(current, dict) and isinstance(value, dict):
            # keep the same dict object: other modules may hold a reference to it
            current.clear()
            current.update(value)
        else:
            setattr(pipeline, name, value)


# ---------------------------------------------------------------------------
# 2. known-missing files
# ---------------------------------------------------------------------------
_TESTS = Path(__file__).resolve().parent
_REPO = _TESTS.parent


def _have_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# test file -> why it cannot be imported right now (only files that cannot even be imported)
_NOT_COLLECTED: dict[str, str] = {}
if not (_REPO / "scripts" / "dashboard_new.py").is_file():
    _NOT_COLLECTED["test_grade_levels.py"] = (
        "needs scripts/dashboard_new.py (missing from the hand-in; "
        "dashboard_web.py imports it)")
if not (_REPO / "scripts" / "dashboard.py").is_file():
    _NOT_COLLECTED["test_dashboard.py"] = "needs scripts/dashboard.py (missing from the hand-in)"
elif not _have_module("tkinter"):
    _NOT_COLLECTED["test_dashboard.py"] = "needs tkinter (install python3-tk)"

# pytest reads this name: files listed here are not collected
collect_ignore = sorted(_NOT_COLLECTED)

_UPDATED_ROBO = (_TESTS / "updated_robo.mod", _REPO / "rapid" / "updated_robo.mod")
_HAVE_UPDATED_ROBO = any(path.is_file() for path in _UPDATED_ROBO)
_HAVE_ULTRALYTICS = _have_module("ultralytics")


def pytest_report_header(config):
    lines = [f"not collected: tests/{name} - {why}" for name, why in sorted(_NOT_COLLECTED.items())]
    if not _HAVE_UPDATED_ROBO:
        lines.append("skipped: tests/test_updated_robo.py - needs rapid/updated_robo.mod")
    if not _HAVE_ULTRALYTICS:
        lines.append("skipped: tests/test_pipeline.py detections_from_result tests - "
                     "needs ultralytics (install CPU torch, then requirements.txt)")
    return lines


def pytest_collection_modifyitems(config, items):
    for item in items:
        file_name = item.path.name
        if file_name == "test_updated_robo.py" and not _HAVE_UPDATED_ROBO:
            item.add_marker(pytest.mark.skip(
                reason="needs rapid/updated_robo.mod (missing from the hand-in)"))
        elif (file_name == "test_pipeline.py" and not _HAVE_ULTRALYTICS
              and _calls_detections_from_result(item)):
            item.add_marker(pytest.mark.skip(
                reason="needs ultralytics: pipeline.detections_from_result imports "
                       "ultralytics.utils.ops"))


def _calls_detections_from_result(item) -> bool:
    function = getattr(item, "function", None)
    if function is None:
        return False
    try:
        return "detections_from_result" in inspect.getsource(function)
    except (OSError, TypeError):
        return False


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _NOT_COLLECTED:
        return
    terminalreporter.section("not collected (missing files)")
    for name, why in sorted(_NOT_COLLECTED.items()):
        terminalreporter.write_line(f"tests/{name}: {why}")
