"""Headless behavioural tests for optimized_anno_img.py.

These exist to protect the 132 already-annotated images: every refactor of the
annotator has to keep producing byte-identical labels for the same input. They
run without a display, so the interactive window is never opened — the key
handlers are called directly instead of via cv2.waitKey().

Run:
    .venv\\Scripts\\python.exe -m pytest test_anno.py -q
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

import optimized_anno_img as anno

# A 100 wide, 50 tall test box; origin is top left.
SQUARE = [[0, 0], [100, 0], [100, 50], [0, 50]]


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the module's output folders at a temp directory, not the real ones."""
    # Keep all test file writes inside a temp folder.
    root = tmp_path / "raw"
    (root / "circle").mkdir(parents=True)
    monkeypatch.setattr(anno, "ROOT", root)
    monkeypatch.setattr(anno, "ROOT_RESOLVED", root.resolve())
    monkeypatch.setattr(anno, "LABELS_DIR", tmp_path / "labels_obb")
    monkeypatch.setattr(anno, "PREVIEW_DIR", tmp_path / "previews")
    return root


def make_session(root, name="circle/circle_0001.png", class_id=0, width=200, height=100):
    """Build a Session on a real (blank) image file inside the sandbox."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return anno.Session(path, cv2.imread(str(path)), class_id)


def click(session, *points):
    """Feed left-click points through the real mouse callback."""
    # Clicks only add points while drawing mode is on.
    session.outlining = True
    for x, y in points:
        session.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)


# --- geometry ----------------------------------------------------------------


def test_order_corners_starts_topmost_leftmost_and_is_clockwise():
    scrambled = np.array([[10, 10], [0, 0], [0, 10], [10, 0]], dtype=np.float32)
    ordered = anno.order_corners(scrambled)
    assert ordered.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_order_corners_is_idempotent():
    scrambled = np.array([[10, 10], [0, 0], [0, 10], [10, 0]], dtype=np.float32)
    once = anno.order_corners(scrambled)
    assert np.array_equal(anno.order_corners(once), once)


def test_order_corners_handles_a_rotated_rectangle():
    rotated = np.array([[100, 50], [120, 45], [115, 25], [95, 30]], dtype=np.float32)
    ordered = anno.order_corners(rotated)
    assert ordered[0].tolist() == [115, 25]   # topmost, and leftmost of the tie
    assert np.allclose(ordered, np.array([[115, 25], [120, 45], [100, 50], [95, 30]]))


def test_polygon_to_corners_always_returns_four_corners():
    triangle = np.array([[0, 0], [100, 0], [50, 60]], dtype=np.float32)
    assert anno.polygon_to_corners(triangle).shape == (4, 2)


def test_polygon_to_corners_of_a_square_is_that_square():
    # minAreaRect returns corners with ~1e-15 float noise, so compare with a
    # tolerance rather than exact equality.
    square = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], dtype=np.float32)
    corners = anno.polygon_to_corners(square)
    # Round away the ~1e-15 noise first: a tiny negative x would otherwise sort
    # a corner into a different place than its exact-value twin.
    rounded = {tuple(np.round(corner, 6).tolist()) for corner in corners}
    assert rounded == {(0.0, 0.0), (0.0, 40.0), (40.0, 0.0), (40.0, 40.0)}


# --- label export ------------------------------------------------------------


def test_yolo_line_normalizes_by_width_then_height():
    # If the (width, height) order were ever swapped, y would normalize to 2.0
    # here instead of 1.0. This is the trap recorded in CONTEXT.md.
    corners = np.array([[100, 50], [200, 50], [200, 100], [100, 100]], dtype=np.float64)
    line = anno.corners_to_yolo_line(corners, 0, (200, 100))
    assert line == "0 0.500000 0.500000 1.000000 0.500000 1.000000 1.000000 0.500000 1.000000"


def test_yolo_line_refuses_geometry_outside_the_image():
    # This box sticks out past the 200 pixel wide image.
    outside = np.array([[0, 0], [300, 0], [300, 50], [0, 50]], dtype=np.float64)
    assert anno.corners_to_yolo_line(outside, 2, (200, 100)) == ""


def test_yolo_line_refuses_a_zero_area_box():
    degenerate = np.array([[0, 0], [100, 0], [100, 0], [0, 0]], dtype=np.float64)
    assert anno.corners_to_yolo_line(degenerate, 2, (200, 100)) == ""


@pytest.mark.parametrize("class_id", [9, -1, True])
def test_yolo_line_refuses_an_invalid_class_id(class_id):
    corners = np.array(SQUARE, dtype=np.float64)
    assert anno.corners_to_yolo_line(corners, class_id, (200, 100)) == ""


def test_flipped_corner_order_gives_the_same_line():
    """Two things are asserted here. (1) The shoelace sum is sign-sensitive, so
    the absolute value is what stops a reversed corner list from being read as a
    zero-area box and refused. (2) A reversed list produces a *different* (but
    equally valid) line, because the corners are printed in the order given --
    YOLO OBB does not care which corner comes first, which is exactly why
    `order_corners()` exists upstream to make the output deterministic."""
    order = np.array(SQUARE, dtype=np.float64)
    flipped = order[::-1]
    # Checked on normalized corners, which is how corners_to_yolo_line calls it:
    # a 0.5 x 0.5 box encloses 0.25, so twice its area is 0.5.
    normalized = order / (200, 100)
    assert anno._doubled_area(normalized) == 0.5
    assert anno._doubled_area(normalized[::-1]) == 0.5

    forward = anno.corners_to_yolo_line(order, 2, (200, 100))
    backward = anno.corners_to_yolo_line(flipped, 2, (200, 100))
    assert forward != "" and backward != ""
    assert forward != backward
    assert set(forward.split()[1:]) == set(backward.split()[1:])


def test_yolo_line_refuses_a_collinear_box():
    collinear = np.array([[0, 0], [100, 0], [200, 0], [300, 0]], dtype=np.float64)
    assert anno._doubled_area(collinear) == 0.0
    assert anno.corners_to_yolo_line(collinear, 3, (400, 100)) == ""


def test_yolo_line_clamps_float_noise_at_the_edges():
    noisy = np.array([[-1e-9, 0], [200.0000001, 0], [200, 50], [0, 50]], dtype=np.float64)
    line = anno.corners_to_yolo_line(noisy, 0, (200, 100))
    assert line.startswith("0 0.000000 0.000000")


# --- sidecar validation ------------------------------------------------------


def test_as_corners_accepts_only_four_finite_points():
    assert anno.as_corners(SQUARE).shape == (4, 2)
    assert anno.as_corners([[1, 2], [3, 4]]) is None
    assert anno.as_corners([[1, 2], [3]]) is None
    assert anno.as_corners([[0, 0], [1, 1], [2, 2], [float("nan"), 0]]) is None


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"status": "nonsense", "objects": []},
        {"status": "draft", "objects": "not-a-list"},
        {"status": "draft", "objects": [{"class_id": True, "corners": SQUARE}]},
        {"status": "draft", "objects": [{"class_id": 9, "corners": SQUARE}]},
        {"status": "draft", "objects": [{"class_id": 0, "corners": [[0, 0]]}]},
    ],
)
def test_clean_sidecar_rejects_malformed_data(data):
    assert anno.clean_sidecar(data) is None


def test_clean_sidecar_keeps_objects_but_drops_a_bad_pending_outline():
    data = {"status": "draft", "objects": [{"class_id": 1, "corners": SQUARE}],
            "pending_points": {"not": "a list"}}
    cleaned = anno.clean_sidecar(data)
    assert cleaned["objects"][0]["class_id"] == 1
    assert cleaned["pending_points"] == []


# --- persistence -------------------------------------------------------------


def test_atomic_write_text_creates_parents_and_leaves_no_temp_file(sandbox):
    target = sandbox / "deep" / "nested" / "file.txt"
    anno.atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert list(target.parent.glob("*.tmp")) == []


def test_persist_skips_an_identical_payload(sandbox, monkeypatch):
    session = make_session(sandbox)
    calls = []
    # Keep the real writer so we can count calls.
    real_write = anno.atomic_write_text
    monkeypatch.setattr(
        anno, "atomic_write_text",
        lambda path, text: (calls.append(path), real_write(path, text))[1],
    )
    session.persist("draft")
    session.persist("draft")
    assert len(calls) == 1
    # Corners are plain nested lists in the real code path (click handlers store
    # corners.tolist()), because json.dumps cannot serialize an ndarray.
    session.objects.append({"class_id": 0, "corners": SQUARE})
    session.persist("draft")
    assert len(calls) == 2


def test_persist_deletes_a_stale_label_when_status_is_not_approved(sandbox):
    session = make_session(sandbox)
    # Plant an old label file that should be removed.
    session.paths.label.parent.mkdir(parents=True, exist_ok=True)
    session.paths.label.write_text("0 0 0 0 0 0 0 0 0\n", encoding="utf-8")
    session.persist("draft")
    assert not session.paths.label.is_file()


def test_sidecar_is_valid_json_on_disk(sandbox):
    session = make_session(sandbox)
    click(session, (10, 10), (80, 10), (80, 60), (10, 60))
    # Read the saved sidecar back as plain JSON.
    on_disk = json.loads(session.paths.sidecar.read_text(encoding="utf-8"))
    assert on_disk["status"] == "draft"
    assert len(on_disk["objects"]) == 1


# --- interactive flow (no window) -------------------------------------------


def test_four_clicks_close_one_object_and_outlining_stays_on(sandbox):
    session = make_session(sandbox)
    session.toggle_outlining()
    click(session, (10, 10), (80, 10), (80, 60), (10, 60))
    assert len(session.objects) == 1
    assert session.points == []
    assert session.outlining is True


def test_class_key_does_not_reclass_an_already_drawn_object(sandbox):
    session = make_session(sandbox)
    click(session, (10, 10), (80, 10), (80, 60), (10, 60))
    assert session.objects[0]["class_id"] == 0
    session.set_class(4)
    assert session.objects[0]["class_id"] == 0
    assert session.class_id == 4


def test_cycle_class_changes_only_the_selected_object(sandbox):
    session = make_session(sandbox)
    click(session, (10, 10), (80, 10), (80, 60), (10, 60))
    session.cycle_class()
    assert session.objects[0]["class_id"] == 1


def test_the_class_of_an_object_reaches_the_saved_label(sandbox):
    session = make_session(sandbox)
    session.set_class(1)
    click(session, (10, 10), (80, 10), (80, 60), (10, 60))
    session.approve()
    assert session.status == "approved"
    assert session.paths.label.read_text(encoding="utf-8").split()[0] == "1"


def test_select_wraps_both_ways(sandbox):
    session = make_session(sandbox)
    click(session, (10, 10), (40, 10), (40, 40), (10, 40))
    click(session, (60, 60), (90, 60), (90, 90), (60, 90))
    assert len(session.objects) == 2
    assert session.selected == 1
    session.select(1)
    assert session.selected == 0
    session.select(-1)
    assert session.selected == 1


# --- approving ---------------------------------------------------------------


def test_approve_refuses_an_empty_image(sandbox):
    session = make_session(sandbox)
    session.approve()
    assert session.status == "draft"
    assert session.message.startswith("0 objects")
    assert not session.paths.label.is_file()


def test_a_approves_a_genuinely_empty_image(sandbox):
    # Records CURRENT behaviour: approving as empty writes the sidecar and the
    # preview but NO .txt. Ultralytics treats an image with no label file as
    # background, so this is valid training data - but prepare_dataset.py must
    # be written knowing that "approved" is the only marker of a background
    # image, and it must not require a .txt to exist.
    session = make_session(sandbox)
    session.approve_empty()
    assert session.status == "approved"
    assert json.loads(session.paths.sidecar.read_text(encoding="utf-8"))["objects"] == []
    assert not session.paths.label.is_file()
    assert session.paths.preview.is_file()


def test_a_is_ignored_and_says_so_when_objects_exist(sandbox):
    session = make_session(sandbox)
    click(session, (10, 10), (80, 10), (80, 60), (10, 60))
    session.approve_empty()
    assert session.status != "approved"
    assert session.message.startswith("A ignored")


def test_write_labels_is_all_or_nothing(sandbox):
    session = make_session(sandbox)
    session.objects = [
        {"class_id": 0, "corners": anno.as_corners(SQUARE)},
        {"class_id": 0, "corners": anno.as_corners([[0, 0], [1, 1], [2, 2], [3, 3]])},
    ]
    written, invalid = session.write_labels()
    assert (written, invalid) == (False, [1])
    assert not session.paths.label.is_file()


def test_approve_writes_label_and_full_resolution_preview(sandbox):
    session = make_session(sandbox)
    click(session, (10, 10), (80, 10), (80, 60), (10, 60))
    session.approve()
    assert session.status == "approved"
    assert len(session.paths.label.read_text(encoding="utf-8").splitlines()) == 1
    preview = cv2.imread(str(session.paths.preview))
    assert preview.shape[:2] == (100, 200)   # the ORIGINAL size, not the display size


def test_a_failed_preview_write_is_reported_not_raised(sandbox, monkeypatch, capsys):
    session = make_session(sandbox)
    click(session, (10, 10), (80, 10), (80, 60), (10, 60))
    # Make the image writer fail on purpose.
    monkeypatch.setattr(anno.cv2, "imwrite", lambda *a, **k: False)
    assert session.write_preview() is None
    assert "preview NOT saved" in capsys.readouterr().out


# --- round trip --------------------------------------------------------------


def test_approve_then_reopen_keeps_the_label(sandbox):
    first = make_session(sandbox)
    click(first, (10, 10), (80, 10), (80, 60), (10, 60))
    first.approve()
    before = first.paths.label.read_text(encoding="utf-8")

    reopened = make_session(sandbox)
    assert reopened.status == "approved"
    assert len(reopened.objects) == 1
    assert reopened.paths.label.read_text(encoding="utf-8") == before


def test_editing_a_reopened_image_voids_its_label(sandbox):
    first = make_session(sandbox)
    click(first, (10, 10), (80, 10), (80, 60), (10, 60))
    first.approve()

    reopened = make_session(sandbox)
    reopened.delete_object()
    assert reopened.status == "draft"
    assert not reopened.paths.label.is_file()


def test_an_unfinished_outline_survives_a_restart(sandbox):
    first = make_session(sandbox)
    first.toggle_outlining()
    click(first, (10, 10), (80, 10))          # only two of four points
    assert len(first.points) == 2

    reopened = make_session(sandbox)
    assert reopened.points == [(10, 10), (80, 10)]


def test_a_malformed_sidecar_is_ignored_with_a_warning(sandbox, capsys):
    session = make_session(sandbox)
    session.paths.sidecar.parent.mkdir(parents=True, exist_ok=True)
    session.paths.sidecar.write_text("{not json", encoding="utf-8")
    fresh = make_session(sandbox)
    assert fresh.objects == []
    assert fresh.status == "draft"
    assert "unreadable sidecar ignored" in capsys.readouterr().out


# --- view / window sizing ----------------------------------------------------


def test_view_downscales_a_wide_photo_and_keeps_a_mapping():
    image = np.zeros((300, 4000, 3), dtype=np.uint8)
    view = anno.View.build(image)
    assert view.base.shape[1] == anno.DISPLAY_MAX_WIDTH
    assert view.scale == pytest.approx(anno.DISPLAY_MAX_WIDTH / 4000)
    assert view.image_size == (4000, 300)


def test_view_render_returns_a_frame_the_size_of_the_base():
    image = np.zeros((300, 4000, 3), dtype=np.uint8)
    view = anno.View.build(image)
    frame = view.render([], -1, [], "hello")
    assert frame.shape == view.base.shape


def test_to_image_falls_back_to_raw_coordinates_without_a_window():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    view = anno.View.build(image)
    # Without a window, screen points equal image points.
    assert view.to_image(37, 11) == (37, 11)


def test_opening_window_size_never_upscales_past_the_base():
    base = np.zeros((1200, 1600, 3), dtype=np.uint8)
    width, height = anno.opening_window_size(base)
    assert width <= base.shape[1]
    assert height <= base.shape[0]
    assert width >= 320 and height >= 240
