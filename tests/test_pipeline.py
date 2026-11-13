"""Headless tests for pipeline.py.

Synthetic images and a tiny generated video in tmp_path, so nothing here needs a
camera, a display, or the real dataset.

Run:
    .venv/Scripts/python.exe -m pytest test_pipeline.py -q
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import optimized_anno_img
import pipeline


def write_image(path, value):
    """Write a 16x16 solid-colour PNG so frames can be told apart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return image


def write_video(path, frames, size=(32, 32), fps=10.0):
    """Write a short MJPG video; codecs may alter pixels, so only shape is asserted."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "MJPG VideoWriter is unavailable"
    for value in frames:
        writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    writer.release()


def test_classes_match_the_training_labels():
    assert list(pipeline.CLASSES) == list(optimized_anno_img.CLASSES)


def test_allow_list_is_the_assigned_three_shapes():
    names = {pipeline.CLASSES[index] for index in pipeline.ALLOWED_CLASS_IDS}
    assert names == {"circle", "hexagon", "triangle"}


def test_detection_names_come_from_the_frozen_order():
    detection = pipeline.Detection(2, 0.9, (10.0, 20.0), 45.0, ((0, 0), (1, 0), (1, 1), (0, 1)))
    assert detection.name == "hexagon"


def test_detection_rejects_an_out_of_range_class():
    with pytest.raises(ValueError):
        pipeline.Detection(5, 0.9, (0.0, 0.0), 0.0, ())


# Check that fields reject changes after creation.
def assert_frozen(instance: object, field: str, value: object) -> None:
    """A frozen dataclass must refuse the write at RUNTIME, not only statically.

    The write goes through `setattr` with a variable name: a literal field name
    is rejected by the type checker before it can prove the guard exists, and
    the guard is the thing under test.
    """
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, value)


def test_detection_and_pick_command_are_immutable():
    detection = pipeline.Detection(0, 0.5, (1.0, 2.0), 0.0, ())
    command = pipeline.PickCommand("circle", 100.0, -20.0, 30.0)
    assert_frozen(detection, "class_id", 1)
    assert_frozen(command, "x_mm", 0.0)


def test_folder_source_reads_images_in_filename_order(tmp_path):
    write_image(tmp_path / "c.png", 30)
    write_image(tmp_path / "a.png", 10)
    write_image(tmp_path / "b.png", 20)

    source = pipeline.ImageFolderSource(tmp_path)
    values = []
    for _ in range(3):
        frame = source.read()
        assert frame is not None
        values.append(int(frame[0, 0, 0]))
    assert values == [10, 20, 30]


def test_folder_source_ends_then_loops(tmp_path):
    write_image(tmp_path / "a.png", 10)
    write_image(tmp_path / "b.png", 20)

    source = pipeline.ImageFolderSource(tmp_path)
    assert source.read() is not None
    assert source.read() is not None
    assert source.read() is None

    looping = pipeline.ImageFolderSource(tmp_path, loop=True)
    assert [looping.read() is not None for _ in range(5)] == [True] * 5


def test_folder_source_ignores_other_files(tmp_path):
    write_image(tmp_path / "a.png", 10)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    source = pipeline.ImageFolderSource(tmp_path)
    assert source.read() is not None
    assert source.read() is None


def test_folder_source_rejects_a_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.ImageFolderSource(tmp_path / "nope")


def test_folder_source_rejects_an_empty_folder(tmp_path):
    with pytest.raises(ValueError):
        pipeline.ImageFolderSource(tmp_path)


def test_video_source_reads_every_frame_then_ends(tmp_path):
    path = tmp_path / "clip.avi"
    write_video(path, [10, 20, 30])

    source = pipeline.VideoFileSource(path)
    try:
        frames = [source.read() for _ in range(3)]
        assert all(frame is not None and frame.shape == (32, 32, 3) for frame in frames)
        assert source.read() is None
    finally:
        source.close()


def test_video_source_loops_over_a_readable_file(tmp_path):
    path = tmp_path / "clip.avi"
    write_video(path, [10, 20])

    source = pipeline.VideoFileSource(path, loop=True)
    try:
        assert [source.read() is not None for _ in range(5)] == [True] * 5
    finally:
        source.close()


def test_video_source_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.VideoFileSource(tmp_path / "nope.mp4")


# Fake detector boxes that look like the real ones.
class FakeObb:
    """Stand-in for Ultralytics' OBB view: corners, conf, cls, and len()."""

    def __init__(self, boxes, confidences, class_ids):
        self.xyxyxyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4, 2)
        self.conf = np.asarray(confidences, dtype=np.float32)
        self.cls = np.asarray(class_ids, dtype=np.float32)

    def __len__(self):
        return len(self.conf)


def fake_result(boxes, confidences, class_ids):
    """An Ultralytics-like result with no detections when boxes is None."""
    if boxes is None:
        return SimpleNamespace(obb=None)
    return SimpleNamespace(obb=FakeObb(boxes, confidences, class_ids))


def rectangle(width, height, angle_deg, center=(50.0, 50.0)):
    """Corners of a rotated rectangle, clockwise from the top-left of its long edge."""
    box = cv2.boxPoints((center, (width, height), angle_deg))
    return box


def test_no_detections_gives_an_empty_list():
    assert pipeline.detections_from_result(fake_result(None, [], [])) == []


def test_result_with_all_five_classes_is_not_filtered():
    boxes = [rectangle(40, 20, 0.0, (20.0 + 30 * i, 50.0)) for i in range(5)]
    result = fake_result(boxes, [0.9] * 5, [0, 1, 2, 3, 4])

    detections = pipeline.detections_from_result(result)

    assert [d.class_id for d in detections] == [0, 1, 2, 3, 4]
    assert [d.name for d in detections] == list(pipeline.CLASSES)


def test_angles_are_canonicalized_to_the_long_edge():
    # A tall box (w < h as given) must come back with the long edge as w and an
    # angle inside [-45, 135): the convention the labels and the mapper assume.
    result = fake_result([rectangle(20, 40, 0.0)], [0.9], [0])
    detection = pipeline.detections_from_result(result)[0]

    assert -45.0 <= detection.angle_deg < 135.0
    assert detection.center == pytest.approx((50.0, 50.0), abs=0.5)


def test_angle_range_holds_for_every_rotation():
    boxes = [rectangle(40, 20, angle) for angle in (0, 30, 45, 90, 135, 179)]
    result = fake_result(boxes, [0.9] * 6, [0] * 6)

    angles = [d.angle_deg for d in pipeline.detections_from_result(result)]

    assert all(-45.0 <= angle < 135.0 for angle in angles)
    # Rotations 90 degrees apart describe the same rectangle, so the angle must
    # repeat modulo 90: otherwise downstream orientation would disagree.
    assert angles[0] == pytest.approx(angles[3] % 90, abs=1e-3) or \
        (angles[0] - angles[3]) % 90 == pytest.approx(0.0, abs=1e-3)


def test_confidences_and_corners_are_carried_through():
    result = fake_result([rectangle(40, 20, 10.0)], [0.42], [2])
    detection = pipeline.detections_from_result(result)[0]

    assert detection.confidence == pytest.approx(0.42, abs=1e-6)
    assert len(detection.corners) == 4
    assert all(len(corner) == 2 for corner in detection.corners)


def test_model_name_check_accepts_the_frozen_order():
    pipeline.check_model_names(dict(enumerate(pipeline.CLASSES)))


def test_model_name_check_rejects_a_renamed_class():
    renamed: dict[int, str] = dict(enumerate(pipeline.CLASSES))
    renamed[2] = "polygon"
    with pytest.raises(ValueError):
        pipeline.check_model_names(renamed)


# Wire line the robot reads: slot, x, y, height, turn.
def test_command_encoding_matches_the_rapid_parser():
    command = pipeline.PickCommand("hexagon", 123.4, -45.6, 30.0)
    assert pipeline.encode_command(command) == "3;123.4;-45.6;10.0;30.0"


def test_slot_is_the_tray_number_not_the_class_id():
    # CONTEXT section 6: the tray is 4 Triangle and 5 Star, while CLASSES puts
    # star at index 3. A class index on the wire swaps two place targets.
    assert pipeline.SLOT["circle"] == 1
    assert pipeline.SLOT["triangle"] == 4
    assert pipeline.SLOT["star"] == 5
    assert pipeline.SLOT["star"] != pipeline.CLASSES.index("star")


def test_command_round_trips_through_text():
    # The wire keeps one decimal, so a round trip is exact only for values that
    # have one: 12.34 would come back as 12.3 and that is the intended precision.
    command = pipeline.PickCommand("triangle", -1.0, 2.5, -12.3, z_mm=4.5)
    assert pipeline.decode_command(pipeline.encode_command(command) + "\n") == command


def test_pick_z_defaults_to_the_pick_plane():
    command = pipeline.PickCommand("circle", 0.0, 0.0, 0.0)
    assert command.z_mm == pipeline.PICK_Z_MM


@pytest.mark.parametrize("line", [
    "3;1;2",
    "3;1;2;3;4;5",
    "0;1;2;3;30",
    "6;1;2;3;30",
    "3;1;2;oops;30",
    "hexagon,1,2,30",          # the retired 4-field comma line: must not be read
])
def test_decode_rejects_lines_the_robot_cannot_use(line):
    with pytest.raises(ValueError):
        pipeline.decode_command(line)


def test_pick_command_rejects_an_unknown_shape():
    with pytest.raises(ValueError):
        pipeline.PickCommand("polygon", 0.0, 0.0, 0.0)


def test_fake_transport_records_what_the_robot_would_receive():
    command = pipeline.PickCommand("circle", 10.0, 20.0, 0.0)
    with pipeline.FakeTransport() as transport:
        transport.send_command(command)
        assert transport.sent == ["1;10.0;20.0;10.0;0.0"]
        assert transport.sent_commands() == [command]


def test_fake_transport_hands_over_injected_lines():
    transport = pipeline.FakeTransport(["picked\n"])
    assert transport.receive_line() == "picked"
    assert transport.receive_line() is None
    transport.inject("picked")
    assert transport.receive_line() == "picked"


def test_fake_transport_refuses_to_send_after_close():
    transport = pipeline.FakeTransport()
    transport.close()
    with pytest.raises(pipeline.TransportError):
        transport.send_command(pipeline.PickCommand("circle", 0.0, 0.0, 0.0))


def test_tcp_sends_nothing_before_a_connection():
    server = pipeline.TcpTransport(host="127.0.0.1", port=0)
    server.open()
    try:
        with pytest.raises(pipeline.TransportError):
            server.send_line("x")
    finally:
        server.close()


# Use local sockets to test the command connection.
def test_tcp_carries_lines_over_a_loopback_socket():
    server = pipeline.TcpTransport(host="127.0.0.1", port=0)
    server.open()
    client = pipeline.TcpTransport(host="127.0.0.1", port=server.port, server=False)
    client.open()
    try:
        server.accept(timeout_s=5.0)
        server.send_command(pipeline.PickCommand("star", 5.0, -6.0, 72.0))
        assert client.receive_line(timeout_s=5.0) == "5;5.0;-6.0;10.0;72.0"
        client.send_line("picked")
        assert server.receive_line(timeout_s=5.0) == "picked"
    finally:
        client.close()
        server.close()


def test_tcp_receive_returns_lines_one_at_a_time():
    server = pipeline.TcpTransport(host="127.0.0.1", port=0)
    server.open()
    client = pipeline.TcpTransport(host="127.0.0.1", port=server.port, server=False)
    client.open()
    try:
        server.accept(timeout_s=5.0)
        client.send_line("first")
        client.send_line("second")
        assert server.receive_line(timeout_s=5.0) == "first"
        assert server.receive_line(timeout_s=5.0) == "second"
    finally:
        client.close()
        server.close()


def test_tcp_receive_times_out_when_nothing_arrives():
    server = pipeline.TcpTransport(host="127.0.0.1", port=0)
    server.open()
    client = pipeline.TcpTransport(host="127.0.0.1", port=server.port, server=False)
    client.open()
    try:
        server.accept(timeout_s=5.0)
        assert server.receive_line(timeout_s=0.2) is None
    finally:
        client.close()
        server.close()


def test_tcp_reports_a_lost_connection():
    server = pipeline.TcpTransport(host="127.0.0.1", port=0)
    server.open()
    client = pipeline.TcpTransport(host="127.0.0.1", port=server.port, server=False)
    client.open()
    server.accept(timeout_s=5.0)
    client.close()
    try:
        with pytest.raises(pipeline.TransportError):
            server.receive_line(timeout_s=2.0)
    finally:
        server.close()


def test_robot_client_answers_the_trigger_with_a_command():
    transport = pipeline.FakeTransport(["pick"])
    with pipeline.RobotClient(transport) as robot:
        assert robot.wait_for_request(timeout_s=1.0) == "pick"
        robot.send_pick(pipeline.PickCommand("circle", 10.0, 20.0, 0.0))
    assert transport.sent == ["1;10.0;20.0;10.0;0.0"]


def test_robot_client_reports_silence_as_none():
    robot = pipeline.RobotClient(pipeline.FakeTransport())
    assert robot.wait_for_request(timeout_s=0.1) is None


def test_robot_client_closes_its_transport():
    transport = pipeline.FakeTransport()
    pipeline.RobotClient(transport).close()
    with pytest.raises(pipeline.TransportError):
        transport.send_line("x")


def test_robot_client_propagates_a_lost_connection():
    server = pipeline.TcpTransport(host="127.0.0.1", port=0)
    server.open()
    client = pipeline.TcpTransport(host="127.0.0.1", port=server.port, server=False)
    client.open()
    server.accept(timeout_s=5.0)
    client.close()
    robot = pipeline.RobotClient(server)
    try:
        with pytest.raises(pipeline.TransportError):
            robot.wait_for_request(timeout_s=2.0)
    finally:
        robot.close()


def make_detection(class_id, center, angle_deg):
    """A Detection with the fields the mapper reads; corners are irrelevant here."""
    return pipeline.Detection(class_id, 0.9, center, angle_deg, ((0, 0),) * 4)


# Convert pixel offsets from the origin into millimetres.
def test_scale_and_origin_map_pixels_to_millimetres():
    mapper = pipeline.CoordinateMapper(pipeline.Calibration(2.0, (100.0, 50.0)))
    assert mapper.to_mm((110.0, 60.0)) == pytest.approx((20.0, 20.0))


def test_rotation_turns_the_work_object_frame():
    # A point 10 px to the right of the origin, on an axis rotated 90 degrees.
    mapper = pipeline.CoordinateMapper(pipeline.Calibration(2.0, (100.0, 50.0), 90.0))
    assert mapper.to_mm((110.0, 50.0)) == pytest.approx((0.0, -20.0), abs=1e-9)


def test_flip_y_mirrors_into_the_robot_frame():
    mapper = pipeline.CoordinateMapper(
        pipeline.Calibration(2.0, (100.0, 50.0), 90.0, flip_y=True))
    assert mapper.to_mm((110.0, 50.0)) == pytest.approx((0.0, 20.0), abs=1e-9)


def test_angle_is_reported_relative_to_the_frame():
    mapper = pipeline.CoordinateMapper(pipeline.Calibration(1.0, rotation_deg=30.0))
    assert mapper.angle_in_frame(60.0) == pytest.approx(30.0)


def test_flipped_angle_changes_sign_and_stays_in_range():
    mapper = pipeline.CoordinateMapper(pipeline.Calibration(1.0, rotation_deg=30.0, flip_y=True))
    assert mapper.angle_in_frame(60.0) == pytest.approx(-30.0)
    assert mapper.angle_in_frame(-20.0) == pytest.approx(50.0)


def test_angles_outside_the_range_are_canonicalized():
    assert pipeline.canonical_angle(180.0) == pytest.approx(0.0)
    assert pipeline.canonical_angle(135.0) == pytest.approx(-45.0)
    assert pipeline.canonical_angle(-50.0) == pytest.approx(130.0)


def test_a_circle_is_never_rotated():
    mapper = pipeline.CoordinateMapper(pipeline.Calibration(1.0, rotation_deg=45.0))
    command = mapper.to_pick_command(make_detection(0, (10.0, 10.0), 80.0))
    assert command.shape == "circle"
    assert command.angle_deg == 0.0


def test_a_square_gets_the_mapped_angle_and_position():
    mapper = pipeline.CoordinateMapper(pipeline.Calibration(3.0, (10.0, 10.0), 30.0))
    command = mapper.to_pick_command(make_detection(1, (10.0, 10.0), 60.0))
    assert command.shape == "square"
    assert (command.x_mm, command.y_mm) == pytest.approx((0.0, 0.0))
    assert command.angle_deg == pytest.approx(30.0)


def test_calibration_rejects_a_non_positive_scale():
    with pytest.raises(ValueError):
        pipeline.Calibration(0.0)
    with pytest.raises(ValueError):
        pipeline.Calibration(-1.0)


def test_mapped_command_survives_the_wire_format():
    mapper = pipeline.CoordinateMapper(pipeline.Calibration(0.1, (1000.0, 800.0)))
    # A hexagon sends no turn (decision #27), so the angle field is 0.0 while the
    # slot still goes out as 3: shape and heading are separate fields.
    command = mapper.to_pick_command(make_detection(2, (1200.0, 700.0), 15.0))
    transport = pipeline.FakeTransport()
    transport.send_command(command)
    assert transport.sent_commands() == [command]
    assert transport.sent[0].startswith("3;20.0;-10.0;10.0;15.0")


def test_only_a_circle_sends_no_turn():
    # Every shape with a heading now sends it, folded into its own symmetry. The
    # circle has no heading at all, so it stays at 0.0 whatever was measured.
    assert pipeline.pick_angle("circle", 37.0) == 0.0
    assert pipeline.pick_angle("square", 22.5) == pytest.approx(22.5)
    assert pipeline.pick_angle("hexagon", 26.6) == pytest.approx(26.6)
    assert pipeline.pick_angle("triangle", 30.0) == pytest.approx(30.0)
    assert pipeline.pick_angle("star", 30.0) == pytest.approx(30.0)
    assert pipeline.pick_angle("triangle", 100.0) == pytest.approx(-20.0)   # 100 - 120
    assert pipeline.pick_angle("hexagon", 44.0) == pytest.approx(-16.0)     # 44 - 60
    assert pipeline.sending_shapes() == "square, hexagon, triangle, star"


def test_the_offset_lands_before_the_fold():
    # 40 + 20 = 60 folds to 30 for a square, while folding first would keep 40.
    assert pipeline.pick_angle("square", 40.0, 20.0) == pytest.approx(-30.0)
    assert pipeline.fold_to_symmetry("square", 40.0) == pytest.approx(40.0)


def test_a_shape_can_carry_its_own_pocket_reference(monkeypatch):
    # Every pocket is taught separately in wobjD, so one shape can sit 60 degrees
    # round from the others. The reference must move that shape only.
    monkeypatch.setitem(pipeline.SHAPE_OFFSET_DEG, "triangle", 60.0)
    assert pipeline.pick_angle("triangle", 30.0) == pytest.approx(-30.0)   # 90 - 120
    assert pipeline.pick_angle("triangle", 0.0) == pytest.approx(-60.0)    # 60 - 120
    assert pipeline.pick_angle("square", 30.0) == pytest.approx(30.0)     # untouched
    assert pipeline.pick_angle("hexagon", 20.0) == pytest.approx(20.0)     # 20 folds to itself


def test_the_shape_reference_and_the_global_offset_add_up(monkeypatch):
    # They answer different questions (this pocket, versus the camera-to-wobjS
    # chain), so the cell can fix one without disturbing the other.
    monkeypatch.setitem(pipeline.SHAPE_OFFSET_DEG, "triangle", 10.0)
    assert pipeline.pick_angle("triangle", 20.0, 5.0) == pytest.approx(35.0)
    assert pipeline.pick_angle("triangle", 20.0) == pytest.approx(30.0)


def test_pick_angle_rejects_an_unknown_shape():
    with pytest.raises(ValueError, match="unknown shape"):
        pipeline.pick_angle("octagon", 10.0)


RED = (0, 0, 255)      # BGR, the order cv2.imread hands back
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def solid(colour, size=16):
    return np.full((size, size, 3), colour, np.uint8)


@pytest.mark.parametrize("colour", ["red", "green", "blue", "black", "white"])
def test_a_solid_crop_names_its_colour(colour):
    bgr = {"red": RED, "green": GREEN, "blue": BLUE, "black": BLACK, "white": WHITE}[colour]
    assert pipeline.colour_of(solid(bgr)) == colour


def test_grey_is_unknown_and_black_is_a_colour():
    assert pipeline.colour_of(solid(GREY)) == pipeline.UNKNOWN_COLOUR
    assert "black" in pipeline.COLOURS
    assert pipeline.colour_of(solid((0, 0, 0))) == "black"


def test_a_shadowed_patch_reads_black_not_a_hue():
    """Darkness has no hue, so black is the only name left for it."""
    assert pipeline.colour_of(solid((0, 0, 25))) == "black"


def test_white_is_a_colour_and_a_bright_hue_is_not_white():
    """Bright and unsaturated is white; bright and saturated is still its hue."""
    assert "white" in pipeline.COLOURS
    assert pipeline.colour_of(solid(WHITE)) == "white"
    assert pipeline.colour_of(solid((200, 200, 200))) == pipeline.UNKNOWN_COLOUR  # light grey
    assert pipeline.colour_of(solid((230, 230, 230))) == "white"   # the cut is at 215
    assert pipeline.colour_of(solid((0, 0, 255))) == "red"


def test_a_tie_between_two_colours_is_unknown():
    half = np.hstack([solid(RED), solid(BLUE)])
    assert pipeline.colour_of(half) == pipeline.UNKNOWN_COLOUR


def test_the_crop_follows_the_detection_box():
    red_on_grey = solid(GREY)
    red_on_grey[0:4, 0:4] = RED
    inner = pipeline.Detection(
        0, 0.9, (2.0, 2.0), 0.0,
        ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    assert pipeline.colour_of(red_on_grey, inner) == "red"

    # The same red patch, with the box somewhere else: only the box votes.
    elsewhere = pipeline.Detection(
        0, 0.9, (12.0, 12.0), 0.0,
        ((8.0, 8.0), (14.0, 8.0), (14.0, 14.0), (8.0, 14.0)))
    assert pipeline.colour_of(red_on_grey, elsewhere) == pipeline.UNKNOWN_COLOUR
    assert pipeline.colour_of(red_on_grey) == "red"


def test_colour_is_not_a_class():
    assert set(pipeline.COLOURS).isdisjoint(pipeline.CLASSES)
    assert pipeline.UNKNOWN_COLOUR not in pipeline.CLASSES


class FakeDetector:
    """A detector that returns whatever the test told it to."""

    def __init__(self, *detections):
        self.detections = list(detections)
        self.frames_seen = []

    def detect(self, frame):
        self.frames_seen.append(frame)
        return list(self.detections)


def make_cycle(folder, detector, *, scale=1.0, incoming=()):
    transport = pipeline.FakeTransport(incoming)
    robot = pipeline.RobotClient(transport)
    mapper = pipeline.CoordinateMapper(pipeline.Calibration(scale))
    cycle = pipeline.PickCycle(pipeline.ImageFolderSource(str(folder)), detector, mapper, robot)
    return cycle, transport


def test_pick_cycle_answers_a_trigger_with_the_mapped_pick(tmp_path):
    write_image(tmp_path / "frame.png", 7)
    detector = FakeDetector(make_detection(0, (20.0, 10.0), 0.0))
    cycle, transport = make_cycle(tmp_path, detector, scale=2.0, incoming=["request"])
    command = cycle.pick_next()
    assert command == pipeline.PickCommand("circle", 40.0, 20.0, 0.0)
    assert transport.sent == ["1;40.0;20.0;10.0;0.0"]
    assert len(detector.frames_seen) == 1


def test_pick_cycle_sends_nothing_without_a_trigger(tmp_path):
    write_image(tmp_path / "frame.png", 7)
    cycle, transport = make_cycle(tmp_path, FakeDetector(make_detection(0, (1.0, 1.0), 0.0)))
    assert cycle.pick_next() is None
    assert transport.sent == []


def test_pick_cycle_stops_when_the_frames_run_out(tmp_path):
    write_image(tmp_path / "frame.png", 7)
    detector = FakeDetector(make_detection(0, (1.0, 1.0), 0.0))
    cycle, transport = make_cycle(tmp_path, detector, incoming=["one", "two"])
    assert cycle.pick_next() is not None
    assert cycle.pick_next() is None
    assert transport.sent == ["1;1.0;1.0;10.0;0.0"]


def test_pick_cycle_sends_nothing_for_an_empty_frame(tmp_path):
    write_image(tmp_path / "frame.png", 7)
    cycle, transport = make_cycle(tmp_path, FakeDetector(), incoming=["request"])
    assert cycle.pick_next() is None
    assert transport.sent == []


# With several shapes, the most confident one becomes the pick.
def test_pick_cycle_chooses_the_most_confident_detection(tmp_path):
    write_image(tmp_path / "frame.png", 7)
    weak = pipeline.Detection(1, 0.5, (1.0, 1.0), 0.0, ((0, 0),) * 4)
    strong = pipeline.Detection(3, 0.95, (5.0, 5.0), 20.0, ((0, 0),) * 4)
    cycle, transport = make_cycle(tmp_path, FakeDetector(weak, strong), incoming=["request"])
    command = cycle.pick_next()
    assert command is not None
    assert command.shape == "star"
    assert transport.sent_commands() == [command]


def test_both_sources_are_frame_sources(tmp_path):
    write_image(tmp_path / "a.png", 10)
    write_video(tmp_path / "clip.avi", [10])
    video = pipeline.VideoFileSource(tmp_path / "clip.avi")
    try:
        assert isinstance(pipeline.ImageFolderSource(tmp_path), pipeline.FrameSource)
        assert isinstance(video, pipeline.FrameSource)
    finally:
        video.close()
