"""Headless tests for the dashboard rules: no window, no camera, no robot."""

from __future__ import annotations

import time
import tkinter as tk
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import calibrate
import camera
import dashboard
import pipeline


def _session_root():
    """One Tk interpreter for the whole session.

    Every destroy finalises Tcl, and a later Tk() then sometimes fails to find
    init.tcl ("Can't find a usable init.tcl"). Keeping one root alive avoids
    that, which is what made these tests skip at random.
    """
    try:
        return tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"no display for Tk: {error}")


@pytest.fixture(scope="session")
def _tk_session():
    root = _session_root()
    yield root


@pytest.fixture
def tk_root(_tk_session):
    """The shared root, emptied of the previous test's widgets."""
    for child in _tk_session.winfo_children():
        child.destroy()
    return _tk_session


def make_detection(class_id, center, angle_deg, confidence=0.9):
    """A Detection with the fields the dashboard reads."""
    return pipeline.Detection(class_id, confidence, center, angle_deg, ((0, 0),) * 4)


# Builds the dashboard model from a simple scale.
def make_model(scale=1.0, rotation_deg=0.0):
    return dashboard.DashboardModel(
        pipeline.CoordinateMapper(
            pipeline.Calibration(scale, rotation_deg=rotation_deg)
        )
    )


def test_rows_show_mapped_millimetres():
    # Scale 2.0 means two millimetres per pixel.
    model = make_model(scale=2.0)
    rows = model.refresh([make_detection(0, (10.0, 5.0), 0.0)])
    assert rows == (dashboard.ShapeRow("circle", "unknown", 0.9, 20.0, 10.0, 0.0),)


def make_plane():
    """A sheared, non-square homography: nothing one scale could reproduce."""
    return calibrate.PlaneCalibration(
        matrix=(0.2, 0.05, 0.0, 0.02, 0.15, 0.0, 0.0, 0.0, 1.0),
        marker_mm=calibrate.MARKER_MM,
        centre_px=(0.0, 0.0),
        rotation_deg=0.0,
        sides_px=(485.0, 485.0, 485.0, 485.0),
    )


# Rows must use the plane maths, not one scale.
def test_rows_use_the_homography_calibration():
    plane = make_plane()
    rows = dashboard.DashboardModel(plane).refresh([make_detection(0, (10.0, 5.0), 0.0)])
    assert (rows[0].x_mm, rows[0].y_mm) == plane.to_mm((10.0, 5.0))
    assert (rows[0].x_mm, rows[0].y_mm) != pipeline.CoordinateMapper(
        plane.as_scalar_calibration()).to_mm((10.0, 5.0))


def test_rows_fold_the_angle_through_the_plane():
    plane = make_plane()
    corners = ((0.0, 0.0), (40.0, 10.0), (36.0, 30.0), (-4.0, 20.0))  # a long box
    # A triangle now sends its folded heading, the same way the square does; only
    # a circle is held at 0.0, because it has no heading to send.
    rows = dashboard.DashboardModel(plane).refresh(
        [pipeline.Detection(4, 0.9, (16.0, 15.0), 0.0, corners)])
    assert rows[0].shape == "triangle"
    assert rows[0].angle_deg == pytest.approx(
        calibrate.pick_angle("triangle", plane.angle_deg(corners)))
    assert rows[0].angle_deg != 0.0
    # A square is, and the fold still has to reach it through the same plane.
    square = dashboard.DashboardModel(plane).refresh(
        [pipeline.Detection(1, 0.9, (16.0, 15.0), 0.0, corners)])
    assert square[0].angle_deg == calibrate.pick_angle("square", plane.angle_deg(corners))
    assert square[0].angle_deg != 0.0


def test_rows_name_the_colour_of_the_crop():
    frame = np.full((16, 16, 3), (0, 0, 255), np.uint8)  # red, BGR
    model = make_model()
    rows = model.refresh(
        [
            pipeline.Detection(
                0,
                0.9,
                (8.0, 8.0),
                0.0,
                ((0.0, 0.0), (16.0, 0.0), (16.0, 16.0), (0.0, 16.0)),
            )
        ],
        frame,
    )
    assert rows[0].colour == "red"
    assert rows[0].shape == "circle"


def test_rows_are_empty_until_a_frame_arrives():
    assert make_model().rows() == ()


def test_rows_keep_detection_order():
    model = make_model()
    model.refresh(
        [make_detection(3, (1.0, 1.0), 0.0), make_detection(1, (2.0, 2.0), 0.0)]
    )
    assert [row.shape for row in model.rows()] == ["star", "square"]


# A circle has no corners, so its angle stays zero.
def test_rows_carry_the_frame_angle_and_never_rotate_a_circle():
    model = make_model(rotation_deg=40.0)
    rows = model.refresh(
        [make_detection(2, (1.0, 1.0), 40.0), make_detection(0, (2.0, 2.0), 40.0)]
    )
    assert rows[0].angle_deg == 0.0
    assert rows[1].angle_deg == 0.0


def test_select_accepts_a_known_shape_and_none():
    model = make_model()
    model.select("triangle")
    assert model.selected == "triangle"
    model.select(None)
    assert model.selected is None


def test_select_rejects_an_unknown_shape():
    model = make_model()
    with pytest.raises(ValueError):
        model.select("octagon")


def test_choose_returns_none_without_detections():
    model = make_model()
    model.refresh([])
    assert model.choose() is None


# With no filter, the best scoring part wins.
def test_choose_takes_the_most_confident_shape():
    model = make_model()
    model.refresh(
        [
            make_detection(3, (1.0, 1.0), 0.0, confidence=0.4),
            make_detection(1, (5.0, 5.0), 0.0, confidence=0.95),
        ]
    )
    command = model.choose()
    assert command is not None
    assert command.shape == "square"


# A chosen shape beats the score order.
def test_choose_honours_the_selected_shape():
    model = make_model()
    model.refresh(
        [
            make_detection(3, (1.0, 1.0), 0.0, confidence=0.4),
            make_detection(1, (5.0, 5.0), 0.0, confidence=0.95),
        ]
    )
    model.select("star")
    command = model.choose()
    assert command is not None
    assert command.shape == "star"


def test_choose_returns_none_when_the_selected_shape_is_absent():
    model = make_model()
    model.refresh([make_detection(1, (5.0, 5.0), 0.0)])
    model.select("triangle")
    assert model.choose() is None


# OpenCV stores colour as blue, green, red.
RED_BGR = (0, 0, 255)
BLUE_BGR = (255, 0, 0)
GREY_BGR = (128, 128, 128)


def boxed_frame(boxes, size=32):
    """A grey tray with one solid colour painted into each box."""
    frame = np.full((size, size, 3), GREY_BGR, np.uint8)
    for box, colour in boxes:
        left, top, right, bottom = box
        frame[top:bottom, left:right] = colour
    return frame


def detection_in(box, class_id, confidence=0.9):
    """A Detection whose box is a real rectangle, so the crop can be read."""
    left, top, right, bottom = box
    centre = ((left + right) / 2.0, (top + bottom) / 2.0)
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))
    return pipeline.Detection(class_id, confidence, centre, 0.0, corners)


def test_select_colour_accepts_a_known_colour_and_none():
    model = make_model()
    model.select_colour("green")
    assert model.selected_colour == "green"
    model.select_colour(None)
    assert model.selected_colour is None


def test_select_colour_rejects_an_unknown_colour():
    model = make_model()
    with pytest.raises(ValueError):
        model.select_colour("purple")


def test_choose_honours_the_selected_colour():
    red_box, blue_box = (2, 2, 10, 10), (20, 20, 28, 28)
    frame = boxed_frame([(red_box, RED_BGR), (blue_box, BLUE_BGR)])
    model = make_model()
    model.refresh(
        [
            detection_in(red_box, 0, confidence=0.4),
            detection_in(blue_box, 3, confidence=0.95),
        ],
        frame,
    )

    model.select_colour("red")
    red = model.choose()
    assert red is not None and red.shape == "circle"

    model.select_colour("blue")
    blue = model.choose()
    assert blue is not None and blue.shape == "star"

    model.select_colour("green")
    assert model.choose() is None


def test_shape_and_colour_filters_combine():
    red_box, blue_box = (2, 2, 10, 10), (20, 20, 28, 28)
    frame = boxed_frame([(red_box, RED_BGR), (blue_box, BLUE_BGR)])
    model = make_model()
    model.refresh([detection_in(red_box, 0), detection_in(blue_box, 3)], frame)

    model.select("star")
    model.select_colour("red")
    assert model.choose() is None  # a red circle is not a red star

    model.select("circle")
    command = model.choose()
    assert command is not None and command.shape == "circle"


def test_a_pick_without_a_frame_has_an_unknown_colour():
    model = make_model()
    model.refresh([make_detection(0, (1.0, 1.0), 0.0)])
    model.select_colour("red")
    assert model.choose() is None
    assert model.rows()[0].colour == pipeline.UNKNOWN_COLOUR


class FakeDetector:
    """The window only calls detect(); no model weights in a UI test."""

    def __init__(self, *detections):
        self.detections = list(detections)
        self.seen = []

    def detect(self, image):
        self.seen.append(image)
        return list(self.detections)


def wait_until(predicate, timeout=2.0):
    """Poll a condition; the only timer a thread test needs."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def write_frames(folder, values):
    """One solid-colour PNG per value, named so the source reads them in order."""
    for value in values:
        cv2.imwrite(
            str(folder / ("frame_" + str(value) + ".png")),
            np.full((16, 16, 3), value, np.uint8),
        )


def test_worker_streams_the_newest_frame_and_detections(tmp_path):
    write_frames(tmp_path, [1, 2, 3])
    detector = FakeDetector(make_detection(1, (1.0, 1.0), 0.0))
    worker = dashboard.PipelineWorker(
        pipeline.ImageFolderSource(str(tmp_path)), detector
    )
    worker.start()
    try:
        # The queue holds one frame, so the fake may see fewer than three:
        # what matters is that it never sees an older frame after a newer one.
        assert wait_until(lambda: detector.seen and detector.seen[-1][0, 0, 0] == 3)
    finally:
        worker.stop()
    assert worker.error is None
    values = [frame[0, 0, 0] for frame in detector.seen]
    assert values == sorted(values)
    # One slot only, so old frames get dropped.
    assert worker.frames.maxsize == 1


def test_worker_reports_a_detector_failure_instead_of_dying_silently(tmp_path):
    write_frames(tmp_path, [1])

    class BrokenDetector:
        def detect(self, image):
            raise RuntimeError("model exploded")

    worker = dashboard.PipelineWorker(
        pipeline.ImageFolderSource(str(tmp_path)), BrokenDetector()
    )
    worker.start()
    try:
        assert wait_until(lambda: worker.error is not None)
    finally:
        worker.stop()
    assert isinstance(worker.error, RuntimeError)
    assert "model exploded" in str(worker.error)


# Stop must end a source that loops forever.
def test_worker_stops_a_looping_source(tmp_path):
    write_frames(tmp_path, [1])
    worker = dashboard.PipelineWorker(
        pipeline.VideoFileSource(str(tmp_path / "frame_1.png"), loop=True),
        FakeDetector(make_detection(0, (1.0, 1.0), 0.0)),
    )
    worker.start()
    time.sleep(0.05)
    worker.stop()
    assert worker._threads == []


def command():
    return pipeline.PickCommand("circle", 1.0, 2.0, 3.0)


def test_sender_delivers_a_queued_command():
    sent = []
    sender = dashboard.CommandSender(sent.append)
    sender.start()
    try:
        sender.offer(command())
        assert wait_until(lambda: sent == [command()])
    finally:
        sender.stop()
    assert sender.error is None


# A newer command replaces the waiting one.
def test_sender_keeps_only_the_newest_command():
    sender = dashboard.CommandSender(lambda sent: None)
    sender.offer(command())
    newest = pipeline.PickCommand("star", 9.0, 9.0, 0.0)
    sender.offer(newest)
    assert dashboard.take_latest(sender.commands) == newest


def test_sender_records_a_failed_send():
    def broken(_command):
        raise RuntimeError("robot is gone")

    sender = dashboard.CommandSender(broken)
    sender.start()
    try:
        sender.offer(command())
        assert wait_until(lambda: sender.error is not None)
    finally:
        sender.stop()
    assert "robot is gone" in str(sender.error)


def test_window_hands_the_pick_to_the_sender(tmp_path, tk_root):
    cv2.imwrite(str(tmp_path / "frame.png"), np.full((16, 16, 3), 7, np.uint8))
    sent = []
    sender = dashboard.CommandSender(sent.append)
    window = dashboard.DashboardWindow(
        make_model(),
        FakeDetector(make_detection(1, (4.0, 6.0), 0.0)),
        pipeline.ImageFolderSource(str(tmp_path)),
        sender=sender,
        root=tk_root,
    )
    window.start_worker()  # no worker here, so only comms starts
    try:
        window.load_next()
        window.pick()
        assert wait_until(
            lambda: sent == [pipeline.PickCommand("square", 4.0, 6.0, 0.0)]
        )
    finally:
        sender.stop()


def test_window_shows_a_frame_and_sends_the_pick(tmp_path, tk_root):
    """The window is thin, so one smoke test proves it is wired to the model."""
    cv2.imwrite(str(tmp_path / "frame.png"), np.full((16, 16, 3), 7, np.uint8))
    sent = []
    window = dashboard.DashboardWindow(
        make_model(),
        FakeDetector(make_detection(1, (4.0, 6.0), 0.0)),
        pipeline.ImageFolderSource(str(tmp_path)),
        send=sent.append,
        root=tk_root,
    )
    window.load_next()
    assert window.rows_list.size() == 1
    assert window.rows_list.get(0).startswith("square")

    window.pick()
    assert sent == [pipeline.PickCommand("square", 4.0, 6.0, 0.0)]
    assert window.command_var.get() == "2;4.0;6.0;10.0;0.0"


def test_window_filter_warns_instead_of_inventing_a_pick(tmp_path, tk_root):
    cv2.imwrite(str(tmp_path / "frame.png"), np.full((16, 16, 3), 7, np.uint8))
    sent = []
    window = dashboard.DashboardWindow(
        make_model(),
        FakeDetector(make_detection(1, (4.0, 6.0), 0.0)),
        pipeline.ImageFolderSource(str(tmp_path)),
        send=sent.append,
        root=tk_root,
    )
    window.load_next()
    window.shape_var.set("star")
    window._on_shape()
    window.pick()
    assert sent == []
    assert "nothing matches" in window.status_var.get()


def test_window_offers_and_applies_the_colour_filter(tmp_path, tk_root):
    cv2.imwrite(str(tmp_path / "frame.png"), np.full((16, 16, 3), 7, np.uint8))
    sent = []
    window = dashboard.DashboardWindow(
        make_model(),
        FakeDetector(make_detection(1, (4.0, 6.0), 0.0)),
        pipeline.ImageFolderSource(str(tmp_path)),
        send=sent.append,
        root=tk_root,
    )
    assert list(window.colour_box["values"]) == list(dashboard.COLOUR_CHOICES)
    assert window.colour_var.get() == "any"

    window.load_next()
    assert "black" in window.rows_list.get(0)  # a near-black frame is a black part

    # Red filter hides the black part, so nothing matches.
    window.colour_box.set("red")
    window._on_colour()
    window.pick()
    assert window.model.selected_colour == "red"
    assert sent == []
    assert "nothing matches" in window.status_var.get()


class FakeWorker(dashboard.PipelineWorker):
    """Records start/stop, so Quit can be watched without a camera.

    Subclasses the real worker for its types only; every method Quit or the
    poll touches is replaced, so no thread is ever created.
    """

    error = None

    def __init__(self):   # no source, no detector, no threads
        self._threads = []   # `running` reads this, and Nothing is alive
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def latest_result(self):
        return None


def test_quit_stops_the_worker_before_the_window_goes(tmp_path, tk_root, monkeypatch):
    """Quit must stop inference, not leave it reading the camera headless."""
    cv2.imwrite(str(tmp_path / "frame.png"), np.full((16, 16, 3), 7, np.uint8))
    destroyed = []
    monkeypatch.setattr(tk_root, "destroy", lambda: destroyed.append(True))
    worker = FakeWorker()
    window = dashboard.DashboardWindow(
        make_model(),
        FakeDetector(make_detection(1, (4.0, 6.0), 0.0)),
        pipeline.ImageFolderSource(str(tmp_path)),
        worker=worker,
        root=tk_root,
    )

    assert tk_root.protocol("WM_DELETE_WINDOW")   # the cross takes the same path
    window.start_worker()
    window.quit_button.invoke()
    window.close()          # a second close must not stop anything twice

    assert worker.started == 1
    assert worker.stopped == 1
    assert destroyed == [True]
    window.drain()          # a poll queued before the quit does nothing
    assert window.status_var.get() == "load a frame to start"


def test_a_second_pick_takes_the_next_matching_part():
    """The emptied point must not be sent twice: the row leaves the list."""
    model = make_model()
    model.refresh([make_detection(0, (10.0, 5.0), 0.0),
                   make_detection(0, (40.0, 5.0), 0.0)])

    first = model.choose(consume=True)
    second = model.choose(consume=True)
    third = model.choose(consume=True)

    assert first is not None and second is not None
    assert (first.x_mm, second.x_mm) == (10.0, 40.0)
    assert third is None
    assert model.rows() == ()


def make_fake_pylon(frame):
    """Only the pypylon API `BaslerSource` touches, so no camera is needed."""

    class Device:
        def GetModelName(self):
            return "fake-basler"

    class Grab:
        def GrabSucceeded(self):
            return True

        def GetArray(self):
            return frame

        def Release(self):
            pass

    class Converter:
        def Convert(self, result):
            return result

    class Camera:
        def __init__(self, device):
            self.opened = False
            self.grabbing = False

        def RegisterConfiguration(self, *args):
            # The exact failure pypylon 3.x raised in the cell when handed the
            # handler class: Next frame died inside open(). Nothing may call it.
            raise TypeError(
                "ConfigurationEventHandler.__disown__() missing 1 required "
                "positional argument: 'self'"
            )

        def Open(self):
            self.opened = True

        def IsOpen(self):
            return self.opened

        def Close(self):
            self.opened = False

        def GetDeviceInfo(self):
            return Device()

        def StartGrabbing(self, *args):
            self.grabbing = True

        def IsGrabbing(self):
            return self.grabbing

        def StopGrabbing(self):
            self.grabbing = False

        def RetrieveResult(self, timeout_ms, handling):
            return Grab()

    class Factory:
        @staticmethod
        def GetInstance():
            return Factory()

        def EnumerateDevices(self):
            return [Device()]

        def CreateDevice(self, device):
            return device

    return SimpleNamespace(
        TlFactory=Factory,
        InstantCamera=Camera,
        ImageFormatConverter=Converter,
        AcquireContinuousConfiguration=object(),
        RegistrationMode_ReplaceAll=object(),
        Cleanup_Delete=object(),
        GrabStrategy_LatestImageOnly=object(),
        TimeoutHandling_ThrowException=object(),
        PixelType_BGR8packed=object(),
        OutputBitAlignment_MsbAligned=object(),
    )


# Copy camera frames and allow repeated camera cleanup.
def test_basler_source_reads_a_private_copy_and_closes():
    frame = np.full((4, 4, 3), 7, np.uint8)
    source = camera.BaslerSource(pylon=make_fake_pylon(frame))
    read = source.read()
    assert read is not None
    assert read is not frame
    assert np.array_equal(read, frame)
    frame[:] = 0
    assert np.array_equal(read, np.full((4, 4, 3), 7, np.uint8))
    source.close()
    source.close()


def test_window_refuses_a_pick_when_the_link_is_down(tmp_path, tk_root):
    """A dead sender must not let the window claim a pick the robot never got."""

    def broken(_command):
        raise RuntimeError("robot is gone")

    sender = dashboard.CommandSender(broken)
    sender.start()
    try:
        sender.offer(command())
        assert wait_until(lambda: sender.broken)
    finally:
        sender.stop()

    cv2.imwrite(str(tmp_path / "frame.png"), np.full((16, 16, 3), 7, np.uint8))
    window = dashboard.DashboardWindow(
        make_model(),
        FakeDetector(make_detection(1, (4.0, 6.0), 0.0)),
        pipeline.ImageFolderSource(str(tmp_path)),
        sender=sender,
        root=tk_root,
    )
    window.load_next()
    window.pick()

    assert "link is down" in window.status_var.get()
    assert window.command_var.get() == "no command yet"
    assert sender.commands.empty()


def test_save_frame_writes_the_full_frame_twice(tmp_path, tk_root):
    """Two presses mean two files, at capture resolution rather than preview."""
    cv2.imwrite(str(tmp_path / "frame.png"), np.full((16, 16, 3), 7, np.uint8))
    frame = np.full((20, 1000, 3), 9, np.uint8)
    saver = dashboard.FrameSaver(tmp_path / "captures")
    window = dashboard.DashboardWindow(
        make_model(),
        FakeDetector(make_detection(1, (4.0, 6.0), 0.0)),
        pipeline.ImageFolderSource(str(tmp_path)),
        saver=saver,
        root=tk_root,
    )
    window.start_worker()  # no worker and no sender, so only the saver starts
    try:
        window.show_frame(frame)
        window.save_frame()
        window.save_frame()
        assert "saving" in window.status_var.get()
    finally:
        saver.stop()

    written = sorted((tmp_path / "captures").glob("*.png"))
    assert len(written) == 2
    for path in written:
        saved = cv2.imread(str(path))
        assert saved is not None
        assert saved.shape == frame.shape
        assert np.array_equal(saved, frame)


def test_save_frame_without_record_explains_itself(tmp_path, tk_root):
    cv2.imwrite(str(tmp_path / "frame.png"), np.full((16, 16, 3), 7, np.uint8))
    window = dashboard.DashboardWindow(
        make_model(),
        FakeDetector(make_detection(1, (4.0, 6.0), 0.0)),
        pipeline.ImageFolderSource(str(tmp_path)),
        root=tk_root,
    )
    window.show_frame(np.zeros((4, 4, 3), np.uint8))
    window.save_frame()
    assert "--record" in window.status_var.get()


def test_record_needs_the_camera(tmp_path):
    """Capturing into the folder being replayed would copy the test set."""
    with pytest.raises(SystemExit) as stop:
        dashboard.main(["--record", "--folder", str(tmp_path)])

    assert "needs --camera" in str(stop.value)
