"""Headless tests for yolo_obb.py's source handling.

Nothing here needs a camera, a display, or the trained model: `import yolo_obb`
is cheap because ultralytics is imported lazily inside `load_model`.

Run:
    .venv/Scripts/python.exe -m pytest test_yolo_obb.py -q
"""

import cv2
import numpy as np
import pytest

import yolo_obb


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


# A folder is read in sorted file-name order.
def test_a_directory_yields_its_images_in_name_order(tmp_path):
    first = write_image(tmp_path / "b.png", 10)
    second = write_image(tmp_path / "a.png", 20)
    (tmp_path / "notes.txt").write_text("not an image")

    frames = list(yolo_obb.file_frames(str(tmp_path)))

    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0].image, second)
    np.testing.assert_array_equal(frames[1].image, first)


def test_frames_are_labelled_for_the_printed_output(tmp_path):
    write_image(tmp_path / "frame.png", 7)
    (frame,) = yolo_obb.file_frames(str(tmp_path / "frame.png"))
    assert frame.label == "frame.png"


def test_an_empty_directory_names_itself(tmp_path):
    with pytest.raises(RuntimeError, match="No images in"):
        list(yolo_obb.file_frames(str(tmp_path)))


def test_a_single_image_yields_one_frame(tmp_path):
    image = write_image(tmp_path / "frame.png", 7)
    (frame,) = yolo_obb.file_frames(str(tmp_path / "frame.png"))
    np.testing.assert_array_equal(frame.image, image)


# Each video frame gets its own numbered label.
def test_a_video_yields_numbered_frames(tmp_path):
    write_video(tmp_path / "clip.avi", [10, 20, 30])
    frames = list(yolo_obb.file_frames(str(tmp_path / "clip.avi")))
    assert len(frames) == 3
    assert all(frame.image.shape == (32, 32, 3) for frame in frames)
    assert [frame.label for frame in frames] == [
        "clip.avi#000000",
        "clip.avi#000001",
        "clip.avi#000002",
    ]


def test_a_missing_path_fails_loudly(tmp_path):
    # A bad path must fail loudly, not stay silent.
    with pytest.raises(RuntimeError, match="Source not found"):
        list(yolo_obb.file_frames(str(tmp_path / "nope.png")))


@pytest.mark.parametrize("source", ["basler", "Basler", " basler "])
def test_basler_source_uses_the_basler_camera(monkeypatch, source):
    # Fake camera frame, so no real hardware is used.
    sentinel = yolo_obb.Frame("basler#000000", np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(yolo_obb, "basler_frames", lambda: iter([sentinel]))
    assert list(yolo_obb.frame_stream(source)) == [sentinel]


@pytest.mark.parametrize("source", ["0", "1", " 3 "])
def test_a_digit_source_is_a_webcam_index(monkeypatch, source):
    seen = []
    monkeypatch.setattr(
        yolo_obb, "webcam_frames", lambda index: seen.append(index) or iter([index])
    )
    assert list(yolo_obb.frame_stream(source)) == [int(source)]
    # Spaces around the number are ignored.
    assert seen == [int(source)]


# Any other name is treated as a file path.
def test_a_path_source_reads_files(monkeypatch):
    monkeypatch.setattr(yolo_obb, "file_frames", lambda path: iter([path]))
    assert list(yolo_obb.frame_stream("captured_frames")) == ["captured_frames"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("basler", True),
        (" 0 ", True),
        ("12", True),
        ("captured_frames", False),
        ("frame.png", False),
    ],
)
def test_only_cameras_count_as_live(source, expected):
    """Live sources get the print throttle; files print every frame."""
    assert yolo_obb.is_live(source) is expected
