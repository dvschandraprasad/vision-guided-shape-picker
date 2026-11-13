import pipeline
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO

# Find YOLO oriented-box detection on camera or files.
MODEL_PATH = pipeline.MODEL_PATH
SOURCE = "basler"

CAMERA_BACKEND = cv2.CAP_DSHOW

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})

# Minimum detection confidence; lower value keeps more boxes.
CONF = 0.25

PRINT_INTERVAL_S = 1.0

SHOW_LABELS = True

class Frame(NamedTuple):

    label: str
    image: np.ndarray

def load_model(model_path: str) -> "YOLO":
    from ultralytics import YOLO

    return YOLO(model_path)

def class_names(model: "YOLO") -> dict[int, str]:
    return model.names

# Generator: frames from the Basler factory camera.
def basler_frames() -> Iterator[Frame]:
    from pypylon import pylon

    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        raise RuntimeError(
            "No Basler camera detected. Check power, cable and that Pylon "
            "Viewer is closed (it locks the camera)."
        )
    if len(devices) > 1:
        print(f"WARNING: {len(devices)} Basler cameras found; using the first.")

    # Select the first camera found on this computer.
    camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    try:
        camera.RegisterConfiguration(
            pylon.AcquireContinuousConfiguration(),
            pylon.RegistrationMode_ReplaceAll,
            pylon.Cleanup_Delete,
        )
        camera.Open()
        print("Camera:", camera.GetDeviceInfo().GetModelName())
        # Start grabbing; only the newest frame is kept.
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        count = 0

        while camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            try:
                if not grab.GrabSucceeded():
                    raise RuntimeError(
                        f"Frame acquisition failed: {grab.GetErrorDescription()}"
                    )
                yield Frame(
                    f"basler#{count:06d}", converter.Convert(grab).GetArray().copy()
                )
            finally:
                grab.Release()
            count += 1
    finally:
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        finally:
            # Always close the camera, even after an error.
            if camera.IsOpen():
                camera.Close()

# Generator: frames from a USB or laptop webcam.
def webcam_frames(index: int) -> Iterator[Frame]:
    capture = cv2.VideoCapture(index, CAMERA_BACKEND)
    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index}. Plug the camera in and "
                "close any other app that is using it, then try the next index."
            )
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Camera index {index} stopped delivering frames.")
            yield Frame(f"camera{index}#{count:06d}", frame)
            count += 1
    finally:
        capture.release()

def file_frames(source: str) -> Iterator[Frame]:
    path = Path(source)
    if path.is_dir():
        # Sort image files so frames play in name order.
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise RuntimeError(f"No images in {path}.")
        for file in files:
            image = cv2.imread(str(file))
            if image is None:
                raise RuntimeError(f"Could not read image: {file}")
            yield Frame(file.name, image)
        return

    if not path.exists():
        raise RuntimeError(f"Source not found: {path}")

    image = cv2.imread(str(path))
    if image is not None:
        yield Frame(path.name, image)
        return

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield Frame(f"{path.name}#{count:06d}", frame)
            count += 1
    finally:
        capture.release()

# True when source is a live camera, not a file.
def is_live(source: str) -> bool:
    text = source.strip().lower()
    return text == "basler" or text.isdigit()

# Pick camera or file source, then give frames.
def frame_stream(source: str) -> Iterator[Frame]:
    if source.strip().lower() == "basler":
        yield from basler_frames()
    elif source.strip().isdigit():
        yield from webcam_frames(int(source))
    else:
        yield from file_frames(source)

def report(result: Any, names: dict[int, str], label: str = "image") -> None:
    obb = result.obb
    if obb is None or len(obb) == 0:
        print(f"{label}: no detections")
        return
    for xywhr, conf, cls in zip(
        obb.xywhr.tolist(), obb.conf.tolist(), obb.cls.tolist(), strict=True
    ):
        # Box is center x, y, width, height and angle.
        x, y, w, h, angle = (float(v) for v in xywhr)
        name = names.get(int(cls), str(cls))
        print(
            f"{label}: [{name}] conf={conf:.2f} "
            f"angle={angle:.2f} "
            f"box=({x:.0f},{y:.0f},{w:.0f},{h:.0f})"
        )

# Main loop: predict boxes, draw them, show window.
def detect(model: "YOLO", source: str) -> None:
    names = class_names(model)
    window = f"YOLO26-nano OBB - {source}"
    live = is_live(source)
    last_print = 0.0
    count = 0
    interval_start = time.monotonic()
    fps = 0.0
    try:
        for frame in frame_stream(source):
            predictions: Any = model.predict(
                source=frame.image, conf=CONF, verbose=False
            )
            result = predictions[0]
            now = time.monotonic()
            if not live or now - last_print >= PRINT_INTERVAL_S:
                report(result, names, frame.label)
                last_print = now
            preview = result.plot(labels=SHOW_LABELS, boxes=True)
            if live:
                count += 1
                elapsed = now - interval_start
                if elapsed >= 1.0:
                    fps = count / elapsed
                    count = 0
                    interval_start = now
                cv2.putText(
                    preview,
                    f"{fps:.1f} FPS",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow(window, preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Close preview windows when the loop ends.
        cv2.destroyAllWindows()

def main() -> None:
    model = load_model(str(MODEL_PATH))
    detect(model, SOURCE)

if __name__ == "__main__":
    try:
        main()
    # Ctrl+C stops the program without a traceback.
    except KeyboardInterrupt:
        print("\nStopped.")
