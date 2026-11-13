import pipeline
from datetime import datetime
from pathlib import Path
import time

import cv2
from pypylon import pylon

# Folder where frames saved with S go.
SAVE_DIR = pipeline.REPO_ROOT / "data" / "captured_frames"
WINDOWS = ("Basler - Live", "OpenCV - Edges")

# Turns one image into edge lines for preview.
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def main():
    camera = None
    try:
        # Ask Basler driver which cameras are connected.
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera detected. Check power, cable and Pylon Viewer.")
        if len(devices) > 1:
            raise RuntimeError("Connect only the intended camera for this starter script.")

        camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
        camera.RegisterConfiguration(
            pylon.AcquireContinuousConfiguration(),
            pylon.RegistrationMode_ReplaceAll,
            pylon.Cleanup_Delete,
        )
        # Open camera once before taking any frames.
        camera.Open()
        print("Camera:", camera.GetDeviceInfo().GetModelName())

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        for name in WINDOWS:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 800, 600)

        # Use newest frame only; older frames are dropped.
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print("Click a preview window: S saves a frame; Q or Esc quits.")
        count = 0
        fps = 0.0
        interval_start = time.perf_counter()

        while camera.IsGrabbing():
            # Wait up to five seconds for the next frame.
            result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            try:
                if not result.GrabSucceeded():
                    raise RuntimeError(f"Frame acquisition failed: {result.GetErrorDescription()}")
                # Convert camera image to colour for OpenCV.
                converted = converter.Convert(result)
                frame = converted.GetArray().copy()
            finally:
                result.Release()

            edges = process_frame(frame)

            count += 1
            elapsed = time.perf_counter() - interval_start
            # Count frames over one second to show speed.
            if elapsed >= 1.0:
                fps = count / elapsed
                count = 0
                interval_start = time.perf_counter()

            preview = frame.copy()
            cv2.putText(preview, f"Loop FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(WINDOWS[0], preview)
            cv2.imshow(WINDOWS[1], edges)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                # Create the save folder only when first needed.
                SAVE_DIR.mkdir(parents=True, exist_ok=True)
                # Time stamp keeps each saved file name unique.
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = SAVE_DIR / f"frame_{stamp}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not save image: {path}")
                print("Saved:", path)

            if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                   for name in WINDOWS):
                break
    finally:
        try:
            # Always close camera, even after an error.
            if camera is not None:
                try:
                    if camera.IsGrabbing():
                        camera.StopGrabbing()
                finally:
                    if camera.IsOpen():
                        camera.Close()
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as error:
        print(f"Error: {error}")
        raise SystemExit(1)
