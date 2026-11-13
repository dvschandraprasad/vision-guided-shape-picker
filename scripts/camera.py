"""Basler live view: OpenCV windows (default) or a web page (--web).

    python scripts/camera.py              # two OpenCV windows on the Pi's screen
    python scripts/camera.py --web        # open http://<pi-address>:8080 on your PC

The camera can only be used by one program: stop the dashboard first.
"""
import pipeline
import argparse
import json
import subprocess
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from pipeline import FrameSource

SAVE_DIR = pipeline.REPO_ROOT / "data" / "captured_frames"
LIVE_WINDOW = "Basler - Live"
EDGE_WINDOW = "OpenCV - Edges"
WINDOWS = (LIVE_WINDOW, EDGE_WINDOW)


def _pypylon() -> Any:
    from pypylon import pylon
    return pylon


# Make a black and white picture of sharp edges.
def edges_of(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, 50, 150)


# Write one picture to the data folder and return its path.
def save_frame(frame: np.ndarray) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVE_DIR / f"frame_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not save image: {path}")
    return path


class BaslerSource(FrameSource):

    def __init__(self, pylon: Any | None = None, timeout_ms: int = 5000) -> None:
        self.timeout_ms = timeout_ms
        self._module: Any = pylon
        self._camera: Any = None
        self._converter: Any = None

    def open(self) -> None:
        module = self._module if self._module is not None else _pypylon()
        factory = module.TlFactory.GetInstance()
        # Ask the camera list how many cameras are plugged in.
        devices = factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera detected. Check power, cable and Pylon Viewer.")
        camera = module.InstantCamera(factory.CreateDevice(devices[0]))
        camera.Open()
        converter = module.ImageFormatConverter()
        # Ask for colour frames in the order OpenCV expects.
        converter.OutputPixelFormat = module.PixelType_BGR8packed
        converter.OutputBitAlignment = module.OutputBitAlignment_MsbAligned
        camera.StartGrabbing(module.GrabStrategy_LatestImageOnly)
        self._module = module
        self._camera = camera
        self._converter = converter
        print("Camera:", camera.GetDeviceInfo().GetModelName())

    def read(self) -> np.ndarray | None:
        # Open the camera on first use if it is still closed.
        camera: Any = self._camera
        if camera is None:
            self.open()
            camera = self._camera
        result = camera.RetrieveResult(
            self.timeout_ms, self._module.TimeoutHandling_ThrowException)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError(f"Frame acquisition failed: {result.GetErrorDescription()}")
            return self._converter.Convert(result).GetArray().copy()
        finally:
            result.Release()

    def close(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is None:
            return
        # Stop grabbing before closing, so other programs can use it.
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()


def preview(frame: np.ndarray, detector: Any, fps: float) -> None:
    marked = frame.copy()
    corners, ids, _rejected = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(marked, corners, ids)
    cv2.putText(marked, f"Loop FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(LIVE_WINDOW, marked)
    cv2.imshow(EDGE_WINDOW, edges_of(frame))


# --------------------------------------------------------------------------
# --web: the same live view in a browser (works over SSH with no screen)
# --------------------------------------------------------------------------
WEB_FPS = 10.0       # frames per second sent to the browser (keeps the Pi cool)
WEB_STEP = 2         # browser picture keeps every 2nd pixel: 2448x2048 -> 1224x1024
JPEG_QUALITY = 70

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Basler live</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}
.wrap{display:flex;gap:12px;flex-wrap:wrap} .wrap div{flex:1;min-width:320px}
img{width:100%;border:1px solid #444;background:#000}
button{font-size:15px;padding:6px 12px;border-radius:4px;border:1px solid #555;
       background:#b8860b;color:#000;font-weight:bold}
#status{color:#9cf;margin:8px 0} h3{margin:0 0 10px} .cap{color:#aaa;font-size:13px}
</style></head><body>
<h3>Basler - Live</h3>
<div><button onclick="save()">Save frame</button> <span id="status">connecting...</span></div>
<div class="wrap">
 <div><div class="cap">Live (ArUco markers outlined)</div><img id="live" alt="live"></div>
 <div><div class="cap">OpenCV - Edges</div><img id="edges" alt="edges"></div>
</div>
<script>
const imgs={live:document.getElementById('live'),edges:document.getElementById('edges')};
let boot=null,last=-1,staleSince=Date.now();
function open_(){for(const k in imgs) imgs[k].src='/stream/'+k+'?t='+Date.now(); staleSince=Date.now();}
for(const k in imgs) imgs[k].onerror=()=>setTimeout(open_,1000);
async function poll(){
  try{const d=await (await fetch('/state')).json();
    document.getElementById('status').textContent=d.status;
    if(d.boot!==boot){boot=d.boot; last=d.frame; open_(); return;}
    if(d.frame!==last){last=d.frame; staleSince=Date.now();}
    else if(Date.now()-staleSince>5000) open_();
  }catch(e){document.getElementById('status').textContent='lost connection to the Pi';}
}
async function save(){
  try{const d=await (await fetch('/save',{method:'POST'})).json();
      document.getElementById('status').textContent=d.status;}
  catch(e){document.getElementById('status').textContent='lost connection to the Pi';}
}
setInterval(poll,1000); poll();
</script></body></html>"""


class WebView:
    """Grabs frames in a thread and keeps the latest live/edge JPEGs for the page."""

    def __init__(self, source: BaslerSource, detector: Any, fps_cap: float = WEB_FPS) -> None:
        self.source = source
        self.detector = detector
        self.interval = 1.0 / fps_cap
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None   # full resolution, for Save
        self.jpegs: dict[str, bytes] = {}
        self.frame_id = 0
        self.fps = 0.0
        self.status = "starting camera..."
        self.error: Exception | None = None
        self.stop = threading.Event()
        self.boot = str(int(time.time() * 1000))
        self.thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        counted, started = 0, time.perf_counter()
        try:
            while not self.stop.is_set():
                tick = time.monotonic()
                frame = self.source.read()
                if frame is None:
                    self.status = "camera returned no frame"
                    return
                counted += 1
                elapsed = time.perf_counter() - started
                if elapsed >= 1.0:
                    self.fps, counted, started = counted / elapsed, 0, time.perf_counter()
                # Keep every second pixel to make a smaller browser picture.
                small = np.ascontiguousarray(frame[::WEB_STEP, ::WEB_STEP])
                live, markers = self._mark(small)
                encode = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                ok_live, live_jpg = cv2.imencode(".jpg", live, encode)
                ok_edge, edge_jpg = cv2.imencode(".jpg", edges_of(small), encode)
                with self.lock:
                    self.frame = frame
                    if ok_live and ok_edge:
                        self.jpegs = {"live": live_jpg.tobytes(), "edges": edge_jpg.tobytes()}
                        self.frame_id += 1
                    seen = ", ".join(str(m) for m in markers) or "none"
                    if not self.status.startswith("Saved"):
                        self.status = (f"{self.fps:.1f} FPS, {frame.shape[1]}x{frame.shape[0]},"
                                       f" ArUco seen: {seen}")
                # Pause to keep the loop at the chosen frame rate.
                self.stop.wait(max(0.0, self.interval - (time.monotonic() - tick)))
        except Exception as error:
            self.error = error
            self.status = f"camera stopped: {error}"

    def _mark(self, small: np.ndarray) -> tuple[np.ndarray, list[int]]:
        marked = small.copy()
        corners, ids, _rejected = self.detector.detectMarkers(
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        found: list[int] = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(marked, corners, ids)
            for marker_corners, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
                found.append(int(marker_id))
                # Average the four corners to find the marker centre.
                cx, cy = np.asarray(marker_corners).reshape(4, 2).mean(axis=0)
                centre = (int(round(cx)), int(round(cy)))
                cv2.drawMarker(marked, centre, (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(marked, f"centre ({cx * WEB_STEP:.0f}, {cy * WEB_STEP:.0f}) px",
                            (centre[0] + 10, centre[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(marked, f"Loop FPS: {self.fps:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return marked, found

    # Save the newest full size frame, not the small browser copy.
    def save(self) -> str:
        with self.lock:
            frame = self.frame
        if frame is None:
            return "no frame yet"
        path = save_frame(frame)  # full resolution, same as the S key
        print("Saved:", path)
        with self.lock:
            self.status = f"Saved: {path.name}"
        return self.status

    def latest(self, kind: str) -> tuple[bytes | None, int]:
        with self.lock:
            return self.jpegs.get(kind), self.frame_id


VIEW: WebView | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/state":
            with VIEW.lock:
                data = {"status": VIEW.status, "frame": VIEW.frame_id, "boot": VIEW.boot}
            self._send(200, "application/json", json.dumps(data).encode("utf-8"))
        elif path in ("/stream/live", "/stream/edges"):
            self._stream(path.rsplit("/", 1)[1])
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/save":
            self._send(404, "text/plain", b"not found")
            return
        try:
            status = VIEW.save()
        except Exception as error:
            status = f"save failed: {error}"
        self._send(200, "application/json", json.dumps({"status": status}).encode("utf-8"))

    def _send(self, code: int, kind: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, kind: str) -> None:
        self.send_response(200)
        # This header keeps one picture stream open in the browser.
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while not VIEW.stop.is_set():
                jpeg, frame_id = VIEW.latest(kind)
                if jpeg and frame_id != last:
                    last = frame_id
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                     + jpeg + b"\r\n")
                    self.wfile.flush()
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            return


def _addresses() -> list[str]:
    try:
        return subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    except OSError:
        return []


def run_web(source: BaslerSource, detector: Any, port: int, fps_cap: float) -> int:
    global VIEW
    VIEW = WebView(source, detector, fps_cap)
    VIEW.start()
    # Listen on all network cards so your PC can connect.
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    print("open the live view in a browser on your PC:")
    for address in _addresses() or ["<pi-address>"]:
        print(f"    http://{address}:{port}")
    print("Save frame button saves a full-size PNG; Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        VIEW.close()
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Basler live view")
    parser.add_argument("--web", action="store_true",
                        help="show the view in a web browser instead of OpenCV windows")
    parser.add_argument("--port", type=int, default=8080, help="browser port (default 8080)")
    parser.add_argument("--fps", type=float, default=WEB_FPS,
                        help="max frames per second in --web mode (default 10)")
    arguments = parser.parse_args(argv)

    source = BaslerSource()
    source.open()
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
        cv2.aruco.DetectorParameters())
    if arguments.web:
        try:
            return run_web(source, detector, arguments.port, arguments.fps)
        finally:
            source.close()
    for name in WINDOWS:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 800, 600)
    print("Click a preview window: S saves a frame; Q or Esc quits.")

    fps = 0.0
    counted = 0
    started = time.perf_counter()
    try:
        # Keep reading frames until the user quits the window.
        while True:
            frame = source.read()
            if frame is None:
                break
            counted += 1
            elapsed = time.perf_counter() - started
            if elapsed >= 1.0:
                fps = counted / elapsed
                counted = 0
                started = time.perf_counter()
            preview(frame, detector, fps)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                print("Saved:", save_frame(frame))
            if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in WINDOWS):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as error:
        print(f"Error: {error}")
        raise SystemExit(1)
