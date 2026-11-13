"""Live camera view in a browser:  http://<pi-address>:8080

Viewing only - nothing is ever sent to the robot.
It opens the camera, so stop dashboard.py / live_pick.py first
(a Basler camera can only be used by one program at a time).

    python scripts/web_view.py            # camera + detections + mm table
    python scripts/web_view.py --raw      # camera only, smooth, no detection
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

import pipeline
from calibrate import load_calibration
from camera import BaslerSource
from dashboard_new import DashboardModel, draw_labels

STREAM_WIDTH = 1224
JPEG_QUALITY = 80


class Latest:
    """The newest JPEG and table, shared between the camera thread and the web server."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.rows: list[dict] = []
        self.status = "starting camera..."
        self.frame_id = 0

    def set(self, jpeg: bytes | None, rows: list[dict], status: str) -> None:
        with self.lock:
            self.jpeg, self.rows, self.status = jpeg, rows, status
            self.frame_id += 1

    def get(self) -> tuple[bytes | None, list[dict], str, int]:
        with self.lock:
            return self.jpeg, self.rows, self.status, self.frame_id


# One shared copy of the newest picture and table.
LATEST = Latest()


# Shrink the picture and pack it as JPEG bytes.
def to_jpeg(image) -> bytes:
    height, width = image.shape[:2]
    if width > STREAM_WIDTH:
        scale = STREAM_WIDTH / width
        image = cv2.resize(image, (STREAM_WIDTH, int(height * scale)))
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buffer.tobytes() if ok else b""


# Camera thread: read a frame, detect parts, publish it.
def camera_loop(raw: bool, model_path: str, conf: float, stop: threading.Event) -> None:
    model = None
    detector = None
    if not raw:
        print(pipeline.apply_angle_teach())
        try:
            model = DashboardModel(load_calibration())
            detector = pipeline.ShapeDetector(model_path=model_path, conf=conf)
        except (FileNotFoundError, ValueError) as error:
            print(f"no usable calibration ({error}) - showing the raw camera only")
            model = detector = None
    try:
        # Prepare a camera source; reading opens it when needed.
        source = BaslerSource()
    except Exception as error:  # camera busy / missing: show it on the page
        LATEST.set(None, [], f"camera error: {error} - close dashboard/live_pick and retry")
        print(f"camera error: {error}")
        return
    try:
        while not stop.is_set():
            frame = source.read()
            if frame is None:
                LATEST.set(None, [], "camera returned no frame")
                time.sleep(0.5)
                continue
            if detector is None or model is None:
                LATEST.set(to_jpeg(frame), [], "raw camera view (no detection)")
                continue
            # Time the detection so the page can show it.
            started = time.perf_counter()
            shown = model.refresh(detector.detect(frame), frame)
            elapsed = (time.perf_counter() - started) * 1000.0
            # One table row per part: shape, x/y mm, angle.
            rows = [{
                "shape": row.shape,
                "colour": row.colour,
                "conf": round(row.confidence, 2),
                "x": round(row.x_mm, 1),
                "y": round(row.y_mm, 1),
                "angle": round(row.angle_deg, 1),
                "line": (f"{pipeline.SLOT[row.shape]};{row.x_mm:.1f};{row.y_mm:.1f};"
                         f"{pipeline.PICK_Z_MM:.1f};{row.angle_deg:.1f}"),
                "blocked": pipeline.pick_blocked(row.shape) or "",
            } for row in shown]
            LATEST.set(to_jpeg(draw_labels(frame, model)), rows,
                       f"{len(rows)} part(s), detection {elapsed:.0f} ms")
    except Exception as error:
        LATEST.set(None, [], f"stopped: {error}")
        print(f"camera loop stopped: {error}")
    finally:
        # Always close the camera when the loop ends.
        source.close()


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>INA700 camera</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}
img{max-width:100%;border:1px solid #444;background:#000}
table{border-collapse:collapse;margin-top:10px;font-family:monospace}
td,th{border:1px solid #444;padding:4px 8px;text-align:right}
th{background:#222} .bad{color:#f66} #status{color:#9cf;margin:8px 0}
</style></head><body>
<h3>INA700 camera - view only, nothing is sent to the robot</h3>
<div id="status">connecting...</div>
<img src="/stream" alt="camera stream">
<table><thead><tr><th>shape</th><th>colour</th><th>conf</th><th>x mm</th>
<th>y mm</th><th>angle</th><th>wire line</th><th>note</th></tr></thead>
<tbody id="rows"></tbody></table>
<script>
async function poll(){
  try{
    const r=await fetch('/rows'); const d=await r.json();
    document.getElementById('status').textContent=d.status;
    document.getElementById('rows').innerHTML=d.rows.map(x=>
      `<tr><td>${x.shape}</td><td>${x.colour}</td><td>${x.conf}</td><td>${x.x}</td>
       <td>${x.y}</td><td>${x.angle}</td><td>${x.line}</td>
       <td class="bad">${x.blocked}</td></tr>`).join('');
  }catch(e){document.getElementById('status').textContent='lost connection to the Pi';}
}
setInterval(poll,1000); poll();
</script></body></html>"""


# Web server: sends the page, the table and the video.
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the terminal quiet
        return

    # Choose what to send for each web address.
    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif self.path == "/rows":
            _jpeg, rows, status, _id = LATEST.get()
            body = json.dumps({"status": status, "rows": rows}).encode("utf-8")
            self._send(200, "application/json", body)
        elif self.path == "/snapshot.jpg":
            jpeg, *_ = LATEST.get()
            if jpeg:
                self._send(200, "image/jpeg", jpeg)
            else:
                self._send(503, "text/plain", b"no frame yet")
        elif self.path == "/stream":
            self._stream()
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code: int, kind: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # Live video: send each new picture as it arrives.
    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                jpeg, _rows, _status, frame_id = LATEST.get()
                if jpeg and frame_id != last:
                    last = frame_id
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                     + jpeg + b"\r\n")
                    self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            return


def addresses() -> list[str]:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout
        return out.split()
    except OSError:
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="live camera view in a web browser")
    parser.add_argument("--port", type=int, default=8080, help="web port (default 8080)")
    parser.add_argument("--raw", action="store_true",
                        help="camera only, no detection (smooth, for aiming the camera)")
    ncnn = pipeline.MODEL_PATH.parent / "best_ncnn_model"
    default_model = ncnn if ncnn.is_dir() else pipeline.MODEL_PATH
    parser.add_argument("--model", default=str(default_model), help="OBB weights")
    parser.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    arguments = parser.parse_args(argv)

    # Shared switch that tells the camera thread to stop.
    stop = threading.Event()
    thread = threading.Thread(target=camera_loop, name="camera", daemon=True,
                              args=(arguments.raw, arguments.model, arguments.conf, stop))
    thread.start()

    # Listen on every network card so other PCs can connect.
    server = ThreadingHTTPServer(("0.0.0.0", arguments.port), Handler)
    server.daemon_threads = True
    print("open in a browser on your PC:")
    for address in addresses() or ["<pi-address>"]:
        print(f"    http://{address}:{arguments.port}")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        thread.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
