"""dashboard_new in a web browser:  http://<pi-address>:8080

The same dashboard as dashboard_new.py - live camera with labels, shape and
colour filter, Pick, Save frame, robot link - but the window is a web page, so
it runs over SSH with no screen and you use it from the browser on your PC.

    python scripts/dashboard_web.py --camera --send --host 192.168.125.205

Start this first, then the RAPID program within 30 s (as with dashboard_new).
The camera can only be used by one program: stop web_view / dashboard first.
Anyone who can open the page can press Pick - keep it on the lab network.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import pipeline
from calibrate import MARKER_ID, load_calibration, make_detector
from camera import BaslerSource
from dashboard_new import (
    COLOUR_CHOICES,
    RAW_DIR,
    ROBOT_HOST,
    ROBOT_PORT,
    SHAPE_CHOICES,
    CommandSender,
    DashboardModel,
    FrameSaver,
    PipelineWorker,
    ShapeRow,
    describe_filters,
    put_latest,
    take_latest,
    open_robot_link,
)
from pipeline import (UNKNOWN_COLOUR, ImageFolderSource, PickCommand, ShapeDetector,
                      colour_of, detections_from_result, encode_command)

STREAM_WIDTH = 1224
JPEG_QUALITY = 70
PREVIEW_STEP = 2  # preview keeps every 2nd pixel: 2448x2048 -> 1224x1024, almost free
POLL_S = 0.05
# The camera can deliver ~75 fps of 5 MP frames. Converting every one of them to
# BGR kept the Pi's CPU busy and slowed inference to ~2 s, so capture is capped:
# the browser still gets a smooth live picture, and inference gets the CPU.
CAPTURE_FPS = 6.0
WARMUP_SHAPE = (2048, 2448, 3)  # acA2440 frame size; warm the model up on this



def to_jpeg(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buffer.tobytes() if ok else b""


def shrink(frame: np.ndarray) -> np.ndarray:
    """Cheap preview: take every PREVIEW_STEP-th pixel (no filtering, ~free on the Pi)."""
    if frame.shape[1] <= STREAM_WIDTH:
        return frame
    return np.ascontiguousarray(frame[::PREVIEW_STEP, ::PREVIEW_STEP])


LABEL_BGR = {"red": (60, 60, 230), "blue": (230, 120, 40), "green": (60, 190, 60),
             "black": (200, 200, 200), "white": (235, 235, 235)}
NEXT_BGR = (0, 255, 255)
DIM_BGR = (140, 140, 140)


def _text(canvas: np.ndarray, text: str, org: tuple[int, int], colour, size: float) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
    x, y = org
    cv2.rectangle(canvas, (x - 3, y - th - 3), (x + tw + 3, y + base), (20, 20, 20), -1)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)


def draw_parts(canvas: np.ndarray, model: DashboardModel, k: float) -> np.ndarray:
    """Box, centre cross, orientation arrow and 'colour shape' + robot angle per part.

    k maps full-frame pixel coordinates onto canvas (0.5 for the 1224 px preview).
    The arrow follows the first edge of the detected box; the number is the angle
    the robot will be sent (same as the table).
    """
    # Work on a copy so the shared preview stays clean.
    canvas = canvas.copy()
    nxt = model.next_index()
    for index, (detection, row) in enumerate(zip(model.detections(), model.rows())):
        matching = model.matches(index)
        colour = (NEXT_BGR if index == nxt
                  else LABEL_BGR.get(row.colour, (160, 160, 160)) if matching else DIM_BGR)
        corners = np.asarray(detection.corners, np.float64) * k
        cx, cy = detection.center[0] * k, detection.center[1] * k
        centre = (int(round(cx)), int(round(cy)))
        cv2.polylines(canvas, [corners.astype(np.int32).reshape(-1, 1, 2)], True, colour,
                      3 if index == nxt else 2 if matching else 1, cv2.LINE_AA)
        # centre: a cross plus a dot
        arm = 8
        cv2.line(canvas, (centre[0] - arm, centre[1]), (centre[0] + arm, centre[1]),
                 colour, 2, cv2.LINE_AA)
        cv2.line(canvas, (centre[0], centre[1] - arm), (centre[0], centre[1] + arm),
                 colour, 2, cv2.LINE_AA)
        cv2.circle(canvas, centre, 3, (0, 0, 0), -1, cv2.LINE_AA)
        # orientation: arrow from the centre along the box's first edge
        edge = corners[1] - corners[0]
        length = float(np.hypot(*edge))
        if length > 1.0:
            tip = (int(round(cx + edge[0] / length * max(25.0, 0.6 * length))),
                   int(round(cy + edge[1] / length * max(25.0, 0.6 * length))))
            cv2.arrowedLine(canvas, centre, tip, colour, 2, cv2.LINE_AA, tipLength=0.3)
        # text: name on the first line, centre in mm and angle on the second
        name = ("NEXT: " if index == nxt else "") + f"{row.colour} {row.shape}"
        detail = f"x{row.x_mm:.0f} y{row.y_mm:.0f}  {row.angle_deg:+.1f} deg"
        top = int(corners[:, 1].min()) - 24
        if top < 14:
            top = int(corners[:, 1].max()) + 18
        for line_no, text in enumerate((name, detail)):
            (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x = min(max(4, centre[0] - tw // 2), canvas.shape[1] - tw - 4)
            _text(canvas, text, (x, top + line_no * 17), colour, 0.5)
    return canvas


ARUCO_EVERY = 3            # look for the marker on every 3rd preview (~2 per second)
ARUCO_BGR = (255, 255, 0)  # cyan outline
CAL_BGR = (255, 0, 255)    # magenta: where calibration says the centre is
DRIFT_WARN_PX = 6.0        # live centre this far from the calibrated one -> warn


def find_arucos(preview: np.ndarray, detector, full_width: int) -> list[dict]:
    """Every ArUco marker in the preview, corners in full-frame pixels."""
    grey = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(grey)
    if ids is None:
        return []
    k = full_width / preview.shape[1]
    found = []
    for marker_corners, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
        pts = np.asarray(marker_corners, np.float64).reshape(4, 2) * k
        found.append({"id": int(marker_id), "corners": pts, "centre": pts.mean(axis=0)})
    return found


# Turn robot millimetres into camera pixels with the calibration.
def _mm_to_px(plane, x_mm: float, y_mm: float) -> np.ndarray:
    inverse = np.linalg.inv(np.asarray(plane.matrix, np.float64).reshape(3, 3))
    u, v, w = inverse @ np.array([x_mm, y_mm, 1.0])
    return np.array([u / w, v / w])


def aruco_summary(markers: list[dict], plane) -> str:
    board = next((m for m in markers if m["id"] == MARKER_ID), None)
    if board is None:
        seen = ", ".join(str(m["id"]) for m in markers) or "none"
        return f"ArUco {MARKER_ID}: NOT FOUND (seen: {seen})"
    cx, cy = board["centre"]
    # Measure how far the marker moved from its calibrated centre.
    drift = float(np.hypot(cx - plane.centre_px[0], cy - plane.centre_px[1]))
    edge = board["corners"][1] - board["corners"][0]
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
    text = (f"ArUco {MARKER_ID}: centre px ({cx:.0f}, {cy:.0f}), edge angle {angle:+.1f} deg, "
            f"{drift:.1f} px from calibration")
    if drift > DRIFT_WARN_PX:
        text += " - board or camera MOVED: re-run calibrate.py"
    return text


def draw_aruco(canvas: np.ndarray, markers: list[dict], plane, k: float) -> None:
    """Marker border, corner 0, live centre, calibrated centre and robot X/Y axes."""
    cal = np.array(plane.centre_px, np.float64) * k
    cal_pt = (int(round(cal[0])), int(round(cal[1])))
    cv2.circle(canvas, cal_pt, 9, CAL_BGR, 2, cv2.LINE_AA)
    # robot frame axes through the calibrated centre (+X and +Y, 50 mm long)
    for axis, colour, name in (((50.0, 0.0), (60, 60, 255), "+X"), ((0.0, 50.0), (60, 220, 60), "+Y")):
        tip = _mm_to_px(plane, *axis) * k
        tip_pt = (int(round(tip[0])), int(round(tip[1])))
        cv2.arrowedLine(canvas, cal_pt, tip_pt, colour, 2, cv2.LINE_AA, tipLength=0.2)
        _text(canvas, name, (tip_pt[0] + 4, tip_pt[1] + 4), colour, 0.45)
    for marker in markers:
        pts = marker["corners"] * k
        board = marker["id"] == MARKER_ID
        colour = ARUCO_BGR if board else (180, 180, 0)
        cv2.polylines(canvas, [pts.astype(np.int32).reshape(-1, 1, 2)], True, colour,
                      3 if board else 1, cv2.LINE_AA)
        first = (int(round(pts[0][0])), int(round(pts[0][1])))
        cv2.circle(canvas, first, 5, colour, -1, cv2.LINE_AA)  # corner 0
        c = marker["centre"] * k
        centre = (int(round(c[0])), int(round(c[1])))
        arm = 12
        cv2.line(canvas, (centre[0] - arm, centre[1]), (centre[0] + arm, centre[1]), colour, 2,
                 cv2.LINE_AA)
        cv2.line(canvas, (centre[0], centre[1] - arm), (centre[0], centre[1] + arm), colour, 2,
                 cv2.LINE_AA)
        label = f"ArUco {marker['id']}" + (" (0,0) mm" if board else "")
        bottom = int(pts[:, 1].max()) + 18
        left = int(pts[:, 0].min())
        _text(canvas, label, (left, bottom), colour, 0.5)


def render_small(frame: np.ndarray, model: DashboardModel,
                 full_width: int | None = None) -> np.ndarray:
    """Draw the labels on a small frame. full_width = width the detections refer to."""
    height, width = frame.shape[:2]
    if full_width is not None and full_width != width:  # frame is already a preview
        return draw_parts(frame, model, width / full_width)
    scale = min(1.0, STREAM_WIDTH / width)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)),
                           interpolation=cv2.INTER_AREA)
    return draw_parts(frame, model, scale)


class FastModel(DashboardModel):
    """DashboardModel that computes each part's pick command once per frame.

    to_pick_command runs a template fit on the image (slow on the Pi). The old
    model did it for every part on refresh and twice more on every Pick; here it
    is done once, in the inference thread, and Pick reuses the stored command.
    """

    _commands: tuple = ()

    def refresh(self, detections, frame=None):
        self._frame = frame
        self._detections = tuple(detections)
        commands, rows = [], []
        for detection in self._detections:
            # Slow template fit: done once per part here, not at Pick time.
            command = self.mapper.to_pick_command(
                detection, angle_offset_deg=self.angle_offset_deg, image=frame)
            colour = UNKNOWN_COLOUR if frame is None else colour_of(frame, detection)
            commands.append(command)
            rows.append(ShapeRow(command.shape, colour, detection.confidence,
                                 command.x_mm, command.y_mm, command.angle_deg))
        self._commands = tuple(commands)
        self._rows = tuple(rows)
        return self._rows

    # Return the next pick command; consume=True removes that part.
    def choose(self, consume: bool = False):
        index = self.next_index()
        if index is None:
            return None
        command = self._commands[index]
        if consume:
            keep = [i for i in range(len(self._detections)) if i != index]
            self._detections = tuple(self._detections[i] for i in keep)
            self._rows = tuple(self._rows[i] for i in keep)
            self._commands = tuple(self._commands[i] for i in keep)
        return command

    def fresh(self) -> "FastModel":
        return FastModel(self.mapper, self.shapes, self.angle_offset_deg)


class WebWorker(PipelineWorker):
    """PipelineWorker that also keeps the newest raw frame for the live picture.

    The picture no longer waits for inference: every captured frame is shown at
    once with the labels from the latest detection drawn on top.
    """

    def __init__(self, source, detector, model: FastModel,
                 capture_fps: float = CAPTURE_FPS) -> None:
        super().__init__(source, detector, interval_s=1.0 / capture_fps)
        self.model = model
        self.preview: np.ndarray | None = None   # already shrunk for the browser
        self.full_width = 0                      # width of the frames detection sees
        self.preview_id = 0

    def _capture(self) -> None:
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                frame = self.source.read()
                if frame is None:
                    self.finished.set()
                    return
                self.full_width = frame.shape[1]
                # Keep a small copy of the frame for the browser.
                self.preview = shrink(frame)
                self.preview_id += 1
                # Hand the full frame to the inference thread.
                put_latest(self.frames, frame)
                self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - started)))
        except Exception as error:
            self._fail(error)

    def _infer(self) -> None:
        """Detect, then build the table and pick commands here, off the web threads."""
        try:
            while not self._stop.is_set():
                frame = take_latest(self.frames)
                if frame is None:
                    if self.finished.is_set():
                        return
                    self._stop.wait(POLL_S)
                    continue
                # Detection runs here, away from the web request threads.
                detections = self.detector.detect(frame)
                started = time.perf_counter()
                model = self.model.fresh()
                model.refresh(detections, frame)
                print(f"pick positions: {(time.perf_counter() - started) * 1000.0:.1f} ms"
                      f" for {len(detections)} part(s)")
                put_latest(self.detections, (frame, detections, model))
        except Exception as error:
            self._fail(error)


class Controller:
    """Everything DashboardWindow does, without Tk. One lock guards the model."""

    def __init__(self, model: FastModel, worker: WebWorker,
                 sender: CommandSender | None, saver: FrameSaver | None) -> None:
        self.model = model
        self.worker = worker
        self.sender = sender
        self.saver = saver
        self.lock = threading.Lock()
        self.last_frame: np.ndarray | None = None
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.status = "starting camera..."
        self._shown_preview = 0
        self._have_result = False
        self._dirty = False  # labels changed (pick / filter): redraw on next tick
        self.aruco_detector = make_detector()
        self.markers: list[dict] = []
        self.aruco_text = "ArUco: looking..."
        self._aruco_count = 0
        self.command = "no command yet"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._drain, name="drain", daemon=True)

    def start(self) -> None:
        if self.sender is not None:
            self.sender.start()
        if self.saver is not None:
            self.saver.start()
        self.worker.start()
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.worker.stop()
        if self.sender is not None:
            self.sender.stop()
        if self.saver is not None:
            self.saver.stop()
        self.thread.join(timeout=2.0)

    # --- the worker loop (was DashboardWindow.drain) -------------------------
    def _drain(self) -> None:
        while not self.stop.is_set():
            if self.worker.error is not None:
                with self.lock:
                    self.status = f"worker stopped: {self.worker.error}"
                return
            result = self.worker.latest_result()
            if result is not None:
                frame, detections, fresh = result
                with self.lock:  # just swap in the ready-made table, keep the filters
                    fresh.selected = self.model.selected
                    fresh.selected_colour = self.model.selected_colour
                    self.model = fresh
                    self.last_frame = frame
                    self._have_result = True
                    if not self._link_problem():
                        self.status = f"{len(detections)} shape(s) in view"
            preview, preview_id = self.worker.preview, self.worker.preview_id
            if preview is not None and (preview_id != self._shown_preview
                                        or result is not None or self._dirty):
                self._shown_preview = preview_id
                self._dirty = False
                full_width = self.worker.full_width or preview.shape[1]
                self._aruco_count += 1
                if self._aruco_count % ARUCO_EVERY == 1:
                    try:
                        markers = find_arucos(preview, self.aruco_detector, full_width)
                        text = aruco_summary(markers, self.model.mapper)
                    except Exception as error:  # never let the marker overlay stop the view
                        markers, text = [], f"ArUco: {error}"
                    with self.lock:
                        self.markers, self.aruco_text = markers, text
                with self.lock:  # only the cheap label drawing holds the lock
                    labelled = render_small(preview, self.model, self.worker.full_width)
                    draw_aruco(labelled, self.markers, self.model.mapper,
                               labelled.shape[1] / full_width)
                    if not self._have_result and not self._link_problem():
                        self.status = "camera live - waiting for the first detection..."
                jpeg = to_jpeg(labelled)  # JPEG encoding happens outside the lock
                with self.lock:
                    self.jpeg = jpeg
                    self.frame_id += 1
            if self.saver is not None and self.saver.error is not None:
                with self.lock:
                    self.status = f"save failed: {self.saver.error}"
            self.stop.wait(POLL_S)

    def _render(self) -> None:
        """Ask the drain thread to redraw the labels; Pick does not wait for it."""
        self._dirty = True

    def _link_problem(self) -> bool:
        if self.sender is not None and self.sender.error is not None:
            self.status = f"robot link is down ({self.sender.error}) - picks are not sent"
            return True
        return False

    # --- buttons ------------------------------------------------------------
    def set_filters(self, shape: str | None, colour: str | None) -> None:
        with self.lock:
            if shape is not None:
                self.model.select(None if shape == SHAPE_CHOICES[0] else shape)
            if colour is not None:
                self.model.select_colour(None if colour == COLOUR_CHOICES[0] else colour)
            self.status = describe_filters(self.model.selected, self.model.selected_colour)
            self._render()

    def pick(self) -> str:
        with self.lock:
            if self.sender is not None and self.sender.broken:
                self.status = f"robot link is down ({self.sender.error}) - pick not sent"
                return self.status
            # Look at the next part first; remove it only after the checks.
            preview = self.model.choose(consume=False)
            if preview is None:
                self.status = "nothing matches the shape and colour filters"
                return self.status
            # Ask the pipeline whether this shape may be picked.
            blocked = pipeline.pick_blocked(preview.shape)
            if blocked:
                self.status = blocked + " - pick not sent"
                return self.status
            command = self.model.choose(consume=True)
            if command is None:
                self.status = "nothing matches the shape and colour filters"
                return self.status
            self._render()
            self.command = encode_command(command)
            self.status = f"picked {command.shape}" + ("" if self.sender else " (not sent: no --send)")
        self._deliver(command)
        return self.status

    def _deliver(self, command: PickCommand) -> None:
        if self.sender is not None:
            self.sender.offer(command)
        else:
            print(f"would send: {encode_command(command)}")

    def save(self) -> str:
        with self.lock:
            if self.saver is None:
                self.status = "not started with --record: nothing saved"
            elif self.last_frame is None:
                self.status = "no frame yet"
            else:
                self.status = f"saving {self.saver.offer(self.last_frame).name}"
            return self.status

    # --- page data ----------------------------------------------------------
    def state(self) -> dict:
        with self.lock:
            nxt = self.model.next_index()
            rows = []
            for index, row in enumerate(self.model.rows()):
                rows.append({
                    "next": index == nxt,
                    "match": self.model.matches(index),
                    "colour": row.colour,
                    "shape": row.shape,
                    "conf": round(row.confidence, 2),
                    "x": round(row.x_mm, 1),
                    "y": round(row.y_mm, 1),
                    "angle": round(row.angle_deg, 1),
                    "blocked": pipeline.pick_blocked(row.shape) or "",
                })
            return {
                "status": self.status,
                "command": self.command,
                "filter": describe_filters(self.model.selected, self.model.selected_colour),
                "shape": self.model.selected or SHAPE_CHOICES[0],
                "colour": self.model.selected_colour or COLOUR_CHOICES[0],
                "rows": rows,
                "sending": self.sender is not None,
                "recording": self.saver is not None,
                "frame": self.frame_id,
                "aruco": self.aruco_text,
            }

    def latest_jpeg(self) -> tuple[bytes | None, int]:
        with self.lock:
            return self.jpeg, self.frame_id


class SizedDetector(ShapeDetector):
    """ShapeDetector that runs the network at --imgsz (smaller = faster, less exact)."""

    def __init__(self, model_path, imgsz: int = 640) -> None:
        super().__init__(model_path=model_path)
        self.imgsz = imgsz

    def detect(self, image: np.ndarray) -> list:
        started = time.perf_counter()
        predictions = list(self.model.predict(image, conf=self.conf, device=self.device,
                                              imgsz=self.imgsz, verbose=False))
        print(f"inference: {(time.perf_counter() - started) * 1000.0:.1f} ms")
        return detections_from_result(predictions[0]) if predictions else []


CONTROLLER: Controller | None = None
BOOT_ID = str(int(time.time() * 1000))  # changes on every restart: the page re-opens the stream

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>INA700 dashboard</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}
.wrap{display:flex;gap:16px;flex-wrap:wrap}
.view{flex:3;min-width:320px} .side{flex:2;min-width:300px}
img{width:100%;border:1px solid #444;background:#000}
select,button{font-size:15px;padding:6px 10px;margin:2px;border-radius:4px;border:1px solid #555;background:#222;color:#eee}
button.pick{background:#b8860b;color:#000;font-weight:bold;border-color:#b8860b}
table{border-collapse:collapse;width:100%;margin:10px 0;font-family:monospace;font-size:13px}
td,th{border:1px solid #444;padding:3px 6px;text-align:right}
th{background:#222} tr.next{background:#443d00} tr.dim{color:#777}
.bad{color:#f66;text-align:left} #status{color:#9cf;margin:8px 0}
#command{font-family:monospace;font-size:16px;color:#ffd75e}
.warn{color:#f99}
</style></head><body>
<h3>INA700 - shape picking <span id="mode"></span></h3>
<div class="wrap">
 <div class="view"><img id="cam" alt="camera"></div>
 <div class="side">
  <div>Pick by shape: <select id="shape"></select>
       colour: <select id="colour"></select></div>
  <div id="filter"></div>
  <table><thead><tr><th>colour</th><th>shape</th><th>conf</th><th>x mm</th>
   <th>y mm</th><th>angle</th><th>note</th></tr></thead><tbody id="rows"></tbody></table>
  <button class="pick" onclick="act('/pick')">Pick</button>
  <button onclick="act('/save')">Save frame</button>
  <div>last command: <span id="command"></span></div>
  <div id="status">connecting...</div>
  <div id="aruco" style="font-size:13px;color:#7ee"></div>
 </div>
</div>
<script>
const SHAPES=__SHAPES__, COLOURS=__COLOURS__;
function fill(id,list){const s=document.getElementById(id);
  s.innerHTML=list.map(v=>`<option>${v}</option>`).join('');
  s.onchange=()=>act(`/filter?${id}=${encodeURIComponent(s.value)}`);}
fill('shape',SHAPES); fill('colour',COLOURS);
async function act(url){
  try{const r=await fetch(url,{method:'POST'}); const d=await r.json();
      document.getElementById('status').textContent=d.status; poll();}
  catch(e){document.getElementById('status').textContent='lost connection to the Pi';}
}
async function poll(){
  try{
    const d=await (await fetch('/state')).json();
    document.getElementById('status').textContent=d.status;
    document.getElementById('command').textContent=d.command;
    document.getElementById('filter').textContent=d.filter;
    const a=document.getElementById('aruco'); a.textContent=d.aruco||'';
    a.style.color=/MOVED|NOT FOUND/.test(d.aruco||'')?'#f66':'#7ee';
    document.getElementById('mode').innerHTML=d.sending
      ? '<span class="warn">- LIVE: Pick sends to the robot</span>' : '- dry run, nothing is sent';
    for(const id of ['shape','colour']){const s=document.getElementById(id);
      if(document.activeElement!==s) s.value=d[id];}
    document.getElementById('rows').innerHTML=d.rows.map(x=>
      `<tr class="${x.next?'next':(x.match?'':'dim')}"><td>${x.next?'NEXT ':''}${x.colour}</td>
       <td>${x.shape}</td><td>${x.conf}</td><td>${x.x}</td><td>${x.y}</td>
       <td>${x.angle}</td><td class="bad">${x.blocked}</td></tr>`).join('');
  }catch(e){document.getElementById('status').textContent='lost connection to the Pi';}
}
// The picture is one long MJPEG connection. When the Pi restarts or the link drops,
// the browser keeps showing the last frame forever - so reconnect it ourselves.
const cam=document.getElementById('cam');
let boot=null, lastFrame=-1, staleSince=Date.now();
function openStream(){cam.src='/stream?t='+Date.now(); staleSince=Date.now();}
cam.onerror=()=>setTimeout(openStream,1000);
async function watch(){
  try{
    const d=await (await fetch('/state')).json();
    if(d.boot!==boot){boot=d.boot; lastFrame=d.frame; openStream(); return;}
    if(d.frame!==lastFrame){lastFrame=d.frame; staleSince=Date.now();}
    else if(Date.now()-staleSince>5000){openStream();}  // server alive, picture stuck
  }catch(e){}
}
setInterval(poll,1000); poll(); setInterval(watch,2000); watch();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            page = (PAGE.replace("__SHAPES__", json.dumps(list(SHAPE_CHOICES)))
                        .replace("__COLOURS__", json.dumps(list(COLOUR_CHOICES))))
            self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
        elif path == "/state":
            self._json({**CONTROLLER.state(), "boot": BOOT_ID})
        elif path == "/stream":
            self._stream()
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/pick":
                status = CONTROLLER.pick()
            elif parsed.path == "/save":
                status = CONTROLLER.save()
            elif parsed.path == "/filter":
                CONTROLLER.set_filters(query.get("shape", [None])[0],
                                       query.get("colour", [None])[0])
                status = CONTROLLER.state()["status"]
            else:
                self._send(404, "text/plain", b"not found")
                return
        except ValueError as error:
            status = str(error)
        self._json({"status": status})

    def _json(self, data: dict) -> None:
        self._send(200, "application/json", json.dumps(data).encode("utf-8"))

    def _send(self, code: int, kind: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                # Send only newer pictures down the open stream.
                jpeg, frame_id = CONTROLLER.latest_jpeg()
                if jpeg and frame_id != last:
                    last = frame_id
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                     + jpeg + b"\r\n")
                    self.wfile.flush()
                time.sleep(POLL_S)
        except (BrokenPipeError, ConnectionResetError):
            return


def addresses() -> list[str]:
    try:
        return subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    except OSError:
        return []


def main(argv: list[str] | None = None) -> int:
    global CONTROLLER
    ncnn = pipeline.MODEL_PATH.parent / "best_ncnn_model"
    default_model = ncnn if ncnn.is_dir() else pipeline.MODEL_PATH

    parser = argparse.ArgumentParser(description="shape picking dashboard in a web browser")
    parser.add_argument("--send", action="store_true",
                        help="send each pick to the robot instead of printing it")
    parser.add_argument("--host", default=ROBOT_HOST,
                        help="address to listen on for the robot (the Pi's robot-side IP)")
    parser.add_argument("--port", type=int, default=ROBOT_PORT,
                        help="TCP port the robot connects to")
    parser.add_argument("--camera", action="store_true",
                        help="read the Basler live instead of replaying a folder")
    parser.add_argument("--folder", default=str(RAW_DIR),
                        help="image folder to replay when --camera is not given")
    parser.add_argument("--record", action="store_true",
                        help="let Save frame write to data/captured_frames")
    parser.add_argument("--angle-offset", type=float, default=pipeline.ANGLE_OFFSET_DEG,
                        help="degrees added before the symmetry fold (decision #17)")
    parser.add_argument("--model", default=str(default_model), help="OBB weights")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="network input size; 480 or 512 is faster (multiple of 32)")
    parser.add_argument("--web-port", type=int, default=8080, help="browser port (default 8080)")
    arguments = parser.parse_args(argv)

    print(pipeline.apply_angle_teach())
    if arguments.send:
        blocked = pipeline.send_blocked()
        if blocked:
            raise SystemExit(blocked)
    # Saved frames would land inside the folder being replayed.
    if arguments.record and not arguments.camera:
        raise SystemExit("--record needs --camera: a folder of frames is already on disk")
    try:
        plane = load_calibration()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"no usable calibration: {error}\n"
                         "run: python scripts/calibrate.py --live") from error

    # the camera before the robot: a busy camera should fail before RAPID connects
    source = BaslerSource() if arguments.camera else ImageFolderSource(Path(arguments.folder))
    detector = SizedDetector(arguments.model, imgsz=arguments.imgsz)
    # The first NCNN inference takes ~20 s (load + warm-up). Do it now, before the
    # camera opens and before RAPID connects, so the page is live straight away.
    print("warming up the model (about 20 s, once)...")
    detector.detect(np.zeros(WARMUP_SHAPE, np.uint8))
    print("model ready")

    link = None
    sender = None
    if arguments.send:
        try:
            link = open_robot_link(arguments.host, arguments.port)
        except pipeline.TransportError as error:
            source.close()
            raise SystemExit(f"robot link failed: {error}") from error
        sender = CommandSender(link.send_pick)

    model = FastModel(plane, angle_offset_deg=arguments.angle_offset)
    CONTROLLER = Controller(model, WebWorker(source, detector, model), sender,
                            FrameSaver() if arguments.record else None)
    CONTROLLER.start()

    server = ThreadingHTTPServer(("0.0.0.0", arguments.web_port), Handler)
    server.daemon_threads = True
    print("open the dashboard in a browser on your PC:")
    for address in addresses() or ["<pi-address>"]:
        print(f"    http://{address}:{arguments.web_port}")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        CONTROLLER.close()
        source.close()
        if link is not None:
            link.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
