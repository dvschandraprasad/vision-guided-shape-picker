"""Shared base for the three grade-level dashboards (INA700).

scripts/dashboard_web.py is reused as it is: the camera, AI inference,
coordinate translation, robot link and web GUI threads are the same for every
grade. A grade only decides
  * which filters the operator may use (shape and/or colour),
  * what one picking cycle is and what starts the next one,
  * how many attempts one shape gets per cycle.

Each grade is one GradeController subclass (grade1_web.py, grade2_web.py,
grade3_web.py). launch() plugs that class, a detector that only reports the
three assigned shapes, the extended web page and the /next_cycle route into
dashboard_web.main(). No file in scripts/ is changed.

The RAPID program (rapid/MainModule_angle.mod) is the same for all grades:
one Pick on the page + one yellow-button press on the robot = one part.
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

# grade_levels/ sits next to scripts/: make the existing modules importable.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import calibrate  # noqa: E402
import dashboard_web  # noqa: E402
import pipeline  # noqa: E402
from pipeline import ALLOWED_CLASS_IDS, CLASSES, PickCommand, encode_command  # noqa: E402

ANY = "any"
ASSIGNED_SHAPES: tuple[str, ...] = tuple(CLASSES[index] for index in ALLOWED_CLASS_IDS)
INFERENCE_LIMIT_MS = 3000.0  # the assignment: detection results within 3 s

# Shape check (see outline_solidity): a circle, hexagon or triangle has a solid
# outline (area / convex-hull area close to 1); a star has deep notches between
# its points (about 0.5-0.65). A detection of a solid shape whose outline is
# below SOLIDITY_MIN is a star the model mistook for that shape, and is dropped.
SOLID_SHAPES = ("circle", "hexagon", "triangle")
SOLIDITY_MIN = 0.80
# Square check (see square_measures): a square fills its tightest box (about 0.96-1.0)
# and its outline has 4 corners; a hexagon fills about 0.78 (6 corners), a circle 0.79,
# a triangle 0.55 (3 corners). A circle/hexagon/triangle detection with 4 corners that
# fills at least RECTANGULARITY_MIN of its box is a square the model mistook, and is dropped.
RECTANGULARITY_MIN = 0.90
# Colour (see part_colour): judged on a small patch at the part's centre, compared
# with the sheet around the part. Measured on the lab camera: coloured parts 100 %
# saturated pixels; black parts 0.27-0.36 of the sheet's brightness; white parts
# 1.03-1.04 of it with no saturated pixels.
COLOURED_MIN = 0.60     # share of saturated pixels in the centre patch -> colour by hue
BLACK_MAX = 0.60        # centre brightness / sheet brightness below this -> black
WHITE_MIN = 0.85        # ... at least this, with few saturated pixels -> white
WHITE_MAX_COLOURED = 0.20
SHAPE_CHECK = True          # --no-shape-check turns it off
MIN_CONFIDENCE = 0.25       # --conf: minimum model confidence (0.25 = model default)
# Size check: the parts are 45 mm or more across, the round pins that hold the sheet
# about 10-30 mm, and the model may call a pin a circle. Smaller boxes are dropped.
MIN_PART_MM = 38.0          # --min-size-mm (0 turns the size check off)
# White parts are not used in this course (instructor's instruction), so white
# detections are dropped and 'white' is not in the colour list. --allow-white keeps them.
IGNORE_WHITE = True

# Captured once at import, so install() can run more than once (tests).
ORIGINAL_PAGE = dashboard_web.PAGE
ORIGINAL_COLOUR_CHOICES = dashboard_web.COLOUR_CHOICES
PATCHED_NAMES = ("SHAPE_CHOICES", "COLOUR_CHOICES", "SizedDetector", "Controller", "Handler",
                 "PAGE", "colour_of")


# --- detection: only the three assigned shapes --------------------------------
def keep_assigned(detections: list) -> list:
    """Drop every detection that is not one of the three assigned shapes."""
    return [detection for detection in detections if detection.class_id in ALLOWED_CLASS_IDS]


def part_outline(image: np.ndarray, detection):
    """The part's outer contour in crop pixels, or None if no clear outline is found.

    The part is separated from the sheet around its box (same method as
    pipeline.refine). None happens for example for a white part on the white
    sheet; then the checks give no verdict and the detection is kept.
    """
    corners = np.asarray(detection.corners, np.float64)
    left, top = corners.min(axis=0)
    right, bottom = corners.max(axis=0)
    pad = max(4, int(0.15 * max(right - left, bottom - top)))
    height, width = image.shape[:2]
    x0, x1 = max(0, int(left) - pad), min(width, int(right) + pad + 1)
    y0, y1 = max(0, int(top) - pad), min(height, int(bottom) + pad + 1)
    crop = image[y0:y1, x0:x1]
    if crop.ndim != 3 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    box = (int(left) - x0, int(top) - y0, int(right) - x0, int(bottom) - y0)
    mask = pipeline._clip_to_box(pipeline.foreground_mask(crop), box)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove specks
    centre = (detection.center[0] - x0, detection.center[1] - y0)
    contour = pipeline._centred_contour(mask, centre)
    if contour is None:
        return None
    area = float(cv2.contourArea(contour))
    box_area = max(1.0, float((right - left) * (bottom - top)))
    if area < 0.10 * box_area:      # too little of the part found: no verdict
        return None
    return contour


def outline_solidity(image: np.ndarray, detection) -> float | None:
    """Outline area / convex-hull area (0..1): about 0.6 for a star, 0.95+ for solid shapes."""
    contour = part_outline(image, detection)
    if contour is None:
        return None
    hull = float(cv2.contourArea(cv2.convexHull(contour)))
    return float(cv2.contourArea(contour)) / hull if hull > 0.0 else None


def square_measures(image: np.ndarray, detection) -> tuple[float, int] | None:
    """(rectangularity, corners) of the outline, or None if no outline is found.

    rectangularity = outline area / area of its tightest rotated rectangle;
    corners = vertices of the simplified convex outline (4 for a square).
    """
    contour = part_outline(image, detection)
    if contour is None:
        return None
    width, height = cv2.minAreaRect(contour)[1]
    if width * height <= 0.0:
        return None
    hull = cv2.convexHull(contour)
    corners = len(cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True))
    return float(cv2.contourArea(contour)) / float(width * height), corners


def looks_like_square(image: np.ndarray, detection) -> float | None:
    """The rectangularity if a circle/hexagon/triangle detection has a square's outline, else None."""
    if detection.name not in SOLID_SHAPES:
        return None
    measures = square_measures(image, detection)
    if measures is not None and measures[1] == 4 and measures[0] >= RECTANGULARITY_MIN:
        return measures[0]
    return None


def wrong_shape(image: np.ndarray, detection) -> str | None:
    """Why this detection is really an unassigned shape (star or square), or None."""
    solidity = looks_like_star(image, detection)
    if solidity is not None:
        return f"outline looks like a star (solidity {solidity:.2f} < {SOLIDITY_MIN})"
    rectangularity = looks_like_square(image, detection)
    if rectangularity is not None:
        return (f"outline looks like a square (4 corners, fills {rectangularity:.2f} "
                f">= {RECTANGULARITY_MIN} of its box)")
    return None


def part_colour(image: np.ndarray, detection) -> str:
    """Colour of a part: red, blue, green, black, white or 'unknown'.

    pipeline.colour_of looks at the whole box, which also holds a lot of sheet;
    under the lab light that makes white parts look blue/green and a black
    triangle look green. Here a small patch at the part's centre is used:
      * mostly saturated pixels  -> the hue decides (pipeline.colour_of on the patch),
      * much darker than the sheet around the part -> black,
      * as bright as the sheet, hardly any colour  -> white,
      * otherwise the old whole-box answer is kept.
    """
    corners = np.asarray(detection.corners, np.float64)
    left, top = corners.min(axis=0)
    right, bottom = corners.max(axis=0)
    size = max(right - left, bottom - top)
    height, width = image.shape[:2]
    cx, cy = int(detection.center[0]), int(detection.center[1])
    half = max(3, int(0.10 * size))                          # centre patch: 20 % of the part
    patch = image[max(0, cy - half):cy + half + 1, max(0, cx - half):cx + half + 1]
    gap = max(6, int(0.12 * size))                           # sheet ring just outside the box
    x0, y0 = max(0, int(left) - 2 * gap), max(0, int(top) - 2 * gap)
    x1, y1 = min(width, int(right) + 2 * gap), min(height, int(bottom) + 2 * gap)
    around = image[y0:y1, x0:x1]
    if patch.size == 0 or around.size == 0 or image.ndim != 3:
        return pipeline.colour_of(image, detection)
    keep = np.ones(around.shape[:2], bool)
    keep[max(0, int(top) - gap - y0):max(0, int(bottom) + gap - y0),
         max(0, int(left) - gap - x0):max(0, int(right) + gap - x0)] = False
    if int(keep.sum()) < 50:
        return pipeline.colour_of(image, detection)
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sheet_v = float(np.median(cv2.cvtColor(around, cv2.COLOR_BGR2HSV)[keep][:, 2]))
    coloured = float(((hsv_patch[..., 1] >= pipeline._MIN_SATURATION)
                      & (hsv_patch[..., 2] >= pipeline._MIN_VALUE)).mean())
    ratio = float(np.median(hsv_patch[..., 2])) / max(1.0, sheet_v)
    if coloured >= COLOURED_MIN:
        hue = pipeline.colour_of(patch)
        if hue in ("red", "green", "blue"):
            return hue
    if ratio < BLACK_MAX:
        return "black"
    if ratio >= WHITE_MIN and coloured < WHITE_MAX_COLOURED:
        return "white"
    return pipeline.colour_of(image, detection)


def looks_like_star(image: np.ndarray, detection) -> float | None:
    """The solidity if a circle/hexagon/triangle detection has a star's outline, else None."""
    if detection.name not in SOLID_SHAPES:
        return None
    solidity = outline_solidity(image, detection)
    if solidity is not None and solidity < SOLIDITY_MIN:
        return solidity
    return None


class AssignedShapeDetector(dashboard_web.SizedDetector):
    """SizedDetector that never reports the two shapes that were not assigned.

    The network outputs all five classes, so the two unassigned shapes are
    removed here - before the table, the overlay or a Pick can see them.
    It also remembers the last inference time so the page can show it.
    """

    last_inference_ms: float | None = None
    _calls = 0

    def __init__(self, model_path, imgsz: int = 640) -> None:
        """Load the model as SizedDetector does, with the --conf minimum confidence.

        The calibration is loaded too, to measure part sizes in mm (no size check without it).
        """
        super().__init__(model_path, imgsz=imgsz)
        self.conf = MIN_CONFIDENCE
        try:
            self.plane = calibrate.load_calibration()
        except (FileNotFoundError, ValueError):
            self.plane = None

    plane = None
    _last_notes: tuple = ()

    def size_mm(self, detection) -> float | None:
        """Longest side of the detection box in mm, or None without a calibration."""
        if self.plane is None:
            return None
        corners = np.asarray(self.plane.corners_mm(detection.corners), np.float64)
        sides = np.hypot(*(np.roll(corners, -1, axis=0) - corners).T)
        return float(sides.max())

    def why_ignored(self, image, detection) -> str | None:
        """Why one assigned-shape detection must not be used, or None to keep it."""
        size = self.size_mm(detection)
        if MIN_PART_MM > 0 and size is not None and size < MIN_PART_MM:
            return f"only {size:.0f} mm across (< {MIN_PART_MM:.0f} mm): a pin, not a part"
        if IGNORE_WHITE and part_colour(image, detection) == "white":
            return "white part (white parts are not used)"
        if SHAPE_CHECK:
            return wrong_shape(image, detection)
        return None

    def detect(self, image):
        """Detect, drop the unassigned shapes, record and check the inference time."""
        started = time.perf_counter()
        detections = super().detect(image)  # prints "inference: X ms"
        kept = keep_assigned(detections)
        notes = []
        unassigned = sorted({d.name for d in detections} - {d.name for d in kept})
        if unassigned:
            notes.append(f"ignored (not assigned): {', '.join(unassigned)}")
        if image is not None and image.ndim == 3:
            checked = []
            for detection in kept:
                reason = self.why_ignored(image, detection)
                if reason is None:
                    checked.append(detection)
                else:
                    notes.append(f"ignored {detection.name} at px ({detection.center[0]:.0f}, "
                                 f"{detection.center[1]:.0f}): {reason}")
            kept = checked
        # Print the notes only when they change (numbers masked: positions jitter per frame).
        key = tuple(re.sub(r"[-+]?\d+(\.\d+)?", "#", note) for note in notes)
        if self._calls > 0:                       # not for the start-up warm-up call
            if key != self._last_notes:
                for note in notes:
                    print(note)
            self._last_notes = key
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._calls += 1
        if self._calls == 1:
            return kept  # the first call is the start-up warm-up, not a real frame
        self.last_inference_ms = elapsed_ms
        if elapsed_ms > INFERENCE_LIMIT_MS:
            print(f"WARNING: inference took {elapsed_ms:.0f} ms - over the 3 s limit")
        return kept


# --- grade rules ---------------------------------------------------------------
class GradeController(dashboard_web.Controller):
    """dashboard_web.Controller plus picking cycles and per-grade pick rules.

    Subclasses only set the class attributes below (and may override the
    _start_cycle hook). Everything else - camera, model, robot link - is the
    unchanged dashboard_web code.
    """

    LEVEL = "?"                 # shown on the page and in the console
    RULES = ""                  # one line telling the operator what this grade does
    SELECT_SHAPE = False        # operator may choose the shape
    SELECT_COLOUR = False       # operator may choose the colour
    SHAPE_REQUIRED = False      # Pick is refused until a shape is chosen
    MULTI_CYCLE = False         # a 'Start next cycle' button exists
    BLOCK_WHEN_COMPLETE = False # once all shapes are sent, wait for the next cycle
    MAX_ATTEMPTS: int | None = None  # attempts per shape per cycle (None = no limit)

    def __init__(self, *args, **kwargs) -> None:
        """Start at cycle 1 with no attempts; arguments as dashboard_web.Controller."""
        super().__init__(*args, **kwargs)
        self.cycle = 1
        self.attempts: dict[str, int] = {}   # shape -> picks sent in this cycle
        self._pick_lock = threading.Lock()   # one Pick / next-cycle at a time
        self._sent: PickCommand | None = None

    # --- rules (called with self.lock held) ---------------------------------
    @property
    def cycle_complete(self) -> bool:
        """True when every assigned shape has been sent in this cycle."""
        return set(ASSIGNED_SHAPES) <= set(self.attempts)

    def shapes_sent(self) -> int:
        """How many different assigned shapes were sent in this cycle."""
        return sum(1 for shape in ASSIGNED_SHAPES if shape in self.attempts)

    def refuse_pick(self) -> str | None:
        """Why the grade rules forbid a Pick right now, or None."""
        if self.BLOCK_WHEN_COMPLETE and self.cycle_complete:
            return f"cycle {self.cycle} is complete - press 'Start next cycle'"
        shape = self.model.selected
        if self.SHAPE_REQUIRED and shape is None:
            return "choose which shape to pick in 'Pick by shape' first"
        if self.MAX_ATTEMPTS is not None and shape is not None:
            if self.attempts.get(shape, 0) >= self.MAX_ATTEMPTS:
                return (f"{shape}: all {self.MAX_ATTEMPTS} attempts used in cycle "
                        f"{self.cycle} - choose another shape or start the next cycle")
        return None

    def _start_cycle(self) -> None:
        """Hook: runs (under self.lock) when a new cycle starts."""

    # --- buttons ------------------------------------------------------------
    def set_filters(self, shape: str | None, colour: str | None) -> None:
        """Apply only the filters this grade allows; reject unassigned shapes."""
        if not self.SELECT_SHAPE:
            shape = None
        if not self.SELECT_COLOUR:
            colour = None
        if shape not in (None, ANY) and shape not in ASSIGNED_SHAPES:
            raise ValueError(f"{shape} is not an assigned shape ({', '.join(ASSIGNED_SHAPES)})")
        if IGNORE_WHITE and colour == "white":
            raise ValueError("white parts are not used - choose another colour")
        super().set_filters(shape, colour)

    def pick(self) -> str:
        """Check the grade rules, then do the normal Pick and count the attempt."""
        with self._pick_lock:
            with self.lock:
                refusal = self.refuse_pick()
                if refusal:
                    self.status = refusal + " - pick not sent"
                    return self.status
            self._sent = None
            status = super().pick()           # unchanged dashboard_web pick
            command, self._sent = self._sent, None
            if command is None:               # nothing matched / link down / blocked
                return status
            with self.lock:
                used = self.attempts.get(command.shape, 0) + 1
                self.attempts[command.shape] = used
                note = f"{self.shapes_sent()} of {len(ASSIGNED_SHAPES)} shapes sent"
                if self.MAX_ATTEMPTS is not None:
                    note += f", {command.shape} attempt {used} of {self.MAX_ATTEMPTS}"
                self.status = f"{status} - cycle {self.cycle}: {note}"
                text = self.status
            print(f"[grade {self.LEVEL}] {text}  ->  {encode_command(command)}")
            return text

    def _deliver(self, command: PickCommand) -> None:
        """Remember the command that was really sent, then send it."""
        self._sent = command                  # remember what was actually sent
        super()._deliver(command)

    def next_cycle(self) -> str:
        """'Start next cycle' button: new cycle number, attempts reset."""
        with self._pick_lock:
            with self.lock:
                if not self.MULTI_CYCLE:
                    self.status = (f"Grade {self.LEVEL} runs a single cycle - "
                                   "restart the program for another run")
                    return self.status
                self.cycle += 1
                self.attempts.clear()
                self._start_cycle()
                self.status = f"cycle {self.cycle} started"
                self._render()
                text = self.status
            print(f"[grade {self.LEVEL}] ----- cycle {self.cycle} started -----")
            return text

    # --- page data ----------------------------------------------------------
    def grade_state(self) -> dict:
        """The grade panel on the page (called with self.lock held)."""
        needed, done = len(ASSIGNED_SHAPES), self.shapes_sent()
        if not self.MULTI_CYCLE:
            headline = (f"All {needed} shapes sent - Grade {self.LEVEL} run complete"
                        if self.cycle_complete else f"{done} of {needed} shapes sent")
        elif self.cycle_complete:
            headline = (f"Cycle {self.cycle} complete: all {needed} shapes sent - "
                        "press 'Start next cycle'")
            if not self.BLOCK_WHEN_COMPLETE:
                headline += " (retries still allowed)"
        else:
            headline = f"Cycle {self.cycle}: {done} of {needed} shapes sent"
        parts = []
        for shape in ASSIGNED_SHAPES:
            used = self.attempts.get(shape, 0)
            if self.MAX_ATTEMPTS is not None:
                parts.append(f"{shape} {used}/{self.MAX_ATTEMPTS}")
            else:
                parts.append(f"{shape} {'sent' if used else '-'}" + (f" x{used}" if used > 1 else ""))
        inference = getattr(self.worker.detector, "last_inference_ms", None)
        return {
            "level": self.LEVEL,
            "rules": self.RULES,
            "cycle": self.cycle,
            "complete": self.cycle_complete,
            "headline": headline,
            "parts": "   ".join(parts),
            "attempts": dict(self.attempts),
            "inference_ms": None if inference is None else round(inference),
            "limit_ms": round(INFERENCE_LIMIT_MS),
        }

    def state(self) -> dict:
        """dashboard_web's page data plus the grade panel."""
        data = super().state()
        with self.lock:
            data["grade"] = self.grade_state()
        return data


class GradeHandler(dashboard_web.Handler):
    """dashboard_web.Handler plus POST /next_cycle."""

    def do_POST(self) -> None:
        """Handle /next_cycle here; every other button as in dashboard_web."""
        if urlparse(self.path).path == "/next_cycle":
            self._json({"status": dashboard_web.CONTROLLER.next_cycle()})
            return
        super().do_POST()


# --- web page ------------------------------------------------------------------
GRADE_JS = """
function gradePoll(d){const g=d.grade; if(!g) return;
  document.getElementById('gradeline').textContent=g.headline;
  document.getElementById('gradeparts').textContent=g.parts;
  const i=document.getElementById('infer');
  if(g.inference_ms===null){i.textContent='inference: waiting for the first frame...';i.style.color='#aaa';}
  else{i.textContent=`inference: ${g.inference_ms} ms (limit ${g.limit_ms} ms)`;
       i.style.color=g.inference_ms>g.limit_ms?'#f66':'#8f8';}
  const b=document.getElementById('nextcycle'); if(b) b.className=g.complete?'pick':'';
}
"""


def build_page(controller_cls: type[GradeController]) -> str:
    """dashboard_web's page with the grade title, panel and allowed filters."""
    level = controller_cls.LEVEL
    hide_shape = "" if controller_cls.SELECT_SHAPE else ' style="display:none"'
    hide_colour = "" if controller_cls.SELECT_COLOUR else ' style="display:none"'
    button = ('<button id="nextcycle" onclick="act(\'/next_cycle\')">Start next cycle</button>'
              if controller_cls.MULTI_CYCLE else "")
    panel = (
        '<div style="margin:10px 0;padding:8px;border:1px solid #666;border-radius:4px">'
        f'<div style="color:#aaa;font-size:13px">Grade level {level}: {controller_cls.RULES}</div>'
        '<div id="gradeline" style="font-weight:bold;margin:4px 0"></div>'
        '<div id="gradeparts" style="font-family:monospace"></div>'
        f'<div id="infer" style="font-family:monospace;font-size:13px"></div>{button}</div>\n'
        '  <div id="status">connecting...</div>'
    )
    replacements = [
        ("<title>INA700 dashboard</title>", f"<title>INA700 Grade {level}</title>"),
        ("<h3>INA700 - shape picking", f"<h3>INA700 - Grade level {level} - shape picking"),
        ('Pick by shape: <select id="shape"></select>',
         f'<span{hide_shape}>Pick by shape: <select id="shape"></select></span>'),
        ('colour: <select id="colour"></select>',
         f'<span{hide_colour}>colour: <select id="colour"></select></span>'),
        ('<div id="status">connecting...</div>', panel),
        ("document.getElementById('command').textContent=d.command;",
         "document.getElementById('command').textContent=d.command; gradePoll(d);"),
        ("</script></body>", GRADE_JS + "</script></body>"),
    ]
    page = ORIGINAL_PAGE
    for old, new in replacements:
        if old not in page:  # dashboard_web's page changed: fail loudly, not silently
            raise RuntimeError(f"build_page: dashboard_web.PAGE no longer contains {old!r}")
        page = page.replace(old, new, 1)
    return page


# --- start-up ------------------------------------------------------------------
def install(controller_cls: type[GradeController]) -> dict:
    """Point dashboard_web at the grade classes. Returns the old values (restore())."""
    previous = {name: getattr(dashboard_web, name) for name in PATCHED_NAMES}
    dashboard_web.SHAPE_CHOICES = (ANY, *ASSIGNED_SHAPES)
    dashboard_web.SizedDetector = AssignedShapeDetector
    dashboard_web.Controller = controller_cls
    dashboard_web.Handler = GradeHandler
    dashboard_web.PAGE = build_page(controller_cls)
    dashboard_web.colour_of = part_colour   # used for the table, overlay and colour filter
    dashboard_web.COLOUR_CHOICES = tuple(c for c in ORIGINAL_COLOUR_CHOICES
                                         if not (IGNORE_WHITE and c == "white"))
    return previous


def restore(previous: dict) -> None:
    """Undo install() (used by the tests)."""
    for name, value in previous.items():
        setattr(dashboard_web, name, value)


def launch(controller_cls: type[GradeController], argv: list[str] | None = None) -> int:
    """Run dashboard_web.main() with this grade's rules (same command-line options)."""
    global MIN_CONFIDENCE, SHAPE_CHECK, MIN_PART_MM, IGNORE_WHITE
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--conf", type=float, default=MIN_CONFIDENCE,
                       help="minimum model confidence for a detection (default 0.25)")
    extra.add_argument("--no-shape-check", action="store_true",
                       help="turn off the star / square check on circle/hexagon/triangle detections")
    extra.add_argument("--min-size-mm", type=float, default=MIN_PART_MM,
                       help="drop detections smaller than this (pins); 0 = off (default 38)")
    extra.add_argument("--allow-white", action="store_true",
                       help="also use white parts (by default they are ignored)")
    options, rest = extra.parse_known_args(argv)
    MIN_CONFIDENCE, SHAPE_CHECK = options.conf, not options.no_shape_check
    MIN_PART_MM, IGNORE_WHITE = options.min_size_mm, not options.allow_white
    install(controller_cls)
    print(f"INA700 Grade level {controller_cls.LEVEL}: {controller_cls.RULES}")
    print(f"minimum confidence {MIN_CONFIDENCE:.2f}, shape check "
          f"{'on' if SHAPE_CHECK else 'off'} (rejects stars and squares the model mislabels)")
    print(f"minimum part size {MIN_PART_MM:.0f} mm (pins are dropped), white parts "
          f"{'ignored' if IGNORE_WHITE else 'used'}")
    print(f"assigned shapes: {', '.join(ASSIGNED_SHAPES)} - the other two are ignored")
    return dashboard_web.main(rest)
