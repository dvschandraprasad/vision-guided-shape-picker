from __future__ import annotations

import pipeline

import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = pipeline.REPO_ROOT
ROOT = REPO_ROOT / "data" / "raw"
LABELS_DIR = REPO_ROOT / "data" / "labels_obb"
PREVIEW_DIR = REPO_ROOT / "data" / "previews"
CLASSES = ["circle", "square", "hexagon", "star", "triangle"]

DISPLAY_MAX_WIDTH = 1600
WINDOW_START_SHARE = 0.95
START_AT = "star_0051"
COORD_TOL = 1e-4
OUTLINE_POINTS = 4
HIGHLIGHT_MIN_SATURATION = 60

STATUSES = ("draft", "approved", "flagged")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
WINDOW = "annotate"
ROOT_RESOLVED = ROOT.resolve()

COLOUR_SELECTED = (0, 0, 255)
COLOUR_OBJECT = (0, 255, 0)
COLOUR_POINT = (255, 0, 255)
COLOUR_TEXT = (0, 255, 255)
COLOUR_HIGHLIGHT = (255, 160, 0)

CLASS_KEYS = {ord("1"): 0, ord("2"): 1, ord("3"): 2, ord("4"): 3, ord("5"): 4}

KEYS_HELP = (
    "1-5 class   M outline   S save+next   A empty   F flag   SPACE next",
    "N/P select  C reclass   D delete   B undo point   X discard   Q quit",
)

@dataclass(frozen=True)
class ImagePaths:

    image: Path
    label: Path
    sidecar: Path
    preview: Path

def paths_for(image_path: Path) -> ImagePaths:
    rel = image_path.resolve().relative_to(ROOT_RESOLVED)
    folder = LABELS_DIR / rel.parent
    return ImagePaths(
        image=image_path,
        label=folder / f"{rel.stem}.txt",
        sidecar=folder / f"{rel.stem}.json",
        preview=PREVIEW_DIR / rel.parent / f"{rel.stem}_preview.jpg",
    )

def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)

# Safe save: write temp file, then rename.
def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def as_corners(value: object) -> np.ndarray | None:
    try:
        corners = np.array(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return None
    return corners

def clean_sidecar(data: object) -> dict | None:
    if not isinstance(data, dict) or data.get("status") not in STATUSES:
        return None
    raw_objects = data.get("objects")
    if not isinstance(raw_objects, list):
        return None

    objects = []
    for obj in raw_objects:
        if not isinstance(obj, dict):
            return None
        class_id = obj.get("class_id")
        if isinstance(class_id, bool) or not isinstance(class_id, (int, np.integer)):
            return None
        if not 0 <= class_id < len(CLASSES):
            return None
        corners = as_corners(obj.get("corners"))
        if corners is None:
            return None
        objects.append({"class_id": int(class_id), "corners": corners.tolist()})

    pending = data.get("pending_points", [])
    usable_pending = isinstance(pending, list) and all(
        isinstance(point, (list, tuple)) and len(point) == 2 for point in pending
    )
    return {
        "status": data["status"],
        "objects": objects,
        "pending_points": pending if usable_pending else [],
    }

def load_state(paths: ImagePaths) -> dict | None:
    if not paths.sidecar.is_file():
        return None
    try:
        data = json.loads(paths.sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"warning: unreadable sidecar ignored: {paths.sidecar}")
        return None
    state = clean_sidecar(data)
    if state is None:
        print(f"warning: malformed sidecar ignored: {paths.sidecar}")
    return state

# Put the four corners in one fixed order.
def order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    edge_a, edge_b = points[1] - points[0], points[2] - points[1]
    if edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0] < 0:
        points = points[::-1]
    start = np.lexsort((points[:, 0], points[:, 1]))[0]
    return np.roll(points, -start, axis=0)

def polygon_to_corners(points: np.ndarray) -> np.ndarray:
    return order_corners(cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32))))

def corners_to_yolo_line(corners: object, class_id: int, size: tuple[int, int]) -> str:
    points = as_corners(corners)
    if points is None:
        return ""
    if isinstance(class_id, bool) or not isinstance(class_id, (int, np.integer)):
        return ""
    if not 0 <= class_id < len(CLASSES):
        return ""

    width, height = size
    normalized = points / (width, height)

    offsets = normalized - 0.5
    # Reject boxes with points outside the image.
    outside = np.abs(offsets) > 0.5 + COORD_TOL
    if np.any(outside):
        return ""
    normalized = np.clip(normalized, 0.0, 1.0)

    if _doubled_area(normalized) == 0.0:
        return ""

    fields = [str(int(class_id))]
    for value in normalized.ravel():
        fields.append(f"{value:.6f}")
    return " ".join(fields)

def _doubled_area(corners: np.ndarray) -> float:
    xs, ys = corners[:, 0], corners[:, 1]
    next_xs, next_ys = np.roll(xs, -1), np.roll(ys, -1)
    return abs(np.dot(xs, next_ys) - np.dot(ys, next_xs))

def render_frame(
    canvas: np.ndarray,
    objects: list[dict],
    selected: int = -1,
    scale: float = 1.0,
    thickness: int = 3,
    font: float = 1.2,
    points: list[tuple[int, int]] | None = None,
    message: str = "",
    highlight: list[np.ndarray] | None = None,
    help_lines: tuple[str, ...] = (),
) -> np.ndarray:
    for extra in highlight or ():
        extra_box = (np.asarray(extra, dtype=np.float32) * scale).astype(np.int32)
        hint_thickness = max(2, thickness - 1)
        cv2.polylines(canvas, [extra_box], True, COLOUR_HIGHLIGHT, hint_thickness)

    for index, obj in enumerate(objects):
        box = np.array(obj["corners"], dtype=np.float32) * scale
        colour = COLOUR_SELECTED if index == selected else COLOUR_OBJECT
        cv2.polylines(canvas, [box.astype(np.int32)], True, colour, thickness)
        x, y = box.min(axis=0).astype(int)
        cv2.putText(canvas, f"{index}:{CLASSES[obj['class_id']]}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font, colour, thickness)

    for px, py in points or ():
        cv2.circle(canvas, (int(px * scale), int(py * scale)), 6, COLOUR_POINT, -1)
    for line_number, line in enumerate(help_lines):
        cv2.putText(canvas, line, (20, 40 + line_number * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOUR_TEXT, 2)
    if message:
        cv2.putText(canvas, message, (20, 40 + len(help_lines) * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOUR_TEXT, 2)
    return canvas

# Find boxes drawn on strongly coloured shapes.
def highlight_boxes(image: np.ndarray, objects: list[dict]) -> list[np.ndarray]:
    if not objects:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    highlighted = []
    for obj in objects:
        corners = np.array(obj["corners"], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
        saturation = hsv[:, :, 1][mask > 0]
        if saturation.size and float(np.median(saturation)) >= HIGHLIGHT_MIN_SATURATION:
            highlighted.append(corners)
    return highlighted

@dataclass
class View:

    base: np.ndarray
    scale: float
    image_size: tuple[int, int]

    @classmethod
    def build(cls, image: np.ndarray) -> View:
        height, width = image.shape[:2]
        # Shrink large images to fit the screen.
        scale = min(1.0, DISPLAY_MAX_WIDTH / width)
        base = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image.copy()
        return cls(base=base, scale=scale, image_size=(width, height))

    # Turn a screen click into an image pixel.
    def to_image(self, x: int, y: int) -> tuple[int, int]:
        width, height = self.image_size
        image_x = round(x / self.scale)
        image_y = round(y / self.scale)
        return (min(max(image_x, 0), width - 1),
                min(max(image_y, 0), height - 1))

    def render(
        self,
        objects: list[dict],
        selected: int,
        points: list[tuple[int, int]],
        message: str,
        highlight: list[np.ndarray] | None = None,
        help_lines: tuple[str, ...] = (),
    ) -> np.ndarray:
        return render_frame(self.base.copy(), objects, selected, self.scale,
                            thickness=3, font=1.0, points=points,
                            message=message, highlight=highlight,
                            help_lines=help_lines)

def usable_screen() -> tuple[int, int]:
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except OSError:
            pass

        rect = wintypes.RECT()
        if user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return rect.right - rect.left, rect.bottom - rect.top
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except (ImportError, AttributeError, OSError):
        return 0, 0

def opening_window_size(base: np.ndarray) -> tuple[int, int]:
    screen_width, screen_height = usable_screen()
    if screen_width <= 0 or screen_height <= 0:
        return base.shape[1], base.shape[0]

    height, width = base.shape[:2]
    target_width = int(min(screen_width * WINDOW_START_SHARE, width))
    target_height = int(target_width * height / width)
    if target_height > screen_height:
        target_height = int(screen_height * WINDOW_START_SHARE)
        target_width = int(target_height * width / height)
    return max(320, target_width), max(240, target_height)

class Session:

    def __init__(self, image_path: Path, image: np.ndarray, class_id: int = 0) -> None:
        self.paths = paths_for(image_path)
        self.image = image
        self.view = View.build(image)

        self.class_id = class_id if 0 <= class_id < len(CLASSES) else 0
        self.objects: list[dict] = []
        self.points: list[tuple[int, int]] = []
        self.selected = -1
        self.outlining = False
        self.stop = False
        self.done = False
        self._last_payload: str | None = None
        self._painted_message = ""

        saved = load_state(self.paths)
        # No saved work, so start empty.
        if saved is None:
            self.status = "draft"
            note = "press 1-5 to pick a class, then M to outline (or A if empty)"
        else:
            self.status = saved["status"]
            self.objects = saved["objects"]
            self.points = [tuple(p) for p in saved["pending_points"]]
            note = "restored saved work"
        self.selected = 0 if self.objects else -1
        self.highlight = highlight_boxes(image, self.objects)
        self.message = f"[{note}] {len(self.objects)} obj | {KEYS_HELP}"

        self.dispatch = {
            ord("s"): self.approve,
            ord("a"): self.approve_empty,
            ord("f"): self.flag,
            ord("m"): self.toggle_outlining,
            13: self.close_outline,
            10: self.close_outline,
            ord("b"): self.undo_point,
            ord("x"): self.discard_outline,
            ord("c"): self.cycle_class,
            ord("d"): self.delete_object,
            ord("n"): partial(self.select, 1),
            ord("p"): partial(self.select, -1),
            32: self.next_image,
            ord("q"): self.quit,
        }

    # Save current work to the sidecar file.
    def persist(self, status: str) -> None:
        self.status = status
        payload = json.dumps(
            {"status": status, "objects": self.objects, "pending_points": self.points},
            indent=2,
        )
        if payload != self._last_payload:
            atomic_write_text(self.paths.sidecar, payload)
            self._last_payload = payload
        # Only approved work keeps old label files.
        if status != "approved" and self.paths.label.is_file():
            self.paths.label.unlink()

    # Write all boxes as YOLO label lines.
    def write_labels(self) -> tuple[bool, list[int]]:
        size = (self.image.shape[1], self.image.shape[0])
        lines, invalid = [], []
        for index, obj in enumerate(self.objects):
            line = corners_to_yolo_line(obj["corners"], obj["class_id"], size)
            if line:
                lines.append(line)
            else:
                invalid.append(index)
        if invalid:
            return False, invalid
        atomic_write_text(self.paths.label, "\n".join(lines) + "\n")
        return True, []

    def write_preview(self) -> Path | None:
        preview = render_frame(self.image.copy(), self.objects)
        out = self.paths.preview
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), preview):
            print(f"warning: preview NOT saved: {out}")
            return None
        return out

    def approve(self) -> None:
        if self.points:
            self.message = "outline unfinished: Enter to close, X discard, then S"
            return
        if not self.objects:
            self.persist("draft")
            self.message = "0 objects: press A to approve as empty, F to flag, or M to outline"
            return
        written, invalid = self.write_labels()
        if not written:
            self.persist("draft")
            self.message = (
            f"NOT saved: {len(invalid)} invalid obj {invalid} - "
            "fix or delete, then S"
        )
            return
        self.persist("approved")
        self.write_preview()
        print(f"{self.paths.image.name}: approved, {len(self.objects)} labels")
        self.done = True

    def approve_empty(self) -> None:
        if self.objects:
            self.message = (
                f"A ignored: {len(self.objects)} object(s) outlined - S to save, "
                "D to delete, F to flag"
            )
            return
        self.persist("approved")
        self.write_preview()
        print(f"{self.paths.image.name}: approved as EMPTY (background image)")
        self.done = True

    def flag(self) -> None:
        self.persist("flagged")
        print(f"{self.paths.image.name}: flagged (stale labels removed)")
        self.done = True

    def next_image(self) -> None:
        self.done = True

    def quit(self) -> None:
        self.persist(self.status)
        self.stop = True
        self.done = True

    def toggle_outlining(self) -> None:
        self.outlining = not self.outlining
        if self.outlining:
            self.message = (
                f"outlining {CLASSES[self.class_id]}: click {OUTLINE_POINTS} corners "
                f"(or 1-5 to change class), X to stop"
            )
        else:
            self.persist(self.status)
            self.message = f"outlining off | {KEYS_HELP}"

    def close_outline(self) -> None:
        if not self.outlining:
            return
        corners = self.finish_outline()
        if corners is None:
            self.message = f"need {OUTLINE_POINTS} points - {len(self.points)} clicked so far"
            return
        self.objects.append({"class_id": self.class_id, "corners": corners.tolist()})
        self.selected = len(self.objects) - 1
        self.persist("draft")
        self.refresh_highlight()
        self.message = (
            f"added {CLASSES[self.class_id]} #{self.selected} - click the next shape's "
            f"{OUTLINE_POINTS} corners, 1-5 changes class, S saves"
        )

    def finish_outline(self) -> np.ndarray | None:
        if len(self.points) < 3:
            return None
        corners = polygon_to_corners(np.array(self.points, dtype=np.float32))
        self.points = []
        return corners

    def undo_point(self) -> None:
        if self.outlining and self.points:
            self.points.pop()
            self.persist(self.status)

    def discard_outline(self) -> None:
        if self.outlining:
            self.points = []
            self.persist(self.status)
            self.outlining = False
            self.message = "outline discarded"

    def delete_object(self) -> None:
        if not 0 <= self.selected < len(self.objects):
            return
        self.objects.pop(self.selected)
        self.selected = min(self.selected, len(self.objects) - 1)
        self.persist("draft")
        self.refresh_highlight()

    def refresh_highlight(self) -> None:
        self.highlight = highlight_boxes(self.image, self.objects)

    def set_class(self, class_id: int) -> None:
        if not 0 <= class_id < len(CLASSES):
            return
        self.class_id = class_id
        self.message = (
            f"class = {CLASSES[class_id]} for the next outline "
            "(C re-classes a drawn box)"
        )

    def cycle_class(self) -> None:
        if not 0 <= self.selected < len(self.objects):
            self.message = "C: no object selected - N/P selects one"
            return
        obj = self.objects[self.selected]
        obj["class_id"] = (obj["class_id"] + 1) % len(CLASSES)
        self.persist("draft")
        self.message = f"object #{self.selected} is now {CLASSES[obj['class_id']]} (C cycles)"

    def select(self, step: int) -> None:
        if self.objects:
            self.selected = (self.selected + step) % len(self.objects)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: None) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or not self.outlining:
            return
        # Each left click adds one corner point.
        self.points.append(self.view.to_image(x, y))
        self.persist("draft" if self.status == "approved" else self.status)
        if len(self.points) >= OUTLINE_POINTS:
            self.close_outline()

    def object_fingerprint(self) -> tuple:
        return tuple(
            (obj["class_id"], tuple(tuple(corner) for corner in obj["corners"]))
            for obj in self.objects
        )

    def view_state(self) -> tuple:
        highlight_shape = tuple(len(corners) for corners in self.highlight)
        return (
            self.status,
            self.selected,
            self.message,
            self.outlining,
            len(self.points),
            self.object_fingerprint(),
            highlight_shape,
        )

    def run(self) -> tuple[str, bool]:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, *opening_window_size(self.view.base))
        cv2.moveWindow(WINDOW, 0, 0)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        painted = None
        while not self.done:
            current = self.view_state()
            if current != painted:
                frame = self.view.render(self.objects, self.selected,
                                         self.points, self.message, self.highlight,
                                         help_lines=KEYS_HELP)
                cv2.imshow(WINDOW, frame)
                painted = current
            # Main loop waits 30 ms for one key.
            key = cv2.waitKey(30) & 0xFF

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self.persist(self.status)
                self.stop = True
                break

            if key in CLASS_KEYS:
                self.set_class(CLASS_KEYS[key])
            elif key in self.dispatch:
                self.dispatch[key]()
            if self.message != self._painted_message:
                print(f"  {self.message}")
                self._painted_message = self.message

        cv2.destroyWindow(WINDOW)
        return self.status, self.stop

def main() -> None:
    images = list_images(ROOT)
    if not images:
        raise SystemExit(f"no images found under {ROOT}")

    if START_AT:
        target = Path(START_AT).stem
        stems = [p.stem for p in images]
        if target not in stems:
            raise SystemExit(
                f"START_AT={START_AT!r} matches no image under {ROOT} "
                f"(use a stem such as {stems[0]!r})"
            )
        first = stems.index(target)
        print(f"resuming at {START_AT} (skipping {first} earlier images)")
        images = images[first:]

    print(f"{len(images)} images under {ROOT}. Q quits at any time.")

    class_id = 0
    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{index + 1}/{len(images)}] skipped: {image_path.name} (unreadable)")
            continue
        session = Session(image_path, image, class_id)
        status, stop = session.run()
        class_id = session.class_id
        print(f"[{index + 1}/{len(images)}] {status}: {image_path.name}")
        if stop:
            break
    # Close all OpenCV windows at exit.
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
