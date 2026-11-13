# Grade levels - one dashboard per grade

Three entry scripts built on `scripts/dashboard_web.py`. **No file in `scripts/` or `rapid/`
is changed**: camera, AI inference, coordinate translation, robot link and the web GUI are
the same code for every grade. Each grade is one small `GradeController` subclass that only
sets its rules.

| Script | Grade | What the operator can do |
| --- | --- | --- |
| `grade1_web.py` | I (E) | Pick only. One cycle, all three shapes, any order. Shape and colour lists hidden. |
| `grade2_web.py` | II (D) | Choose a **colour**. A cycle ends when the three shapes are sent; **Start next cycle** is required before the next one. |
| `grade3_web.py` | III (C, B-A) | Choose the **shape** (required) and colour for every pick. Up to **3 attempts per shape per cycle** (1 + 2 retries). **Start next cycle** resets attempts and the shape choice. |

All grades:

- Only the assigned shapes (circle, hexagon, triangle - `ALLOWED_CLASS_IDS` in `pipeline.py`)
  are reported. Square and star are dropped right after inference and are not in the shape list.
- **Shape check:** a circle / hexagon / triangle detection is dropped when its outline is really
  a **star** (deep notches: outline area / convex-hull area below 0.80; stars measure about 0.6,
  real shapes 0.95+) or a **square** (4 corners and fills 0.90+ of its tightest box; squares
  measure 0.96+, hexagons about 0.8 with 6 corners). So a star or square the model mislabels is
  never picked. The console prints `rejected triangle ... outline looks like a star` (or square).
  Turn off with `--no-shape-check`.
- **Colour:** judged on a patch at the part's centre against the sheet around it, so white parts
  read *white* (not blue/green) and black parts *black*. Orange parts count as *red*.
- **Pins** (the small round magnets holding the sheet) are dropped: detections under 38 mm across
  are ignored (parts are 45 mm or more, pins 10-30 mm). `--min-size-mm 0` turns this off.
- **White parts are ignored** (instructor's instruction) and *white* is not in the colour list.
  `--allow-white` uses them again.
- The console prints each ignored detection once, e.g. `ignored circle ...: a pin, not a part`.
- `--conf 0.5` raises the minimum model confidence (default 0.25); a lower value such as
  `--conf 0.15` makes it report less certain detections.
- Inference time is printed in the console (`inference: X ms`) and shown on the page; over 3 s turns red.
- Robot side unchanged: `rapid/MainModule_angle.mod`. One **Pick** on the page + one **yellow button** = one part (wait for "Done." before the next Pick).
- A missed pick is retried by pressing Pick again (the part is still in view).

## Run on the Pi

Copy this `grade_levels` folder into `~/Documents/pi4` (next to `scripts/`), then:

```bash
chmod +x grade_levels/run_grade.sh
./grade_levels/run_grade.sh 1      # or 2, or 3
```

or directly, with the same options as `dashboard_web.py`:

```bash
python grade_levels/grade3_web.py --camera --send --host 192.168.125.205
```

Open `http://<pi-address>:8080` on the PC, then start the RAPID program within 30 s.

## Files

| File | Purpose |
| --- | --- |
| `grade_common.py` | `GradeController` (cycles, attempts, rules), `AssignedShapeDetector`, `GradeHandler` (`POST /next_cycle`), page builder, `launch()` |
| `grade1_web.py`, `grade2_web.py`, `grade3_web.py` | one subclass each - only the grade's rule settings |
| `run_grade.sh` | Pi launcher, like `run_dashboard.sh` |
| `../tests/test_grade_levels.py` | 43 tests, no camera/robot: `python -m pytest tests -q` |

## Design (for the OOP part)

`GradeController` extends `dashboard_web.Controller` (open/closed: the existing class is extended,
not edited). The grade rules are class attributes plus one hook (`_start_cycle`), so each grade
only states what differs (template method). `launch()` injects the grade classes into
`dashboard_web.main()`, so start-up, threads and command-line options stay in one place.
