# Changelog

All notable changes to this repository are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-09

First public release of the INA700 final project as a GitHub repository.

### Included (unchanged from the hand-in)

- `scripts/` - vision pipeline, calibration, angle teach, web dashboard, tools.
- `grade_levels/` - grade I, II and III dashboards and their README.
- `rapid/MainModule_angle.mod` and `ServerCommuncatio.sys` - ABB RAPID side.
- `tests/` - pytest suite.
- `data/calibration.json`, `data/angle_teach.json` - lab-cell calibration and angle teach.
- `data/captured_frames/calibration_frame.png`, `overlay_live.png` - sample frames.
- `runs/obb/obb_train/weights/best.onnx` and `best_ncnn_model/` - trained YOLO OBB weights.
- `requirements.txt`, `run_dashboard.sh`, `grade_levels/run_grade.sh`.

### Added

- `README.md`, `docs/pipeline.md`, `docs/robot_setup.md`, `docs/images/` (small copies of the
  two sample frames).
- `LICENSE` (MIT), `CONTRIBUTING.md`, this `CHANGELOG.md`.
- `.gitignore`, `.gitattributes` (keeps RAPID files byte-for-byte).
- `requirements-dev.txt` (test-only dependencies).
- `pytest.ini` - puts `scripts/`, `grade_levels/` and `tests/` on the import path for pytest.
- `tests/conftest.py` - resets the angle-teach values in `pipeline.py` after each test, so the
  shipped `data/angle_teach.json` cannot leak between tests. It also leaves out or skips only
  the tests that need a known-missing file (with the reason printed); they run again as soon
  as the file is added. No program logic changed, no test file changed.
- `.github/workflows/tests.yml` - GitHub Actions: a `light` job (`requirements-dev.txt`) and a
  `full` job (CPU torch + `requirements.txt`, Python 3.12 and 3.13).

### Changed

- `run_dashboard.sh` and `grade_levels/run_grade.sh` are marked executable (file mode only, the
  content is unchanged).

### Removed

- All `__pycache__/` folders and `*.pyc` files.

### Known issues

- `scripts/dashboard_new.py` and `scripts/dashboard.py` are missing from the hand-in, so the web
  dashboards, `web_view.py`, the launchers, `test_dashboard.py` and `test_grade_levels.py`
  cannot run.
- `rapid/updated_robo.mod` is missing, so `tests/test_updated_robo.py` is skipped.
- The test frame `scripts/frame_20261030_171238_018578.png` is missing, so 7 tests in
  `tests/test_angle_teach.py` are skipped.
- `requirements.txt` pins `numpy==2.5.3`, which needs Python 3.12 or newer.

### Test run for this release

Python 3.13, `opencv-python 5.0.0.93`, `numpy 2.5.3`, `pytest 9.1.1`, plus `pypylon`, `ncnn`
and `onnxruntime`. **`torch` / `ultralytics` could not be installed** in the packaging
environment (the CPU torch index was not reachable), and there was no Tk.

```
364 passed, 21 skipped   (test_dashboard.py and test_grade_levels.py not collected)
```

The same result in a clean virtual environment with only `requirements-dev.txt`.

| Result | Cause |
| --- | --- |
| 5 skipped in `test_pipeline.py` | `ultralytics` not installed |
| 9 skipped in `test_updated_robo.py` | `rapid/updated_robo.mod` not in the repo |
| 7 skipped in `test_angle_teach.py` | sample frame `frame_20261030_171238_018578.png` not in the repo |
| 2 files not collected | `test_dashboard.py` (no `dashboard.py`), `test_grade_levels.py` (no `dashboard_new.py`) |

Before the missing-file handling in `tests/conftest.py`, plain `python -m pytest` stopped at the
2 collection errors; with `--continue-on-collection-errors` it gave
`14 failed, 364 passed, 7 skipped, 2 errors` (the 14 failures were the 5 + 9 tests above).
Without the state-restore fixture there were also 10 more failures in `test_pipeline.py` and
`test_verify_positions.py`, only because of test order (every file passed on its own).
