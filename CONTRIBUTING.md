# Contributing

This repository holds a graded university final project. The program logic in `scripts/`,
`grade_levels/`, `rapid/`, `ServerCommuncatio.sys` and the shell launchers is kept as it was
handed in.

You are welcome to:

- open an issue for a bug, a question or a setup problem;
- improve the documentation (`README.md`, `docs/`);
- add tests that need no camera, robot or display.

If you change code in a fork:

1. Create a virtual environment with Python 3.12 or newer and install CPU torch first, then
   `requirements.txt` (see the README).
2. Run `python -m pytest` before and after your change. GitHub Actions runs the same tests on
   every push and pull request (`.github/workflows/tests.yml`).
3. Keep the scripts importable without hardware: import `pypylon` and `ultralytics` inside
   functions, as the existing code does.
4. Do not commit captured frames, datasets, `.venv/` or `__pycache__/` (see `.gitignore`).
5. Do not change `data/calibration.json` or `data/angle_teach.json` unless you re-calibrated
   or re-taught on the real cell, and say so in the commit message.
6. Keep RAPID files with CRLF line endings (`.gitattributes` does this).

If you have one of the files listed in the README section "Missing files", copy it to the
place given there and open a pull request. No other change is needed.

Use short, clear commit messages in English.
