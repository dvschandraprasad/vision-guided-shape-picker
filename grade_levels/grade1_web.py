"""Grade level I (grade E) - web dashboard.

One cycle: the system detects the three assigned shapes and the operator
presses Pick (then the robot's yellow button) once per part. No shape or
colour choice - the most confident assigned part is picked first.

    python grade_levels/grade1_web.py --camera --send --host 192.168.125.205
"""
from grade_common import GradeController, launch


class Grade1Controller(GradeController):
    """Grade I rules: one cycle, no shape or colour choice."""
    LEVEL = "I"
    RULES = "one cycle, all three shapes, any order (no shape or colour choice)"
    SELECT_SHAPE = False
    SELECT_COLOUR = False
    MULTI_CYCLE = False


if __name__ == "__main__":
    raise SystemExit(launch(Grade1Controller))
