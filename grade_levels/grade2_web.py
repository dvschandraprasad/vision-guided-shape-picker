"""Grade level II (grade D) - web dashboard.

Adds colour detection and continuous cycles to Grade I:
  * the operator chooses a colour - only parts of that colour are picked,
  * a cycle ends when all three shapes have been sent; the next cycle starts
    only after 'Start next cycle' is pressed (user input), with no restart,
  * inference time is printed in the console and shown on the page (limit 3 s).

    python grade_levels/grade2_web.py --camera --send --host 192.168.125.205
"""
from grade_common import GradeController, launch


class Grade2Controller(GradeController):
    """Grade II rules: colour choice, cycles started by the operator."""
    LEVEL = "II"
    RULES = ("choose a colour; each cycle picks the three shapes, "
             "then 'Start next cycle' starts the next one")
    SELECT_SHAPE = False
    SELECT_COLOUR = True
    MULTI_CYCLE = True
    BLOCK_WHEN_COMPLETE = True


if __name__ == "__main__":
    raise SystemExit(launch(Grade2Controller))
