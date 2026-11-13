"""Grade level III (grade C, B-A with the OOP structure) - web dashboard.

Adds to Grade II:
  * the operator chooses WHICH shape to pick (and optionally a colour) for
    every pick - Pick is refused until a shape is chosen,
  * each shape may be tried up to 3 times per cycle (first try + 2 retries);
    a failed pick is retried by simply pressing Pick again,
  * 'Start next cycle' resets the attempts and the shape choice.

    python grade_levels/grade3_web.py --camera --send --host 192.168.125.205
"""
from grade_common import GradeController, launch


class Grade3Controller(GradeController):
    """Grade III rules: shape choice per pick, 3 attempts per shape, cycles."""
    LEVEL = "III"
    RULES = ("choose the shape (and colour) for each pick; up to 2 retries per shape; "
             "'Start next cycle' starts the next one")
    SELECT_SHAPE = True
    SELECT_COLOUR = True
    SHAPE_REQUIRED = True
    MULTI_CYCLE = True
    BLOCK_WHEN_COMPLETE = False   # the last shape may still need a retry
    MAX_ATTEMPTS = 3              # first attempt + two retries

    def _start_cycle(self) -> None:
        # The user must select the object again in every cycle.
        """Clear the shape choice so it is selected again in every cycle."""
        self.model.selected = None


if __name__ == "__main__":
    raise SystemExit(launch(Grade3Controller))
