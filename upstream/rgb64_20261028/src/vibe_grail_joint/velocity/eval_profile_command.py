"""Held-out profile trials reuse the physical terrain's exact height function."""

from dataclasses import dataclass
import itertools

from .eval_command import EvaluationCommand, EvaluationCommandCfg
from .recovery_terrain import ProfileMotionCommand, ProfileMotionCommandCfg


class ProfileEvaluationCommand(EvaluationCommand, ProfileMotionCommand):
    """Deterministic case initialization and irregular-profile ground heights."""


@dataclass(kw_only=True)
class ProfileEvaluationCommandCfg(EvaluationCommandCfg, ProfileMotionCommandCfg):
    def build(self, env):
        return ProfileEvaluationCommand(self, env)


def profile_cases(settings):
    cases = []
    variants = settings.get('profile_variants', 3)
    for level, variant, kind, stream, repeat in itertools.product(
        [0, 3, 5, 7, 10], [1, 2], range(3), [1, 3, 4], range(2)
    ):
        cases.append(dict(id=len(cases), tile=(level * variants + variant) * 4 + kind,
                          stream=stream, speed=settings['planner_speeds'][stream],
                          phase=[0, 10][repeat], spawn_xy=[[-.15, -.15], [.15, .15]][repeat]))
    for stream, repeat in itertools.product([1, 3, 4], range(4)):
        cases.append(dict(id=len(cases), tile=3, stream=stream,
                          speed=settings['planner_speeds'][stream], phase=repeat * 5,
                          spawn_xy=[0., (repeat-1.5) * .1]))
    return cases
