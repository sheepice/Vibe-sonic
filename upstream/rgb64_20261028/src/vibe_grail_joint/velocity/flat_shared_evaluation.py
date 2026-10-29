"""Flat gait recording using the unchanged bounded training-sharing guard."""

import sys

from . import closeup_shared_guard as launcher


def main():
    launcher.MODULE = "vibe_grail_joint.velocity.flat_shared_evaluation"
    if "--worker" in sys.argv[1:]:
        from . import evaluate_flat_closeup, evaluate_uniform20_closeup

        evaluate_uniform20_closeup.main = evaluate_flat_closeup.main
    launcher.main()


if __name__ == "__main__":
    main()
