"""Continuous-route recording with pinned training identities and memory limits."""

import sys
from . import closeup_shared_guard as launcher


def main():
    launcher.MODULE = "vibe_grail_joint.velocity.stairs_dance_shared_evaluation"
    if "--worker" in sys.argv[1:]:
        from . import evaluate_stairs_dance, evaluate_uniform20_closeup
        evaluate_uniform20_closeup.main = evaluate_stairs_dance.main
    launcher.main()


if __name__ == "__main__":
    main()
