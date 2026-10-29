"""Continuous-route recording with pinned training identities and memory limits."""

import sys
from . import closeup_shared_guard as launcher


def main():
    launcher.MODULE = "vibe_grail_joint.velocity.long_walk_shared_evaluation"
    if "--worker" in sys.argv[1:]:
        from . import evaluate_long_walk, evaluate_uniform20_closeup
        evaluate_uniform20_closeup.main = evaluate_long_walk.main
    launcher.main()


if __name__ == "__main__":
    main()
