#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
exec .venv/bin/python -m sim2sim.server "$@"
