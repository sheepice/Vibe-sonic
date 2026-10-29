#!/usr/bin/env bash
# Project-local runtime for this G1's JetPack 6.2.1 / CUDA 12.6 / TRT 10.3.
# An interactive .bashrc otherwise loads ~/TensorRT 10.7 and a conflicting
# nvidia/libcudla.so.1 symlink. Do not inherit or modify that global setup.
set -euo pipefail
vibe_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/lib"
exec "$vibe_root/.venv-real/bin/python" "$@"
