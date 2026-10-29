#!/usr/bin/env bash
# Private Python 3.11 bindings only; retain system CUDA/TensorRT/JetPack.
set -euo pipefail
cd /home/unitree/vibe_g1
# NVIDIA TensorRT source: 5b990f0a739d8faf962dfe54f4829942633c639c.
# Parser helper headers: 62bdde2a04fcd53c2409cb895ee18db445b7e755.
# See records/deployment_20261019/optimization_environment.json for hashes.
VIBE_PY_INCLUDE=$(.venv-real/bin/python -c 'import sysconfig; print(sysconfig.get_path("include"))')
cmake -S third_party/TensorRT-10.3/python -B third_party/TensorRT-10.3-build-py311 \
  -DCMAKE_BUILD_TYPE=Release -DTARGET=aarch64 \
  -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=11 -DTENSORRT_MODULE=tensorrt \
  -DTENSORRT_ROOT="$PWD/third_party/TensorRT-10.3" \
  -DTENSORRT_BUILD=/usr/lib/aarch64-linux-gnu \
  -DONNX_INC_DIR=/usr/include/aarch64-linux-gnu \
  -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include \
  -DPYBIND11_DIR="$PWD/third_party/librealsense-build-py311/third-party/pybind11/include" \
  -DPY_INCLUDE="$VIBE_PY_INCLUDE" -DPY_CONFIG_INCLUDE="$VIBE_PY_INCLUDE" \
  -DEXT_PATH="$PWD/third_party"
cmake --build third_party/TensorRT-10.3-build-py311 -j4
.venv-trt-build/bin/python hardware/deployment/package_tensorrt_wheel.py \
  --source third_party/TensorRT-10.3 --build third_party/TensorRT-10.3-build-py311 \
  --staging third_party/tensorrt-wheel-rebuild-py311 --output wheelhouse
