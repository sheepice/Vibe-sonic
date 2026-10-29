#!/usr/bin/env bash
# Rebuild only the project-private Python extension against existing V4L2.
# Source must be pinned to bf2778061d5dd29776e9aca8765f75852671760b.
set -euo pipefail
cd /home/unitree/vibe_g1
cmake -S third_party/librealsense-2.58.1 -B third_party/librealsense-build-py311 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=/home/unitree/vibe_g1/.venv-real/bin/python \
  -DPython_EXECUTABLE=/home/unitree/vibe_g1/.venv-real/bin/python \
  -DBUILD_SHARED_LIBS=OFF -DFORCE_RSUSB_BACKEND=OFF \
  -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_GLSL_EXTENSIONS=OFF -DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_DDS=OFF -DBUILD_ROSBAG2=OFF \
  -DBUILD_RS2_ALL=OFF -DCHECK_FOR_UPDATES=OFF
cmake --build third_party/librealsense-build-py311 --target pyrealsense2 -j4
# No system install. Packaging refuses to overwrite a prior staging directory.
.venv-real/bin/python hardware/deployment/package_realsense_wheel.py \
  --source third_party/librealsense-2.58.1 --build third_party/librealsense-build-py311 \
  --staging third_party/realsense-wheel-v4l2-rebuild-py311 --output wheelhouse \
  --local-version v4l2
