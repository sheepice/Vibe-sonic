#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null; then
  echo '需要 uv；当前电脑的 uv 位于 /home/wjw/.local/bin/uv。' >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python --no-cache -r requirements.txt
echo '环境就绪。运行 ./run.sh，然后打开 http://127.0.0.1:8080'
