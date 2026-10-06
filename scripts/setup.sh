#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/python_env.sh
"$RECURSIVE_SSD_PYTHON" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
"$RECURSIVE_SSD_PYTHON" -m pip install -e '.[test,evaluation]'
"$RECURSIVE_SSD_PYTHON" -m pip check
"$RECURSIVE_SSD_PYTHON" -m pytest -q
"$RECURSIVE_SSD_PYTHON" -m recursive_ssd.cli prepare
"$RECURSIVE_SSD_PYTHON" -m recursive_ssd.cli qualify
"$RECURSIVE_SSD_PYTHON" -m recursive_ssd.cli preflight
