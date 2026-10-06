#!/usr/bin/env bash
# Use the active project Conda environment, or an explicit existing interpreter.
if [[ -n "${PYTHON:-}" ]]; then
  RECURSIVE_SSD_PYTHON="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  RECURSIVE_SSD_PYTHON="$CONDA_PREFIX/bin/python"
elif [[ -x .venv/bin/python ]]; then
  RECURSIVE_SSD_PYTHON="$(pwd)/.venv/bin/python"
else
  echo 'Activate a project Conda environment (Python 3.11/3.12), or set PYTHON to its interpreter.' >&2
  return 1
fi
RECURSIVE_SSD_PYTHON="$("$RECURSIVE_SSD_PYTHON" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), "Python 3.11/3.12 required"; print(sys.executable)')"
export RECURSIVE_SSD_PYTHON
