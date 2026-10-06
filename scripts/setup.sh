#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3.11}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
docker build --tag recursive-ssd-eval:0.3.1 docker
.venv/bin/python -m recursive_ssd.cli prepare
.venv/bin/python -m recursive_ssd.cli preflight
