#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/python_env.sh
RESEARCH_SETUP_QUEUE=""
RESEARCH_SETUP_SECONDS=1800
while (($#)); do
  case "$1" in
    --queue) RESEARCH_SETUP_QUEUE="${2:?--queue requires a directory}"; shift 2 ;;
    --seconds) RESEARCH_SETUP_SECONDS="${2:?--seconds requires a positive integer}"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/setup.sh --queue <original-budget-directory> [--seconds 1800]"
      echo "Initialize that authorization with scripts/research.py budget first."
      exit 0 ;;
    *) echo "Unknown setup argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$RESEARCH_SETUP_QUEUE" || ! "$RESEARCH_SETUP_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Provide --queue for the original budget and a positive --seconds limit." >&2
  exit 2
fi
RESEARCH_CHECK_SECONDS=$((RESEARCH_SETUP_SECONDS < 300 ? RESEARCH_SETUP_SECONDS : 300))
"$RECURSIVE_SSD_PYTHON" scripts/research.py setup --queue "$RESEARCH_SETUP_QUEUE" --seconds "$RESEARCH_SETUP_SECONDS"
"$RECURSIVE_SSD_PYTHON" scripts/research.py check --queue "$RESEARCH_SETUP_QUEUE" --seconds "$RESEARCH_CHECK_SECONDS"
"$RECURSIVE_SSD_PYTHON" scripts/research.py assets --queue "$RESEARCH_SETUP_QUEUE" --seconds "$RESEARCH_SETUP_SECONDS"
"$RECURSIVE_SSD_PYTHON" scripts/research.py model --queue "$RESEARCH_SETUP_QUEUE" --seconds "$RESEARCH_SETUP_SECONDS"
