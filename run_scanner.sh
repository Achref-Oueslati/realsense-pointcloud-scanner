#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/realsense3d"
source .venv/bin/activate
exec python src/scanner_gui.py

