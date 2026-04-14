#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DROID_DIR="${REPO_DIR}/thirdparty/DROID-SLAM"

export MAX_JOBS="${MAX_JOBS:-8}"
export VERBOSE=1

cd "$DROID_DIR"
python setup.py build_ext --verbose install
