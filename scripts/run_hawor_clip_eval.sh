#!/usr/bin/env bash
# One-shot HaWoR + metrics (+ optional mp4) for a single HOT3D-clip folder.
# Run from HaWoR repo root (same as demo.py).
#
# Usage:
#   bash scripts/run_hawor_clip_eval.sh
# Or override paths:
#   CLIP_DIR=... GT_SEQ_DIR=... SEQ_NAME=... bash scripts/run_hawor_clip_eval.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CLIP_DIR="${CLIP_DIR:-${ROOT}/example/clip/clip-001881}"
SEQ_NAME="${SEQ_NAME:-clip-001881}"
HOT3D_MODULE_PATH="${HOT3D_MODULE_PATH:-/workspace/hot3d-private/hot3d}"
GT_SEQ_DIR="${GT_SEQ_DIR:-/data/hot3d-clip/clip_gt/train/${SEQ_NAME}}"
HAWOR_WORK_ROOT="${HAWOR_WORK_ROOT:-${ROOT}/example/hawor_work}"
OUT_DIR="${OUT_DIR:-${ROOT}/example/hawor_out/${SEQ_NAME}}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/weights/hawor/checkpoints/hawor.ckpt}"
INFILLER="${INFILLER:-${ROOT}/weights/hawor/checkpoints/infiller.pt}"
EXPORT_VIDEO="${EXPORT_VIDEO:-1}"

ARGS=(
  python scripts/eval_hawor_hot3d_clip.py
  --clip-dir "${CLIP_DIR}"
  --gt-seq-dir "${GT_SEQ_DIR}"
  --hawor-work-root "${HAWOR_WORK_ROOT}"
  --seq-name "${SEQ_NAME}"
  --hot3d-module-path "${HOT3D_MODULE_PATH}"
  --checkpoint "${CHECKPOINT}"
  --infiller-weight "${INFILLER}"
  --out-dir "${OUT_DIR}"
)
if [[ "${EXPORT_VIDEO}" == "1" ]]; then
  ARGS+=(--export-video)
fi

exec "${ARGS[@]}"
