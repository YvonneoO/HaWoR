#!/usr/bin/env bash
# HaWoR clip eval from repo root. Override: CLIP_DIR, GT_SEQ_DIR, SEQ_NAME, EXPORT_VIDEO=0, etc.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -n "${GPU_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

CLIP_DIR="${CLIP_DIR:-${ROOT}/example/clip/clip-001881}"
SEQ_NAME="${SEQ_NAME:-clip-001881}"
HOT3D_MODULE_PATH="${HOT3D_MODULE_PATH:-/workspace/hot3d-private/hot3d}"
export HOT3D_REPO_DIR="${HOT3D_REPO_DIR:-${HOT3D_MODULE_PATH}}"
GT_SEQ_DIR="${GT_SEQ_DIR:-${ROOT}/example/clip_gt/${SEQ_NAME}}"
HAWOR_WORK_ROOT="${HAWOR_WORK_ROOT:-${ROOT}/example/hawor_work}"
OUT_DIR="${OUT_DIR:-${ROOT}/example/hawor_out/${SEQ_NAME}}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/weights/hawor/checkpoints/hawor.ckpt}"
INFILLER="${INFILLER:-${ROOT}/weights/hawor/checkpoints/infiller.pt}"
EXPORT_VIDEO="${EXPORT_VIDEO:-1}"
EXPORT_VIDEO_RENDERER="${EXPORT_VIDEO_RENDERER:-mesh_pyrender}"
EXPORT_VIDEO_FPS="${EXPORT_VIDEO_FPS:-30}"
MAX_FRAMES="${MAX_FRAMES:-}"

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
if [[ -n "${MAX_FRAMES:-}" ]]; then
  ARGS+=(--max-frames "${MAX_FRAMES}")
fi
if [[ "${EXPORT_VIDEO}" == "1" ]]; then
  ARGS+=(--export-video --export-video-renderer "${EXPORT_VIDEO_RENDERER}" --export-video-fps "${EXPORT_VIDEO_FPS}")
fi

exec "${ARGS[@]}"
