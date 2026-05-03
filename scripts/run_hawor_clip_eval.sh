#!/usr/bin/env bash
# HaWoR clip pipeline from repo root (prepare → inference → video + poses npz).
# No HOT3D GT metrics: no --gt-seq-dir, no joint-vs-GT eval (--skip-metrics).
# Override: CLIP_DIR, SEQ_NAME, HAWOR_WORK_ROOT, OUT_DIR, EXPORT_VIDEO=0, PINHOLE_* for pinhole mode, etc.
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
HAWOR_WORK_ROOT="${HAWOR_WORK_ROOT:-${ROOT}/example/hawor_work}"
OUT_DIR="${OUT_DIR:-${ROOT}/example/hawor_out/${SEQ_NAME}}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/weights/hawor/checkpoints/hawor.ckpt}"
INFILLER="${INFILLER:-${ROOT}/weights/hawor/checkpoints/infiller.pt}"
EXPORT_VIDEO="${EXPORT_VIDEO:-1}"
EXPORT_VIDEO_RENDERER="${EXPORT_VIDEO_RENDERER:-mesh_pyrender}"
EXPORT_VIDEO_FPS="${EXPORT_VIDEO_FPS:-30}"
MAX_FRAMES="${MAX_FRAMES:-}"
# Pinhole mode (optional): set PINHOLE_IMAGE_DIR and CAM_K JSON; then CLIP_DIR / HOT3D_MODULE_PATH are unused for prepare.
PINHOLE_IMAGE_DIR="${PINHOLE_IMAGE_DIR:-}"
CAM_K="${CAM_K:-}"

ARGS=(
  python scripts/eval_hawor_hot3d_clip.py
  --skip-metrics
  --hawor-work-root "${HAWOR_WORK_ROOT}"
  --seq-name "${SEQ_NAME}"
  --checkpoint "${CHECKPOINT}"
  --infiller-weight "${INFILLER}"
  --out-dir "${OUT_DIR}"
)
if [[ -n "${PINHOLE_IMAGE_DIR}" && -n "${CAM_K}" ]]; then
  ARGS+=(--pinhole-image-dir "${PINHOLE_IMAGE_DIR}" --cam-k "${CAM_K}")
  if [[ -n "${PINHOLE_EXTENSIONS:-}" ]]; then
    ARGS+=(--pinhole-extensions "${PINHOLE_EXTENSIONS}")
  fi
else
  ARGS+=(
    --clip-dir "${CLIP_DIR}"
    --hot3d-module-path "${HOT3D_MODULE_PATH}"
  )
fi
if [[ -n "${MAX_FRAMES:-}" ]]; then
  ARGS+=(--max-frames "${MAX_FRAMES}")
fi
if [[ "${EXPORT_VIDEO}" == "1" ]]; then
  ARGS+=(--export-video --export-video-renderer "${EXPORT_VIDEO_RENDERER}" --export-video-fps "${EXPORT_VIDEO_FPS}")
fi
if [[ "${NO_SAVE_POSES:-0}" == "1" ]]; then
  ARGS+=(--no-save-poses)
fi

exec "${ARGS[@]}"
