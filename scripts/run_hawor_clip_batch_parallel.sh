#!/usr/bin/env bash
# Batch HaWoR clip eval: waves of NUM_GPUS concurrent jobs; each job sets
# CUDA_VISIBLE_DEVICES to a single index 0..NUM_GPUS-1 in the *current* device
# namespace (so parent may export CUDA_VISIBLE_DEVICES=0,1,2,3 and each child
# still pins one logical GPU).
#
# Discovers: TRAIN_ARIA/clip* (directories), e.g. clip_000123 or clip-001881.
#
# From HaWoR repo root:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   export HAWOR_DROID_BUFFER=192
#   export TRAIN_ARIA=/lp-dev/qianqian/hot3d-private/hot3d-clip/raw/train_aria
#   export HAWOR_WORK_ROOT=/path/to/hawor_work/train_aria
#   export OUT_DIR=/path/to/hawor_out/train_aria
#   export HOT3D_MODULE_PATH=/lp-dev/qianqian/hot3d-private/hot3d
#   ./scripts/run_hawor_clip_batch_parallel.sh
#
# Optional env: CLIP_GLOB='clip_00*'  EXPORT_VIDEO=1  SKIP_PREPARE=1
#               OVERWRITE_PREPARE=1  NUM_GPUS=4
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TRAIN_ARIA="${TRAIN_ARIA:-/lp-dev/qianqian/hot3d-private/hot3d-clip/raw/train_aria}"
NUM_GPUS="${NUM_GPUS:-4}"
export HAWOR_DROID_BUFFER="${HAWOR_DROID_BUFFER:-192}"

HAWOR_WORK_ROOT="${HAWOR_WORK_ROOT:-${ROOT}/example/hawor_work/train_aria}"
OUT_DIR="${OUT_DIR:-${ROOT}/example/hawor_out/train_aria}"
HOT3D_MODULE_PATH="${HOT3D_MODULE_PATH:-/lp-dev/qianqian/hot3d-private/hot3d}"
export HOT3D_REPO_DIR="${HOT3D_REPO_DIR:-${HOT3D_MODULE_PATH}}"

CHECKPOINT="${CHECKPOINT:-${ROOT}/weights/hawor/checkpoints/hawor.ckpt}"
INFILLER="${INFILLER:-${ROOT}/weights/hawor/checkpoints/infiller.pt}"

EXPORT_VIDEO="${EXPORT_VIDEO:-0}"
EXPORT_VIDEO_RENDERER="${EXPORT_VIDEO_RENDERER:-mesh_pyrender}"
EXPORT_VIDEO_FPS="${EXPORT_VIDEO_FPS:-30}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
OVERWRITE_PREPARE="${OVERWRITE_PREPARE:-0}"
CLIP_GLOB="${CLIP_GLOB:-clip*}"

mkdir -p "${HAWOR_WORK_ROOT}" "${OUT_DIR}" "${OUT_DIR}/_logs"

mapfile -t CLIPS < <(
  find "${TRAIN_ARIA}" -mindepth 1 -maxdepth 1 -type d -name "${CLIP_GLOB}" | sed 's|/$||' | sort
)

if [[ ${#CLIPS[@]} -eq 0 ]]; then
  echo "No directories matching ${CLIP_GLOB} under TRAIN_ARIA=${TRAIN_ARIA}" >&2
  exit 1
fi

echo "TRAIN_ARIA=${TRAIN_ARIA}"
echo "NUM_GPUS=${NUM_GPUS}  HAWOR_DROID_BUFFER=${HAWOR_DROID_BUFFER}"
echo "Clips to run: ${#CLIPS[@]}"

run_one() {
  local clip_dir="$1"
  local gpu="$2"
  local seq_name
  seq_name="$(basename "${clip_dir}")"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export HAWOR_DROID_BUFFER
  cd "${ROOT}"

  local -a py=(
    python scripts/eval_hawor_hot3d_clip.py
    --skip-metrics
    --hawor-work-root "${HAWOR_WORK_ROOT}"
    --seq-name "${seq_name}"
    --checkpoint "${CHECKPOINT}"
    --infiller-weight "${INFILLER}"
    --out-dir "${OUT_DIR}"
    --clip-dir "${clip_dir}"
    --hot3d-module-path "${HOT3D_MODULE_PATH}"
  )
  if [[ "${SKIP_PREPARE}" == "1" ]]; then
    py+=(--skip-prepare)
  fi
  if [[ "${OVERWRITE_PREPARE}" == "1" ]]; then
    py+=(--overwrite-prepare)
  fi
  if [[ "${EXPORT_VIDEO}" == "1" ]]; then
    py+=(--export-video --export-video-renderer "${EXPORT_VIDEO_RENDERER}" --export-video-fps "${EXPORT_VIDEO_FPS}")
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] start seq=${seq_name} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  set +e
  "${py[@]}" 2>&1 | tee "${OUT_DIR}/_logs/${seq_name}.log"
  local st="${PIPESTATUS[0]}"
  set -e
  if [[ "${st}" -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ok   seq=${seq_name}"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL seq=${seq_name} exit=${st} log=${OUT_DIR}/_logs/${seq_name}.log" >&2
    return "${st}"
  fi
}

# Waves of NUM_GPUS jobs: clip index i runs on GPU (i % NUM_GPUS).
i=0
for clip_dir in "${CLIPS[@]}"; do
  gpu=$((i % NUM_GPUS))
  run_one "${clip_dir}" "${gpu}" &
  ((++i))
  if (( i % NUM_GPUS == 0 )); then
    wait
  fi
done
wait

echo "Batch finished. Per-clip logs: ${OUT_DIR}/_logs/"
