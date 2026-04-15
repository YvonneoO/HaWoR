#!/usr/bin/env bash
set -euo pipefail

# run_docker.sh — build and run the HaWoR container
# Paths (override via env vars if needed):
#   HF_CACHE_DIR, MANO_ROOT, WEIGHTS_DIR, METRIC3D_DIR
#   HOT3D_DATASET_DIR, HOT3D_GT_ROOT, HOT3D_SPLIT_SUMMARY

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_TARGET="${GPU_TARGET:-a100}"
IMAGE_NAME="${IMAGE_NAME:-}"
DOCKERFILE="${DOCKERFILE:-}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"

HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
MANO_ROOT="${MANO_ROOT:-${MANO_DIR:-/lp-dev/qianqian/mano_v1_2}}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${REPO_DIR}/weights}"
METRIC3D_DIR="${METRIC3D_DIR:-${REPO_DIR}/thirdparty/Metric3D/weights}"
HOT3D_REPO_DIR="${HOT3D_REPO_DIR:-/lp-dev/qianqian/hot3d-private}"
HOT3D_DATASET_DIR="${HOT3D_DATASET_DIR:-/lp-dev/qianqian/hot3d-private/hot3d/dataset}"
HOT3D_GT_ROOT="${HOT3D_GT_ROOT:-/lp-dev/qianqian/hot3d_gt}"
HOT3D_SPLIT_SUMMARY="${HOT3D_SPLIT_SUMMARY:-${HOT3D_GT_ROOT}/split_summary.json}"
STAGE1_DIR="${STAGE1_DIR:-/lp-dev/qianqian/stage1_mano_gt_v2}"
export MANO_ROOT

BUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_TARGET="$2"
            shift 2
            ;;
        --build)
            BUILD=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [[ -z "$DOCKERFILE" ]]; then
    DOCKERFILE="${REPO_DIR}/Dockerfile.${GPU_TARGET}"
fi

if [[ -z "$IMAGE_NAME" ]]; then
    IMAGE_NAME="hawor-${GPU_TARGET}"
fi

CONTAINER_NAME="${CONTAINER_NAME:-${IMAGE_NAME//[:\/]/-}}"

if [[ "$BUILD" -eq 1 ]]; then
    echo "[run_docker] Building image: $IMAGE_NAME"
    DOCKER_BUILDKIT=0 docker build --no-cache -f "$DOCKERFILE" -t "$IMAGE_NAME" "$REPO_DIR"
fi

mkdir -p "$HF_CACHE_DIR" "$WEIGHTS_DIR" "$METRIC3D_DIR"

if [[ ! -d "${MANO_ROOT}/models" ]]; then
    echo "[run_docker] Expected MANO models directory not found: ${MANO_ROOT}/models" >&2
    exit 1
fi

if [[ ! -d "${HOT3D_DATASET_DIR}" ]]; then
    echo "[run_docker] HOT3D dataset dir not found: ${HOT3D_DATASET_DIR}" >&2
    exit 1
fi

if [[ ! -d "${HOT3D_GT_ROOT}" ]]; then
    echo "[run_docker] HOT3D GT root not found: ${HOT3D_GT_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${HOT3D_SPLIT_SUMMARY}" ]]; then
    echo "[run_docker] HOT3D split summary not found: ${HOT3D_SPLIT_SUMMARY}" >&2
    exit 1
fi

echo "[run_docker] HOT3D dataset: ${HOT3D_DATASET_DIR}"
echo "[run_docker] HOT3D GT root: ${HOT3D_GT_ROOT}"
echo "[run_docker] HOT3D split summary: ${HOT3D_SPLIT_SUMMARY}"
echo "[run_docker] Entering interactive bash..."
docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    --gpus "${DOCKER_GPUS}" \
    --shm-size=16gb \
    --network=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/root/.cache/huggingface \
    -e HOT3D_REPO_DIR="${HOT3D_REPO_DIR}" \
    -e HOT3D_DATASET_DIR=/data/hot3d_dataset \
    -e HOT3D_GT_ROOT=/data/hot3d_gt \
    -e HOT3D_SPLIT_SUMMARY=/data/hot3d_gt/split_summary.json \
    -e STAGE1_DIR="${STAGE1_DIR}" \
    -v "${REPO_DIR}:/workspace/HaWoR" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${WEIGHTS_DIR}:/workspace/HaWoR/weights" \
    -v "${METRIC3D_DIR}:/workspace/HaWoR/thirdparty/Metric3D/weights" \
    -v "${MANO_ROOT}/models:/workspace/HaWoR/_DATA/data/mano:ro" \
    -v "${MANO_ROOT}/models:/workspace/HaWoR/_DATA/data_left/mano_left:ro" \
    -v "${HOT3D_REPO_DIR}:${HOT3D_REPO_DIR}:ro" \
    -v "${HOT3D_DATASET_DIR}:/data/hot3d_dataset:ro" \
    -v "${HOT3D_GT_ROOT}:/data/hot3d_gt:ro" \
    -v "${STAGE1_DIR}:${STAGE1_DIR}" \
    "$IMAGE_NAME" \
    /bin/bash
