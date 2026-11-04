#!/usr/bin/env bash
# Restart the Isaac env server until STOP_FILE appears.
# Usage: run_online_env.sh <gpu> <port> <task> <env_cfg> <num_envs> <once:0|1> <stop_file> <container_name>
set -euo pipefail

GPU="$1"
PORT="$2"
TASK="$3"
ENV_CFG="$4"
NUM_ENVS="$5"
ONCE="$6"
STOP_FILE="$7"
NAME="$8"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/sim_image.sh"
resolve_sim_image

ONCE_ARG=()
if [[ "${ONCE}" == "1" ]]; then
  ONCE_ARG=(--once)
fi

KIT_ARGS="--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
if [[ ! -e "/dev/nvidia${GPU}" ]]; then
  echo "Host GPU device /dev/nvidia${GPU} is not available." >&2
  exit 1
fi
# This host's container runtime ignores --gpus, so the simulator only sees a
# GPU when its device node and driver libraries are mounted. Only this card is
# mounted, and the process uses it as cuda:0.
ICD_PATH="${ROOT}/outputs/eval/nvidia_icd.json"
mkdir -p "${ROOT}/outputs/eval"
cat > "${ICD_PATH}" <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
        "api_version" : "1.3.277"
    }
}
EOF
maps=()
shopt -s nullglob
for lib in \
  /usr/lib/x86_64-linux-gnu/libcuda.so* \
  /usr/lib/x86_64-linux-gnu/libnvidia*.so* \
  /usr/lib/x86_64-linux-gnu/libEGL_nvidia* \
  /usr/lib/x86_64-linux-gnu/libGLX_nvidia* \
  /usr/lib/x86_64-linux-gnu/libGLESv*_nvidia* \
  /usr/lib/x86_64-linux-gnu/libnvcuvid.so* \
  /usr/lib/x86_64-linux-gnu/libnvoptix.so*
do
  [[ -e "${lib}" ]] || continue
  maps+=(-v "${lib}:${lib}:ro")
done
shopt -u nullglob
while [[ ! -f "${STOP_FILE}" ]]; do
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  echo "[env] starting ${TASK} on host GPU ${GPU}, port ${PORT}, num_envs ${NUM_ENVS}"
  docker run --rm --name "${NAME}" \
    --device "/dev/nvidia${GPU}" \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    --device /dev/nvidia-modeset \
    --device /dev/nvidia-uvm-tools \
    --device /dev/dri \
    "${maps[@]}" \
    --network host --ipc host \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e "VK_DRIVER_FILES=${ICD_PATH}" \
    -e "VK_ICD_FILENAMES=${ICD_PATH}" \
    -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e STACK_BLOCKS_ROOT=/stack-blocks \
    -e PYTHONPATH=/stack-blocks/src:/workspace/RoboDojo \
    -v "${ROOT}/third_party/RoboDojo/Assets:/workspace/RoboDojo/Assets:ro" \
    -v "${ROOT}/data:/workspace/RoboDojo/data" \
    -v "${ROOT}:/stack-blocks" \
    -v "${ROOT}:${ROOT}" \
    -v "${ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro" \
    "${IMAGE}" \
    python -u /stack-blocks/scripts/isaac_env_server.py \
      --task "${TASK}" \
      --env-cfg "${ENV_CFG}" \
      --port "${PORT}" \
      --host 127.0.0.1 \
      --num-envs "${NUM_ENVS}" \
      --headless \
      --enable_cameras \
      --device cuda:0 \
      --kit_args "${KIT_ARGS}" \
      "${ONCE_ARG[@]}" || true
  if [[ -f "${STOP_FILE}" ]]; then
    break
  fi
  echo "[env] server exited; starting a fresh process" >&2
  sleep 2
done
