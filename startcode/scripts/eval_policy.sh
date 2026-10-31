#!/usr/bin/env bash
# Evaluate one student checkpoint.
# The policy server runs in the conda env named by configs/eval.yaml.
# The Isaac client runs in the simulator image.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-${ROOT}/third_party/RoboDojo}"
CONFIG="${1:-${ROOT}/configs/eval.yaml}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export STACK_BLOCKS_CONFIG="${CONFIG}"
cd "${ROOT}"

if [[ ! -f "${ROBODOJO_ROOT}/scripts/robodojo.sh" ]]; then
  echo "RoboDojo 不在 ${ROBODOJO_ROOT}。先运行 scripts/bootstrap_third_party.sh。" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/sim_image.sh"
resolve_sim_image

eval "$(python3 - <<'PY'
import os
import shlex
from pathlib import Path
from starter.config import load_config, require

cfg = load_config(os.environ["STACK_BLOCKS_CONFIG"])
policy_dir = Path(require(cfg, "policy_dir"))
if not policy_dir.is_absolute():
    policy_dir = Path.cwd() / policy_dir
fields = {
    "POLICY_DIR": str(policy_dir),
    "TASK": require(cfg, "task"),
    "CKPT_NAME": require(cfg, "ckpt"),
    "POLICY_ENV": require(cfg, "policy_env"),
    "ENV_CFG": require(cfg, "env_cfg"),
    "ACTION_TYPE": require(cfg, "action_type"),
    "SEED": require(cfg, "seed"),
    "EVAL_NUM": require(cfg, "eval_num"),
    "POLICY_GPU": require(cfg, "policy_gpu"),
    "ENV_GPU": require(cfg, "env_gpu"),
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

if [[ "${POLICY_GPU}" == "${ENV_GPU}" ]]; then
  echo "策略 GPU ${POLICY_GPU} 和 Isaac GPU ${ENV_GPU} 必须不同。" >&2
  exit 1
fi
if [[ ! -x "${POLICY_DIR}/setup_eval_policy_server.sh" ]]; then
  echo "缺少可执行的 ${POLICY_DIR}/setup_eval_policy_server.sh" >&2
  exit 1
fi
if [[ ! -e "/dev/nvidia${ENV_GPU}" ]]; then
  echo "主机上没有 /dev/nvidia${ENV_GPU}。" >&2
  exit 1
fi

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
export ROBODOJO_RUN_ID="${RUN_ID}"
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

PORT="$(python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"

echo "[eval] policy server in conda env ${POLICY_ENV} on GPU ${POLICY_GPU}"
echo "[eval] Isaac client in ${IMAGE} on GPU ${ENV_GPU}, port ${PORT}"

set -m
bash "${POLICY_DIR}/setup_eval_policy_server.sh" \
  RoboDojo "${TASK}" "${CKPT_NAME}" "${ENV_CFG}" "${ACTION_TYPE}" \
  "${SEED}" "${POLICY_GPU}" "${POLICY_ENV}" "${PORT}" 127.0.0.1 &
SERVER_PID=$!
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  docker rm -f stack-blocks-eval >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || true
    echo "[eval] 策略服务已退出，没有启动仿真镜像。" >&2
    exit 1
  fi
  if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  echo "[eval] 策略服务在 60 秒内没有监听 ${PORT}。" >&2
  exit 1
fi

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

docker rm -f stack-blocks-eval >/dev/null 2>&1 || true
docker run --rm --name stack-blocks-eval \
  --device "/dev/nvidia${ENV_GPU}" \
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
  -e "ROBODOJO_ROOT=${ROBODOJO_ROOT}" \
  -e "ROBODOJO_RUN_ID=${RUN_ID}" \
  -v "${ROOT}:${ROOT}" \
  -v "${ROBODOJO_ROOT}/Assets:/workspace/RoboDojo/Assets:ro" \
  -v "${ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro" \
  "${IMAGE}" \
  bash "${ROBODOJO_ROOT}/scripts/robodojo.sh" client \
    --policy-dir "${POLICY_DIR}" \
    --task "${TASK}" \
    --policy-host 127.0.0.1 \
    --policy-port "${PORT}" \
    --ckpt "${CKPT_NAME}" \
    --env-cfg "${ENV_CFG}" \
    --action-type "${ACTION_TYPE}" \
    --seed "${SEED}" \
    --env-gpu 0 \
    --eval-num "${EVAL_NUM}"
