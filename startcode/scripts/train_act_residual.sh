#!/usr/bin/env bash
# Online ACT residual training. Isaac runs in the simulator image.
# Activate your own policy environment before running this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

mapfile -t META < <(python3 -c '
from starter.config import load_config, require
cfg = load_config("configs/env.yaml")
print(require(cfg, "policy_gpu"))
print(require(cfg, "env_gpu"))
print(require(cfg, "env_port"))
print(require(cfg, "rollout_task"))
print(require(cfg, "env_cfg"))
print(require(cfg, "num_envs"))
')

if [[ "${META[0]}" == "${META[1]}" ]]; then
  echo "策略 GPU ${META[0]} 和 Isaac GPU ${META[1]} 必须不同。" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${META[0]}"
echo "[act] policy GPU ${META[0]}, Isaac GPU ${META[1]}, image from configs/sim_image.env"
STOP="${ROOT}/outputs/act/.stop"
rm -f "${STOP}"
mkdir -p "${ROOT}/outputs/act"
bash "${ROOT}/scripts/run_online_env.sh" \
  "${META[1]}" "${META[2]}" "${META[3]}" "${META[4]}" "${META[5]}" 0 "${STOP}" stack-blocks-act-env &
ENV_PID=$!
cleanup() {
  touch "${STOP}"
  docker rm -f stack-blocks-act-env >/dev/null 2>&1 || true
  kill "${ENV_PID}" >/dev/null 2>&1 || true
  wait "${ENV_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python3 -m starter.pipelines.act online --config configs/env.yaml
