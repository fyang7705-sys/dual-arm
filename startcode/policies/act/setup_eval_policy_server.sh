#!/usr/bin/env bash
# RoboDojo calls this script to start the policy server on the host.
# Arguments, in order:
#   bench_name task_name ckpt_name env_cfg action_type seed policy_gpu policy_conda_env port host
set -euo pipefail

if [[ $# -lt 9 ]]; then
  echo "用法: setup_eval_policy_server.sh bench task ckpt env_cfg action_type seed policy_gpu conda_env port [host]" >&2
  exit 2
fi

bench_name="$1"
task_name="$2"
ckpt_name="$3"
env_cfg_type="$4"
action_type="$5"
seed="$6"
policy_gpu_id="$7"
policy_conda_env="$8"
policy_server_port="$9"
policy_server_host="${10:-127.0.0.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[ACT-SERVER] bench=${bench_name} task=${task_name} ckpt=${ckpt_name}"
echo "[ACT-SERVER] env=${env_cfg_type} action=${action_type} gpu=${policy_gpu_id}"
echo "[ACT-SERVER] listen=${policy_server_host}:${policy_server_port}"

# Activate the policy environment RoboDojo handed us.
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "${policy_conda_env}"; then
    conda activate "${policy_conda_env}"
  else
    echo "[ACT-SERVER] warning: conda env ${policy_conda_env} not found, keeping current env" >&2
  fi
fi

exec env \
  PYTHONWARNINGS=ignore::UserWarning \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  SEED="${seed}" \
  PYTHONPATH="${XPL_WS_PATH:-${HOME}/.local/xpl_ws}:${PYTHONPATH:-}" \
  python "${ROOT}/policies/eval_server.py" \
    --kind act \
    --ckpt "${ckpt_name}" \
    --env-cfg "${env_cfg_type}" \
    --action-type "${action_type}" \
    --host "${policy_server_host}" \
    --port "${policy_server_port}"

