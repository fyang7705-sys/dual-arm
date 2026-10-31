#!/usr/bin/env bash
# RoboDojo calls this script to start the policy server on the host.
# Arguments, in order:
#   bench_name task_name ckpt_name env_cfg action_type seed policy_gpu policy_conda_env port host
set -euo pipefail

if [[ $# -lt 9 ]]; then
  echo "用法: setup_eval_policy_server.sh bench task ckpt env_cfg action_type seed policy_gpu conda_env port [host]" >&2
  exit 2
fi

BENCH="$1"
TASK="$2"
CKPT="$3"
ENV_CFG="$4"
ACTION_TYPE="$5"
SEED="$6"
POLICY_GPU="$7"
CONDA_ENV="$8"
PORT="$9"
HOST="${10:-127.0.0.1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export STACK_BLOCKS_ROOT="${ROOT}"
export CUDA_VISIBLE_DEVICES="${POLICY_GPU}"
export TOKENIZERS_PARALLELISM=false

if [[ -n "${CONDA_ENV}" && "${CONDA_ENV}" != "none" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[serve-dp] conda is not on PATH; cannot activate ${CONDA_ENV}." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

echo "[serve-dp] bench=${BENCH} task=${TASK} ckpt=${CKPT} gpu=${POLICY_GPU} env=${CONDA_ENV} port=${PORT}"

exec python "${ROOT}/policies/diffusion/serve_policy.py" \
  --ckpt "${CKPT}" \
  --task "${TASK}" \
  --env-cfg "${ENV_CFG}" \
  --action-type "${ACTION_TYPE}" \
  --seed "${SEED}" \
  --port "${PORT}" \
  --host "${HOST}"
