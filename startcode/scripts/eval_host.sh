#!/usr/bin/env bash
# Evaluate a checkpoint on the host: Isaac Sim runs directly in the local conda
# environment, so no simulator Docker image is needed.
#
# This is an *additional* entry point. scripts/eval_policy.sh is left untouched;
# that one keeps using Docker.
#
# Usage:
#   conda activate robodojo
#   bash scripts/eval_host.sh [config] [options]
#
# Arguments:
#   config                config file, default configs/eval.yaml
#   --policy-dir PATH     override policy_dir (e.g. third_party/XPolicyLab/policy/DP)
#   --ckpt NAME           override ckpt label
#   --policy-env ENV      override the conda env used by the policy server
#   --sim-env ENV         conda env used by the Isaac client (default: robodojo)
#   --port PORT           policy server port (default: a free port)
#   --server-only         start the policy server and stay in the foreground
#   --client-only         skip starting a server, use --port / config/env var
#   --dry-run             print the resolved commands and exit
#   -h | --help           show this help
#
# Environment overrides: POLICY_DIR, CKPT_NAME, POLICY_ENV, SIM_ENV, POLICY_PORT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-${ROOT}/third_party/RoboDojo}"

CONFIG="${ROOT}/configs/eval_host.yaml"
[[ -f "${CONFIG}" ]] || CONFIG="${ROOT}/configs/eval.yaml"
POLICY_DIR_OVERRIDE=""
CLIENT_POLICY_DIR_OVERRIDE=""
EVAL_NUM_OVERRIDE=""
CKPT_OVERRIDE=""
POLICY_ENV_OVERRIDE=""
SIM_ENV_OVERRIDE=""
PORT_OVERRIDE=""
SERVER_ONLY=0
CLIENT_ONLY=0
DRY_RUN=0

usage() {
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-dir) POLICY_DIR_OVERRIDE="${2:?--policy-dir needs a value}"; shift 2 ;;
    --client-policy-dir) CLIENT_POLICY_DIR_OVERRIDE="${2:?--client-policy-dir needs a value}"; shift 2 ;;
    --eval-num) EVAL_NUM_OVERRIDE="${2:?--eval-num needs a value}"; shift 2 ;;
    --ckpt) CKPT_OVERRIDE="${2:?--ckpt needs a value}"; shift 2 ;;
    --policy-env) POLICY_ENV_OVERRIDE="${2:?--policy-env needs a value}"; shift 2 ;;
    --sim-env) SIM_ENV_OVERRIDE="${2:?--sim-env needs a value}"; shift 2 ;;
    --port) PORT_OVERRIDE="${2:?--port needs a value}"; shift 2 ;;
    --server-only) SERVER_ONLY=1; shift ;;
    --client-only) CLIENT_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "[eval_host] unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) CONFIG="$1"; shift ;;
  esac
done

if [[ ! -f "${CONFIG}" ]]; then
  echo "[eval_host] config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${ROBODOJO_ROOT}/scripts/robodojo.sh" ]]; then
  echo "[eval_host] RoboDojo not found at ${ROBODOJO_ROOT}." >&2
  echo "[eval_host] run scripts/bootstrap_third_party.sh first." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Read the course config (same keys as configs/eval.yaml).
# ---------------------------------------------------------------------------
FIELDS_FILE="$(mktemp)"
trap 'rm -f "${FIELDS_FILE}"' EXIT

CONFIG="${CONFIG}" ROOT="${ROOT}" \
POLICY_DIR="${POLICY_DIR_OVERRIDE}" \
CLIENT_POLICY_DIR="${CLIENT_POLICY_DIR_OVERRIDE}" \
EVAL_NUM="${EVAL_NUM_OVERRIDE}" \
CKPT="${CKPT_OVERRIDE}" \
POLICY_ENV="${POLICY_ENV_OVERRIDE}" \
PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" python3 - <<'PY' > "${FIELDS_FILE}"
import os
import shlex
from pathlib import Path

from starter.config import load_config, require

cfg = load_config(os.environ["CONFIG"])
root = Path(os.environ["ROOT"])

policy_dir = Path(os.environ.get("POLICY_DIR") or require(cfg, "policy_dir"))
if not policy_dir.is_absolute():
    policy_dir = root / policy_dir

client_policy_dir = Path(
    os.environ.get("CLIENT_POLICY_DIR") or cfg.get("client_policy_dir") or policy_dir
)
if not client_policy_dir.is_absolute():
    client_policy_dir = root / client_policy_dir

fields = {
    "POLICY_DIR": str(policy_dir),
    "CLIENT_POLICY_DIR": str(client_policy_dir),
    "CLIENT_POLICY_NAME": client_policy_dir.name,
    "POLICY_NAME": str(cfg.get("policy_name") or policy_dir.name),
    "TASK": require(cfg, "task"),
    "CKPT_NAME": os.environ.get("CKPT") or require(cfg, "ckpt"),
    "POLICY_ENV": os.environ.get("POLICY_ENV") or require(cfg, "policy_env"),
    "ENV_CFG": require(cfg, "env_cfg"),
    "ACTION_TYPE": require(cfg, "action_type"),
    "SEED": require(cfg, "seed"),
    "EVAL_NUM": os.environ.get("EVAL_NUM") or require(cfg, "eval_num"),
    "POLICY_GPU": require(cfg, "policy_gpu"),
    "ENV_GPU": require(cfg, "env_gpu"),
    "SIM_ENV": cfg.get("sim_env", "robodojo"),
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY

# shellcheck disable=SC1090
source "${FIELDS_FILE}"
rm -f "${FIELDS_FILE}"

POLICY_DIR="${POLICY_DIR}"
SIM_ENV="${SIM_ENV_OVERRIDE:-${SIM_ENV}}"
PORT="${PORT_OVERRIDE:-${POLICY_PORT:-}}"

if [[ "${POLICY_DIR}" != /* ]]; then
  POLICY_DIR="${ROOT}/${POLICY_DIR}"
fi

if [[ "${POLICY_GPU}" == "${ENV_GPU}" ]]; then
  echo "[eval_host] policy_gpu (${POLICY_GPU}) and env_gpu (${ENV_GPU}) must differ." >&2
  exit 1
fi
if [[ ! -f "${POLICY_DIR}/setup_eval_policy_server.sh" ]]; then
  echo "[eval_host] missing ${POLICY_DIR}/setup_eval_policy_server.sh" >&2
  exit 1
fi
if [[ ! -e "/dev/nvidia${ENV_GPU}" ]]; then
  echo "[eval_host] host has no /dev/nvidia${ENV_GPU}." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Isaac Sim / Vulkan environment, same values the Docker wrapper uses.
# ---------------------------------------------------------------------------
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

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=YES
export NVIDIA_DRIVER_CAPABILITIES=all
export VK_DRIVER_FILES="${ICD_PATH}"
export VK_ICD_FILENAMES="${ICD_PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
# `XPolicyLab.policy.<name>.deploy` lives in third_party/XPolicyLab, so the
# parent directory has to be importable by the Isaac client. The directory
# itself carries `client_server`, which the eval client imports directly.
# Kept as a plain variable on purpose: exporting it would leak RoboDojo's
# `utils` / `env` packages into the policy server and crash it natively.
# The pinned curobo is listed first: the editable install in this env comes
# from another checkout and breaks on the warp version Isaac Sim ships.
CLIENT_PYTHONPATH="${ROBODOJO_ROOT}/third_party/curobo:${ROOT}/third_party:${ROOT}/third_party/XPolicyLab:${ROBODOJO_ROOT}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

if [[ -z "${PORT}" ]]; then
  PORT="$(python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
fi

activate_env() {
  local name="$1"
  if ! command -v conda >/dev/null 2>&1; then
    echo "[eval_host] conda not found on PATH; activate ${name} yourself." >&2
    return 0
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "${name}"; then
    conda activate "${name}"
  else
    echo "[eval_host] warning: conda env ${name} not found; keeping $(python3 -V 2>&1)." >&2
  fi
}

# RoboDojo uses --ckpt as a path label; an absolute path would turn the result
# directory into a deep tree, so only the file name is forwarded to the client.
CLIENT_CKPT_LABEL="$(basename "${CKPT_NAME}")"

SERVER_CMD=(
  bash "${ROBODOJO_ROOT}/scripts/robodojo.sh" server
  --policy-dir "${POLICY_DIR}"
  --task "${TASK}"
  --ckpt "${CKPT_NAME}"
  --env-cfg "${ENV_CFG}"
  --action-type "${ACTION_TYPE}"
  --seed "${SEED}"
  --policy-gpu "${POLICY_GPU}"
  --policy-env "${POLICY_ENV}"
  --policy-port "${PORT}"
  --bind-host 127.0.0.1
)

CLIENT_CMD=(
  bash "${ROBODOJO_ROOT}/scripts/robodojo.sh" client
  --policy-dir "${CLIENT_POLICY_DIR}"
  --task "${TASK}"
  --policy-host 127.0.0.1
  --policy-port "${PORT}"
  --ckpt "${CLIENT_CKPT_LABEL:-${CKPT_NAME}}"
  --env-cfg "${ENV_CFG}"
  --action-type "${ACTION_TYPE}"
  --seed "${SEED}"
  --env-gpu "${ENV_GPU}"
  --eval-num "${EVAL_NUM}"
)

echo "[eval_host] config     = ${CONFIG}"
echo "[eval_host] policy_dir = ${POLICY_DIR} (name: ${POLICY_NAME})"
echo "[eval_host] client_dir = ${CLIENT_POLICY_DIR} (deploy module: XPolicyLab.policy.${CLIENT_POLICY_NAME}.deploy)"
echo "[eval_host] task       = ${TASK}  env_cfg=${ENV_CFG}  action_type=${ACTION_TYPE}"
echo "[eval_host] ckpt       = ${CKPT_NAME}  eval_num=${EVAL_NUM}"
echo "[eval_host] gpus       = policy ${POLICY_GPU} / sim ${ENV_GPU}"
echo "[eval_host] envs       = policy ${POLICY_ENV} / sim ${SIM_ENV}"
echo "[eval_host] port       = ${PORT}"
echo "[eval_host] run_id     = ${ROBODOJO_RUN_ID}"
printf '[eval_host] server: '; printf '%q ' "${SERVER_CMD[@]}"; echo
printf '[eval_host] client: '; printf '%q ' "${CLIENT_CMD[@]}"; echo

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[eval_host] dry-run, nothing started."
  exit 0
fi

cd "${ROOT}"

if [[ "${SERVER_ONLY}" == "1" ]]; then
  activate_env "${POLICY_ENV}"
  echo "[eval_host] starting policy server only; Ctrl-C to stop."
  exec env -u PYTHONPATH "${SERVER_CMD[@]}"
fi

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${CLIENT_ONLY}" != "1" ]]; then
  activate_env "${POLICY_ENV}"
  echo "[eval_host] starting policy server..."
  set -m
  env -u PYTHONPATH "${SERVER_CMD[@]}" &
  SERVER_PID=$!
  set +m

  ready=0
  for _ in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" || true
      echo "[eval_host] policy server exited before binding ${PORT}." >&2
      exit 1
    fi
    if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    echo "[eval_host] policy server did not listen on ${PORT} within 120s." >&2
    exit 1
  fi
  echo "[eval_host] policy server is up on 127.0.0.1:${PORT}"
fi

activate_env "${SIM_ENV}"
echo "[eval_host] running the Isaac client on host GPU ${ENV_GPU}..."
PYTHONPATH="${CLIENT_PYTHONPATH}" "${CLIENT_CMD[@]}"
client_rc=$?

# RoboDojo nests results as eval_result/<bench>/<task>/<policy>/<env_cfg>/...
# keyed by the run id, so locate the directory instead of assuming a depth.
RESULT_DIR="$(find "${ROBODOJO_ROOT}/eval_result" -type d -name "${ROBODOJO_RUN_ID}" 2>/dev/null | head -n 1)"
echo "[eval_host] client exit code: ${client_rc}"
if [[ -n "${RESULT_DIR}" && -d "${RESULT_DIR}" ]]; then
  echo "[eval_host] results: ${RESULT_DIR}"
  find "${RESULT_DIR}" -maxdepth 1 \( -name '*.json' -o -name '*.mp4' \) | sort | head -20
else
  echo "[eval_host] no result directory named ${ROBODOJO_RUN_ID} under ${ROBODOJO_ROOT}/eval_result" >&2
fi
exit "${client_rc}"
