#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 9 || "$5" != ee ]]; then
  echo "Usage: $0 bench task checkpoint env_cfg ee seed gpu conda_env port [host]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$ROOT/src:$ROOT/third_party:$ROOT/third_party/XPolicyLab:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$7"
cd "$ROOT"
exec conda run --no-capture-output -n "$8" python -m starter.evaluation \
  --kind diffusion --checkpoint "$3" --seed "$6" --port "$9" --host "${10:-127.0.0.1}" --device cuda:0
