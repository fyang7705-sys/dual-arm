#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src:$ROOT/third_party:$ROOT/third_party/XPolicyLab:${PYTHONPATH:-}"
cd "$ROOT"
python3 -m starter.launch diffusion online "$@"
