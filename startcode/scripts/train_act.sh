#!/usr/bin/env bash
# Offline ACT behavior cloning. Does not start Isaac.
# Activate the policy conda env (torch + h5py + opencv) before running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

python3 -m starter.pipelines.act offline --config configs/act.yaml
