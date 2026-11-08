#!/usr/bin/env bash
# Offline Diffusion Policy. Does not start Isaac.
# Activate the policy conda env (torch + h5py + opencv) before running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

python3 -m starter.pipelines.diffusion offline --config configs/dp.yaml
