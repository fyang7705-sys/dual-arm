#!/usr/bin/env bash
# Download only the standard stack_blocks demonstrations.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL="${ROOT}/data/hf_robodojo"
mkdir -p "${LOCAL}"

if command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli)
elif command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf)
else
  echo "Neither huggingface-cli nor hf is on PATH. Install huggingface_hub, then re-run this script." >&2
  echo "The dataset is RoboDojo-Benchmark/RoboDojo, path data/RoboDojo/stack_blocks/arx_x5." >&2
  exit 1
fi

"${HF_CLI[@]}" download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset \
  --include "data/RoboDojo/stack_blocks/arx_x5/*" \
  --include "data/RoboDojo/stack_blocks/arx_x5/**" \
  --local-dir "${LOCAL}"

mkdir -p "${ROOT}/data/RoboDojo/stack_blocks"
ln -sfn "${LOCAL}/data/RoboDojo/stack_blocks/arx_x5" "${ROOT}/data/RoboDojo/stack_blocks/arx_x5"
ln -sfn "${ROOT}/data" "${ROOT}/third_party/data"
echo "stack_blocks demos linked at data/RoboDojo/stack_blocks/arx_x5"
echo "Eval uses stack_blocks, the same task as these demonstrations."
