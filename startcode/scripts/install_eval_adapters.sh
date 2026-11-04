#!/usr/bin/env bash
# Link policies/<name> into the XPolicyLab checkout so RoboDojo can import
# XPolicyLab.policy.<name>.deploy. Run once after bootstrap_third_party.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XPL="${ROOT}/third_party/XPolicyLab"
ROBODOJO_XPL="${ROOT}/third_party/RoboDojo/XPolicyLab"

if [[ -d "${ROBODOJO_XPL}" && -z "$(ls -A "${ROBODOJO_XPL}" 2>/dev/null)" ]]; then
  rmdir "${ROBODOJO_XPL}"
fi
if [[ ! -e "${ROBODOJO_XPL}" && ! -L "${ROBODOJO_XPL}" && -d "${XPL}" ]]; then
  ln -s ../XPolicyLab "${ROBODOJO_XPL}"
  echo "[adapters] linked RoboDojo/XPolicyLab -> third_party/XPolicyLab"
fi

for name in diffusion act; do
  for base in "${XPL}" "${ROBODOJO_XPL}"; do
    [[ -d "${base}" ]] || continue
    mkdir -p "${base}/policy"
    ln -sfn "${ROOT}/policies/${name}" "${base}/policy/${name}"
    echo "[adapters] ${base}/policy/${name} -> ${ROOT}/policies/${name}"
  done
done

if [[ ! -d "${XPL}" ]]; then
  echo "[adapters] ${XPL} is missing; run scripts/bootstrap_third_party.sh first." >&2
  exit 1
fi
