#!/usr/bin/env bash
# Check out the pinned RoboDojo and XPolicyLab commits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="${ROOT}/third_party/pins.lock"

pin_of() {
  awk -v name="$1" '$1 == name { print $2 }' "${LOCK}"
}

clone_pin() {
  local name="$1"
  local url="$2"
  local dest="${ROOT}/third_party/${name}"
  local pin
  pin="$(pin_of "${name}")"
  if [[ -z "${pin}" ]]; then
    echo "missing pin for ${name}" >&2
    exit 1
  fi
  if [[ ! -e "${dest}/.git" ]]; then
    git clone --depth 1 "${url}" "${dest}"
  fi
  if [[ -n "$(git -C "${dest}" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to overwrite local dependency edits: ${dest}" >&2
    exit 1
  fi
  git -C "${dest}" fetch --depth 1 origin "${pin}"
  git -C "${dest}" checkout "${pin}"
  echo "${name} ${pin}"
}

clone_pin RoboDojo https://github.com/RoboDojo-Benchmark/RoboDojo.git
clone_pin XPolicyLab https://github.com/XPolicyLab/XPolicyLab.git

mkdir -p "${ROOT}/third_party"
ln -sfn "${ROOT}/data" "${ROOT}/third_party/data"
ln -sfn "${ROOT}/third_party/RoboDojo/env_cfg" "${ROOT}/third_party/env_cfg"
bash "${ROOT}/scripts/install_eval_adapters.sh"
echo "Submodules are pinned. RoboDojo's Isaac submodules are not initialized."
echo "Build the simulator image from third_party/RoboDojo after: git submodule update --init --recursive"
