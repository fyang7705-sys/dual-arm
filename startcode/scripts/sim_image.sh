# Resolve the simulator image declared in configs/sim_image.env.
# shellcheck disable=SC1091
if [[ -z "${ROOT:-}" ]]; then
  echo "sim_image.sh must be sourced after ROOT is set." >&2
  return 1 2>/dev/null || exit 1
fi
set -a
# shellcheck disable=SC1091
source "${ROOT}/configs/sim_image.env"
set +a

resolve_sim_image() {
  if [[ -z "${SIM_IMAGE:-}" ]]; then
    echo "configs/sim_image.env 没有 SIM_IMAGE。" >&2
    return 1
  fi
  if docker image inspect "localhost/${SIM_IMAGE}" >/dev/null 2>&1; then
    IMAGE="localhost/${SIM_IMAGE}"
  elif docker image inspect "${SIM_IMAGE}" >/dev/null 2>&1; then
    IMAGE="${SIM_IMAGE}"
  else
    echo "本机没有仿真镜像 ${SIM_IMAGE}。" >&2
    echo "从 ${SIM_IMAGE_URL} 下载 tar 包，然后 docker load -i stack-blocks-sim_isaacsim5.1.tar。" >&2
    echo "这不是镜像仓库地址，不要 docker pull。" >&2
    return 1
  fi
}
