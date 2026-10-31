#!/usr/bin/env bash
# RoboDojo calls this script to start the policy server on the host.
# Arguments, in order:
#   bench_name task_name ckpt_name env_cfg action_type seed policy_gpu policy_conda_env port host
set -euo pipefail

if [[ $# -lt 9 ]]; then
  echo "用法: setup_eval_policy_server.sh bench task ckpt env_cfg action_type seed policy_gpu conda_env port [host]" >&2
  exit 2
fi

echo "ACT 评测服务还没有实现。" >&2
echo "在你自己的环境 ${8} 里加载 checkpoint ${3}，监听 ${10:-127.0.0.1}:${9}，并按 16 维 ee 响应观测。" >&2
exit 1
