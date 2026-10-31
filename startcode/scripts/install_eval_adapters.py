"""Install self-contained deploy bridges into the pinned, ignored XPolicyLab."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import shlex

ROOT = Path(__file__).resolve().parents[1]


def install(root):
    root = Path(root)
    for name in ("StudentACT", "StudentDiffusion"):
        kind = "act" if name == "StudentACT" else "diffusion"
        directory = root / "XPolicyLab" / "policy" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
        shutil.copyfile(ROOT / "src/starter/eval_deploy.py", directory / "deploy.py")
        (directory / "model.py").write_text("from starter.evaluation import Model\n", encoding="utf-8")
        (directory / "deploy.yml").write_text(
            f"policy_name: {name}\nprotocol: ws\nhost: localhost\nport: null\nbench_name: RoboDojo\n"
            "task_name: null\nckpt_name: null\nenv_cfg_type: arx_x5\naction_type: ee\nseed: 0\neval_batch: false\n",
            encoding="utf-8")
        server = "#!/usr/bin/env bash\nset -euo pipefail\nexec bash " + shlex.quote(
            str(ROOT / "policies" / kind / "setup_eval_policy_server.sh")) + ' "$@"\n'
        (directory / "setup_eval_policy_server.sh").write_text(server, encoding="utf-8")
        # Native official robodojo.sh eval integration. The course wrapper
        # remains the recommended Docker entry point.
        header = ("#!/usr/bin/env bash\nset -euo pipefail\nexport ROBODOJO_ROOT=" + shlex.quote(str(root.resolve())) +
                  "\nexport PYTHONPATH=" + shlex.quote(str(ROOT / "src")) + ':"${PYTHONPATH:-}"\n' +
                  'export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"\nexport no_proxy="$NO_PROXY"\ncd ' +
                  shlex.quote(str(ROOT)) + "\n")
        evaluator = header + '''if [[ $# -ne 10 || "$5" != ee ]]; then
  echo "Expected bench task ckpt env_cfg ee seed policy_gpu env_gpu policy_env sim_env" >&2
  exit 2
fi
export SIM_PYTHON="$(conda run --no-capture-output -n "${10}" python -c 'import sys;print(sys.executable)')"
exec conda run --no-capture-output -n "$9" python -m starter.eval_launch \\
  --set kind=__KIND__ --set "task=$2" --set "ckpt=$3" --set "env_cfg=$4" \\
  --set "seed=$6" --set "policy_gpu=$7" --set "env_gpu=$8" --set env_launch=native \\
  --set "eval_num=${EVAL_NUM:-native}" --set "allow_shared_gpu=${ALLOW_SHARED_GPU:-false}"
'''
        (directory / "eval.sh").write_text(evaluator.replace("__KIND__", kind), encoding="utf-8")
        client = header + '''if [[ $# -lt 10 ]]; then
  echo "Expected bench task ckpt env_cfg action seed gpu sim_env extra port [host]" >&2
  exit 2
fi
python3 scripts/configure_sim.py --robodojo-root "$ROBODOJO_ROOT" --num-envs "${SIM_NUM_ENVS:-1}"
exec conda run --no-capture-output -n "$8" bash "$ROBODOJO_ROOT/scripts/robodojo.sh" client \\
  --policy-name __NAME__ --task "$2" --ckpt "$3" --env-cfg "$4" --action-type "$5" \\
  --seed "$6" --env-gpu "$7" --policy-port "${10}" --policy-host "${11:-127.0.0.1}"
'''
        (directory / "setup_eval_env_client.sh").write_text(client.replace("__NAME__", name), encoding="utf-8")
        for script in directory.glob("*.sh"):
            script.chmod(0o755)
    print(f"Installed StudentACT / StudentDiffusion in {root / 'XPolicyLab/policy'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robodojo-root", default=str(ROOT / "third_party/RoboDojo"))
    install(parser.parse_args().robodojo_root)
