"""Install self-contained RoboDojo eval bridges for this project's policies.

The pinned RoboDojo launcher expects a policy directory with
``setup_eval_policy_server.sh``, ``setup_eval_env_client.sh``, ``deploy.py``
and ``deploy.yml``.  This project keeps its real policy server in
``policies/<kind>/setup_eval_policy_server.sh``; the generated adapters only
forward arguments, so there is one implementation of checkpoint loading and
one implementation of the evaluation loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def install(root) -> Path:
    root = Path(root).expanduser().resolve()
    if (root / "XPolicyLab").is_dir():
        xpl = root / "XPolicyLab"
    elif root.name == "XPolicyLab" and (root / "policy").is_dir():
        xpl = root
        root = xpl.parent
    else:
        raise SystemExit(
            f"XPolicyLab not found under {root}. Run scripts/bootstrap_third_party.sh first."
        )

    for name, kind in (("StudentACT", "act"), ("StudentDiffusion", "diffusion")):
        directory = xpl / "policy" / name
        directory.mkdir(parents=True, exist_ok=True)
        _write(directory / "__init__.py", "")
        shutil.copyfile(ROOT / "src/starter/eval_deploy.py", directory / "deploy.py")
        _write(
            directory / "deploy.yml",
            f"""policy_name: {name}
protocol: ws
host: localhost
port: null
bench_name: RoboDojo
task_name: null
ckpt_name: null
env_cfg_type: arx_x5
action_type: ee
seed: 0
eval_batch: false
""",
        )
        _write(
            directory / "setup_eval_policy_server.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"exec bash {shlex.quote(str(ROOT / 'policies' / kind / 'setup_eval_policy_server.sh'))} \"$@\"\n",
            executable=True,
        )
        # Arguments from RoboDojo's run_policy_eval.sh:
        #   bench task ckpt env_cfg action seed env_gpu eval_env extra port [host]
        client = f"""#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 10 ]]; then
  echo "Expected bench task ckpt env_cfg action seed env_gpu eval_env extra port [host]" >&2
  exit 2
fi
ROBODOJO_ROOT={shlex.quote(str(root))}
export ROBODOJO_ROOT
export NO_PROXY="${{NO_PROXY:-}},127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
python3 {shlex.quote(str(ROOT / 'scripts/configure_sim.py'))} \\
  --robodojo-root "$ROBODOJO_ROOT" --num-envs "${{SIM_NUM_ENVS:-1}}"
CLIENT=(bash "$ROBODOJO_ROOT/scripts/robodojo.sh" client
  --policy-name {name}
  --task "$2"
  --ckpt "$3"
  --env-cfg "$4"
  --action-type "$5"
  --seed "$6"
  --env-gpu "$7"
  --policy-port "${{10}}"
  --policy-host "${{11:-127.0.0.1}}")
if [[ -n "${{8:-}}" ]] && command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -n "$8" "${{CLIENT[@]}}"
fi
exec "${{CLIENT[@]}}"
"""
        _write(directory / "setup_eval_env_client.sh", client, executable=True)
        print(f"[adapters] {directory} -> {kind}")

    print(f"[adapters] installed StudentACT / StudentDiffusion in {xpl / 'policy'}")
    return xpl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", default=str(ROOT / "third_party/RoboDojo"))
    install(parser.parse_args().robodojo_root)


if __name__ == "__main__":
    main()
