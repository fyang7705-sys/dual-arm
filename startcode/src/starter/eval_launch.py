"""Start our official-protocol policy server and RoboDojo's unmodified eval client."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from starter.config import load_config, parse_simple_yaml
from starter.paths import REPO_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config)
    for item in args.set:
        key, _, value = item.partition("=")
        config.update(parse_simple_yaml(f"{key}: {value}"))
    kind = config.get("kind", "act")
    if kind not in ("act", "diffusion") or config.get("action_type", "ee") != "ee":
        raise ValueError("Evaluation supports act/diffusion with ee actions")
    if not config.get("ckpt"):
        raise ValueError("Supply --set ckpt=/absolute/path/to/policy_last.ckpt")
    checkpoint = str(Path(config["ckpt"]).expanduser().resolve())
    if config.get("env_launch", "docker") not in ("docker", "native"):
        raise ValueError("Evaluation env_launch must be docker or native")
    if str(config.get("policy_gpu", 0)) == str(config.get("env_gpu", 1)) and not config.get("allow_shared_gpu", False):
        raise ValueError("Single-GPU evaluation requires --set allow_shared_gpu=true and a measured memory budget")
    from starter.policies.runtime import load_payload
    if load_payload(checkpoint)["kind"] != kind:
        raise ValueError("Checkpoint kind mismatch")
    root = Path(os.environ.get("ROBODOJO_ROOT", REPO_ROOT / "third_party/RoboDojo")).resolve()
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts/install_eval_adapters.py"),
                    "--robodojo-root", str(root)], check=True)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts/configure_sim.py"),
                    "--robodojo-root", str(root), "--num-envs", str(config.get("num_envs", 1))], check=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(config.get("policy_gpu", 0))
    command = [sys.executable, "-m", "starter.evaluation", "--kind", kind, "--checkpoint", checkpoint,
               "--port", str(port), "--device", str(config.get("device", "cuda:0")), "--seed", str(config.get("seed", 0))]
    server = subprocess.Popen(command, env=environment, start_new_session=True)
    name = f"dual-arm-eval-{os.getpid()}"
    try:
        deadline = time.monotonic() + float(config.get("server_timeout", 180))
        while True:
            if server.poll() is not None:
                raise RuntimeError("Policy server exited before readiness")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Policy server did not become ready")
                time.sleep(0.5)
        client = ["bash", str(root / "scripts/robodojo.sh"), "client", "--policy-name",
                  "StudentACT" if kind == "act" else "StudentDiffusion", "--task", str(config.get("task", "stack_blocks_random")),
                  "--policy-host", "127.0.0.1", "--policy-port", str(port), "--ckpt", Path(checkpoint).parent.name,
                  "--env-cfg", str(config.get("env_cfg", "arx_x5")), "--action-type", "ee",
                  "--seed", str(config.get("seed", 0)), "--eval-num", str(config.get("eval_num", "native")), "--env-gpu", "0"]
        sim_env = dict(os.environ)
        bypass = sim_env.get("NO_PROXY", sim_env.get("no_proxy", "")) + ",127.0.0.1,localhost"
        sim_env["NO_PROXY"] = bypass
        sim_env["no_proxy"] = bypass
        if str(config.get("eval_num", "native")) == "native":
            sim_env.pop("EVAL_NUM", None)
        sim_env["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "XPolicyLab"), str(root.parent)))
        sim_env["CUDA_VISIBLE_DEVICES"] = str(config.get("env_gpu", 0))
        if config.get("env_launch", "docker") == "native":
            python = os.environ.get("SIM_PYTHON") or config.get("sim_python")
            if not python:
                raise ValueError("Native eval needs SIM_PYTHON")
            # robodojo's shell invokes python; PATH selects the configured sim interpreter.
            sim_env["PATH"] = str(Path(python).parent) + os.pathsep + sim_env["PATH"]
            client[-1] = str(config.get("env_gpu", 0))
            subprocess.run(client, cwd=root, env=sim_env, check=True)
        else:
            docker = ["docker", "run", "--rm", "--name", name, "--gpus", f"device={config.get('env_gpu', 0)}",
                      "--network", "host", "--ipc", "host", "--workdir", str(root),
                      "-v", f"{REPO_ROOT}:{REPO_ROOT}", "-e", "CUDA_VISIBLE_DEVICES=0",
                      "-e", "ACCEPT_EULA=Y", "-e", "PRIVACY_CONSENT=Y", "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
                      "-e", f"PYTHONPATH={sim_env['PYTHONPATH']}"]
            docker += ["-e", f"NO_PROXY={bypass}", "-e", f"no_proxy={bypass}"]
            assets = (root / "Assets").resolve()
            docker += ["-v", f"{assets}:{assets}:ro"]
            if root != REPO_ROOT / "third_party/RoboDojo":
                docker += ["-v", f"{root}:{root}"]
            docker += [str(config.get("sim_image", "stack-blocks-sim:isaacsim5.1")), *client]
            subprocess.run(docker, check=True)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=20)
        if config.get("env_launch", "docker") == "docker":
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
