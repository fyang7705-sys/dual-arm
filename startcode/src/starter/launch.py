"""Launch a learner and an optional native/Docker simulator with shared config."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from starter.cli import arguments
from starter.paths import REPO_ROOT


def simulator_command(config):
    root = Path(os.environ.get("ROBODOJO_ROOT", REPO_ROOT / "third_party/RoboDojo")).resolve()
    if not (root / "task/RoboDojo").is_dir() or not (root / "Assets").is_dir():
        raise FileNotFoundError(f"Missing RoboDojo source or Assets under {root}")
    gpu = str(config.get("env_gpu", 0))
    simulator = [str(REPO_ROOT / "scripts/isaac_env_server.py"), "--task", str(config["rollout_task"]),
                 "--env-cfg", str(config["env_cfg"]), "--num-envs", str(config.get("num_envs", 1)),
                 "--port", str(config.get("env_port", 5555)), "--host", str(config.get("env_host", "127.0.0.1")),
                 "--seed", str(config.get("seed", 0)), "--layout-source", "random", "--device", "cuda:0",
                 "--headless", "--enable_cameras", "--kit_args",
                 "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"]
    if config.get("domain_randomization", False):
        simulator.append("--domain-randomization")
    environment = dict(os.environ)
    environment.update({"STACK_BLOCKS_ROOT": str(REPO_ROOT), "CUDA_VISIBLE_DEVICES": gpu,
                        "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "src"), str(root),
                                                       str(root / "XPolicyLab"), str(root.parent)))})
    mode = config.get("env_launch", "docker")
    if mode == "native":
        python = os.environ.get("SIM_PYTHON") or str(config.get("sim_python", ""))
        if not python or not Path(python).is_file():
            raise ValueError("Native mode needs SIM_PYTHON pointing to the configured RoboDojo Python interpreter")
        return [python, "-u", *simulator], environment, root
    if mode != "docker":
        raise ValueError("env_launch must be docker, native or external")
    image = str(config.get("sim_image", "stack-blocks-sim:isaacsim5.1"))
    subprocess.run(["docker", "image", "inspect", image], check=True, stdout=subprocess.DEVNULL)
    command = ["docker", "run", "--rm", "--name", f"dual-arm-env-{os.getpid()}",
               "--gpus", f"device={gpu}", "--network", "host", "--ipc", "host",
               "--workdir", str(root)]
    for key in ("STACK_BLOCKS_ROOT", "PYTHONPATH"):
        command += ["-e", f"{key}={environment[key]}"]
    for value in ("CUDA_VISIBLE_DEVICES=0", "ACCEPT_EULA=Y", "PRIVACY_CONSENT=Y", "NVIDIA_DRIVER_CAPABILITIES=all"):
        command += ["-e", value]
    command += ["-v", f"{REPO_ROOT}:{REPO_ROOT}"]
    if root != REPO_ROOT / "third_party/RoboDojo":
        command += ["-v", f"{root}:{root}"]
    # Resolve symlinked assets at the same absolute path, as cuRobo embeds it.
    assets = (root / "Assets").resolve()
    command += ["-v", f"{assets}:{assets}:ro"]
    command += [image, "python", "-u", *simulator]
    return command, environment, root


def main():
    kind = sys.argv[1]
    if kind not in ("act", "diffusion"):
        raise SystemExit("First argument must be act or diffusion")
    stage, config = arguments(kind, sys.argv[2:])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("policy_gpu", 0))
    process = None
    log = None
    container = f"dual-arm-env-{os.getpid()}"
    try:
        if stage == "online":
            if config.get("env_backend", "remote") not in ("remote", "fake"):
                raise ValueError("env_backend must be remote or fake")
            # Validate the checkpoint before starting any expensive simulation.
            from starter.policies.runtime import load_payload
            source = config.get("resume") or config.get("offline_checkpoint")
            if not source:
                raise ValueError("Set offline_checkpoint or resume before launching simulation")
            if load_payload(source)["kind"] != kind:
                raise ValueError("Checkpoint kind mismatch")
            mode = config.get("env_launch", "docker")
            if config.get("env_backend", "remote") != "fake" and mode != "external":
                same = str(config.get("policy_gpu", 0)) == str(config.get("env_gpu", 1))
                if same and not config.get("allow_shared_gpu", False):
                    raise ValueError("GPU sharing requires allow_shared_gpu: true and a measured memory budget")
                command, environment, cwd = simulator_command(config)
                log_dir = REPO_ROOT / "outputs" / "simulator"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{kind}-{os.getpid()}.log"
                log = log_path.open("w", encoding="utf-8")
                print(f"Simulator log: {log_path}", flush=True)
                process = subprocess.Popen(command, env=environment, cwd=cwd, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                # Fail quickly for startup errors, then RemoteEnv handles slow scene creation.
                time.sleep(1)
                if process.poll() is not None:
                    raise RuntimeError(f"Simulator exited early; inspect {log_path}")
            from starter.policies.online import train
        else:
            from starter.policies.offline import train
        train(kind, config)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process is not None and config.get("env_launch", "docker") == "docker":
            subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
