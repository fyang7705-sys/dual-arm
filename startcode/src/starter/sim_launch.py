"""Standalone simulator server for a learner launched in external mode."""
import os
import subprocess
import sys

from starter.cli import arguments
from starter.launch import simulator_command


def main():
    _, config = arguments("simulator", ["online", *sys.argv[1:]])
    command, environment, cwd = simulator_command(config)
    try:
        subprocess.run(command, env=environment, cwd=cwd, check=True)
    finally:
        if config.get("env_launch", "docker") == "docker":
            subprocess.run(["docker", "rm", "-f", f"dual-arm-env-{os.getpid()}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
