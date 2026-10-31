"""Shared preprocessing, checkpoint format, inference, and run artifacts."""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from starter.data.demos import resize_rgb
from starter.interfaces import ACTION_DIM, pack_ee_action
from starter.paths import REPO_ROOT
from starter.policies.models import ACT, Diffusion, ResidualActor

FORMAT = "dual-arm.policy.v1"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    if state:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def pick_device(config):
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; device: cpu is supported for smoke tests")
    return device


def model_config(config):
    cfg = {"width": 128, "chunk": 10, "latent_dim": 32, "heads": 4, "layers": 2,
           "diffusion_steps": 20, "image_size": 96, "execute_steps": 4}
    cfg.update({key: config[key] for key in cfg if key in config})
    if int(cfg["width"]) % int(cfg["heads"]) or int(cfg["width"]) % 2:
        raise ValueError("width must be even and divisible by heads")
    if not 1 <= int(cfg["execute_steps"]) <= int(cfg["chunk"]):
        raise ValueError("execute_steps must be between 1 and chunk")
    return cfg


def build_model(kind, config):
    if kind not in ("act", "diffusion"):
        raise ValueError(f"Unknown policy kind {kind}")
    return ACT(config) if kind == "act" else Diffusion(config)


def run_directory(config, kind, stage):
    root = Path(config.get("output_root", REPO_ROOT / "outputs")).expanduser()
    stamp = str(config.get("run_stamp") or datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f"))
    directory = root / kind / stage / f"seed{int(config.get('seed', 0))}" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return directory


def save_checkpoint(path, payload):
    """Atomic replacement and format/tensor checks; no file digest checks."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def checkpoint_path(path):
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "policy_last.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}; supply a file or completed run directory")
    return path


def load_payload(path, device="cpu"):
    # Only load project checkpoints received from trusted collaborators.
    payload = torch.load(checkpoint_path(path), map_location=device, weights_only=False)
    if payload.get("format") != FORMAT or payload.get("action_dim") != ACTION_DIM:
        raise ValueError("Incompatible checkpoint format or action dimension")
    return payload


def append_metrics(directory, metrics):
    with (Path(directory) / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def observation_arrays(observations, stats, size):
    images, states = [], []
    for observation in observations:
        if "image" in observation:
            image, state = observation["image"], observation["proprio"]
        else:
            image = observation["vision"]["cam_head"]["color"]
            state = pack_ee_action(observation["state"])
        images.append(resize_rgb(image, size).transpose(2, 0, 1))
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError("Policy needs a finite 16-D end-effector state")
        states.append((state - stats["state_mean"]) / stats["state_std"])
    return np.stack(images), np.asarray(states, dtype=np.float32)


def tensor_observations(images, states, device):
    return (torch.as_tensor(images, device=device).float() / 255,
            torch.as_tensor(states, device=device).float())


def project_actions(actions, fallback=None):
    """Normalize wxyz quaternions and clamp grippers at the environment boundary.

    Rollout and evaluation share this deterministic map. PPO likelihoods are
    calculated on the original sampled latent action, before projection.
    """
    values = np.asarray(actions, dtype=np.float32).copy()
    if values.shape[-1] != ACTION_DIM or not np.isfinite(values).all():
        raise ValueError("Non-finite or malformed predicted action")
    for start in (3, 11):
        quaternion = values[..., start:start + 4]
        norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
        default = np.array([1, 0, 0, 0], dtype=np.float32)
        if fallback is not None:
            default = np.asarray(fallback, dtype=np.float32)[..., start:start + 4]
            n = np.linalg.norm(default, axis=-1, keepdims=True)
            default = np.where(n > 1e-6, default / np.maximum(n, 1e-6), [1, 0, 0, 0])
        values[..., start:start + 4] = np.where(norm > 1e-6, quaternion / np.maximum(norm, 1e-6), default)
    values[..., [7, 15]] = values[..., [7, 15]].clip(0, 1)
    return values


class Policy:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.payload = load_payload(checkpoint)
        self.kind, self.config, self.stats = self.payload["kind"], self.payload["model_config"], self.payload["stats"]
        self.model = build_model(self.kind, self.config).to(self.device)
        self.model.load_state_dict(self.payload.get("ema") or self.payload["model"])
        self.model.eval()
        self.residual = None
        self.diffusion_base = None
        if self.payload.get("diffusion_base") is not None:
            self.diffusion_base = build_model(self.kind, self.config).to(self.device).eval()
            self.diffusion_base.load_state_dict(self.payload["diffusion_base"])
        if self.payload.get("residual") is not None:
            self.residual = ResidualActor(self.config).to(self.device)
            self.residual.load_state_dict(self.payload["residual"])
            self.residual.eval()

    @torch.no_grad()
    def predict(self, observations):
        arrays = observation_arrays(observations, self.stats, self.config["image_size"])
        image, state = tensor_observations(*arrays, self.device)
        if self.kind == "act":
            action = self.model(image, state)[0]
            if self.residual is not None:
                latent = self.residual.distribution(image, state, action).mean
                action = action + float(self.payload["residual_scale"]) * latent.tanh()
        else:
            action, _ = self.model.sample(image, state, prefix_model=self.diffusion_base,
                                          finetune_steps=self.payload.get("finetune_denoising_steps"))
        mean = torch.as_tensor(self.stats["action_mean"], device=self.device)
        std = torch.as_tensor(self.stats["action_std"], device=self.device)
        decoded = (action * std + mean).cpu().numpy()
        fallback = np.asarray([obs["proprio"] if "proprio" in obs else pack_ee_action(obs["state"])
                               for obs in observations])[:, None, :]
        return project_actions(decoded, fallback)[:, :int(self.config["execute_steps"])]


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def assert_same_state(model, snapshot):
    if any(not torch.equal(value.detach().cpu(), snapshot[key]) for key, value in model.state_dict().items()):
        raise AssertionError("Frozen ACT base changed during residual training")
