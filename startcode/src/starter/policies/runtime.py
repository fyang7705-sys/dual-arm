"""Shared preprocessing, checkpoint format, inference, and run artifacts."""
from __future__ import annotations

import json
import math
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
           "diffusion_steps": 20, "image_size": 96, "execute_steps": 4,
           "backbone": "cnn", "dropout": 0.1, "obs_steps": 1, "inference_steps": 0,
           "temporal_ensemble": False, "ensemble_k": 0.01, "min_snr_gamma": 0.0,
           "decoder_layers": 0, "dim_feedforward": 0}
    cfg.update({key: config[key] for key in cfg if key in config})
    if int(cfg["width"]) % int(cfg["heads"]) or int(cfg["width"]) % 2:
        raise ValueError("width must be even and divisible by heads")
    if not 1 <= int(cfg["execute_steps"]) <= int(cfg["chunk"]):
        raise ValueError("execute_steps must be between 1 and chunk")
    if int(cfg["obs_steps"]) < 1 or int(cfg["inference_steps"]) < 0:
        raise ValueError("obs_steps must be positive and inference_steps nonnegative")
    if str(cfg["backbone"]) not in ("cnn", "resnet18"):
        raise ValueError("backbone must be cnn or resnet18")
    if int(cfg["obs_steps"]) > 1 and str(cfg["backbone"]) != "resnet18":
        raise ValueError("obs_steps > 1 requires backbone: resnet18")
    if int(cfg["inference_steps"]) > int(cfg["diffusion_steps"]):
        raise ValueError("inference_steps cannot exceed diffusion_steps")
    if float(cfg["min_snr_gamma"]) < 0:
        raise ValueError("min_snr_gamma must be nonnegative")
    if int(cfg["decoder_layers"]) < 0 or int(cfg["dim_feedforward"]) < 0:
        raise ValueError("decoder_layers and dim_feedforward must be nonnegative")
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


def observation_image(observation):
    return observation["image"] if "image" in observation else observation["vision"]["cam_head"]["color"]


class FrameHistory:
    """Per-slot frame buffer for multi-frame policies; pads with the oldest frame."""

    def __init__(self, steps, size):
        self.steps, self.size, self.slots = int(steps), size, {}

    def push(self, slot, image):
        buffer = self.slots.setdefault(slot, [])
        buffer.append(resize_rgb(image, self.size).transpose(2, 0, 1))
        del buffer[:-self.steps]
        return self.stack(slot)

    def stack(self, slot):
        buffer = self.slots.get(slot)
        if not buffer:
            raise KeyError(f"No frame history for slot {slot}; push before stack")
        return np.stack([buffer[0]] * (self.steps - len(buffer)) + buffer)

    def reset(self):
        self.slots.clear()


class TemporalEnsemble:
    """ACT ``temporal_agg`` with the reference weighting convention.

    Predictions are weighted ``exp(-k * rank)`` where ``rank`` orders the
    chunks covering the current step, oldest first.  This is exactly the ACT
    reference evaluation script, so with ``k = 0.01`` older chunks dominate;
    a negative ``k`` favors newer chunks instead.
    """

    def __init__(self, chunk, k):
        self.chunk, self.k, self.slots = int(chunk), float(k), {}

    def predict(self, slot, chunk):
        state = self.slots.setdefault(slot, {"t": 0, "chunks": []})
        t = state["t"]
        state["chunks"] = [(start, candidate) for start, candidate in state["chunks"] if start + self.chunk > t]
        state["chunks"].append((t, np.asarray(chunk, dtype=np.float32)))
        total, weight = np.zeros(chunk.shape[-1], dtype=np.float32), 0.0
        for rank, (start, candidate) in enumerate(state["chunks"]):
            age = t - start
            if 0 <= age < self.chunk:
                value = math.exp(-self.k * rank)
                total += value * candidate[age]
                weight += value
        state["t"] = t + 1
        return total / max(weight, 1e-8)

    def reset(self):
        self.slots.clear()


def observation_arrays(observations, stats, size, history=None, slots=None, push=False):
    images, states = [], []
    for index, observation in enumerate(observations):
        image = observation_image(observation)
        state = observation["proprio"] if "proprio" in observation else pack_ee_action(observation["state"])
        if history is not None:
            slot = slots[index] if slots is not None else index
            # ``push`` advances the control-rate history (deployment); without
            # it the stack is read, initializing a never-seen slot from the
            # current frame so standalone sampling still works.
            images.append(history.push(slot, image) if push or slot not in history.slots
                          else history.stack(slot))
        else:
            images.append(resize_rgb(image, size).transpose(2, 0, 1))
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError("Policy needs a finite 16-D end-effector state")
        # Training data canonicalizes quaternions to w >= 0; match it at inference.
        for start in (3, 11):
            if state[start] < 0:
                state[start:start + 4] = -state[start:start + 4]
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
        obs_steps = int(self.config.get("obs_steps", 1))
        self.history = FrameHistory(obs_steps, self.config["image_size"]) if obs_steps > 1 else None
        self.ensemble = (TemporalEnsemble(int(self.config["chunk"]), float(self.config.get("ensemble_k", 0.01)))
                         if self.config.get("temporal_ensemble") else None)

    def reset(self):
        """Clear per-episode state; the evaluation server calls this on reset."""
        if self.history is not None:
            self.history.reset()
        if self.ensemble is not None:
            self.ensemble.reset()

    @torch.no_grad()
    def predict(self, observations):
        slots = [obs.get("env_idx", index) if isinstance(obs, dict) else index
                 for index, obs in enumerate(observations)]
        arrays = observation_arrays(observations, self.stats, self.config["image_size"],
                                    history=self.history, slots=slots, push=True)
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
        if self.ensemble is not None:
            # Re-query every step; average the overlapping chunks per slot.
            actions = np.stack([self.ensemble.predict(slot, decoded[j])
                                for j, slot in enumerate(slots)])[:, None, :]
            return project_actions(actions, fallback)
        # A multi-frame policy must advance one control step at a time so its
        # frame history never skips observations between queries.
        steps = 1 if self.history is not None else int(self.config["execute_steps"])
        return project_actions(decoded, fallback)[:, :steps]


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def assert_same_state(model, snapshot):
    if any(not torch.equal(value.detach().cpu(), snapshot[key]) for key, value in model.state_dict().items()):
        raise AssertionError("Frozen ACT base changed during residual training")
