"""Diffusion Policy core: observation encoder, epsilon network, DDPM/DDIM schedule,
demonstration dataset, and checkpoint helpers.

The module deliberately depends only on ``torch`` + ``numpy`` at import time.
Heavy data dependencies (``h5py`` / ``cv2`` / XPolicyLab's ``decode_image_bit``)
are imported inside the data-loading functions, so the model can be smoke-tested
on a CPU-only box without the simulator stack.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starter.interfaces import ACTION_DIM, ARM_KEYS, ARM_WIDTHS, HEAD_CAMERA

IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
GRIPPER_INDICES = (7, 15)


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str = "cuda") -> torch.device:
    requested = str(name or "cuda")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[dp] CUDA is not available; falling back to CPU (requested {requested}).", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def cfg_get(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = config.get(key, default)
    if value is None:
        return default
    return value


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """A small convolutional torso. 96x96 -> 6x6 -> global pool -> ``out_dim``."""

    def __init__(self, out_dim: int = 128, in_channels: int = 3) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, cout),
                nn.SiLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class ProprioEncoder(nn.Module):
    def __init__(self, in_dim: int = ACTION_DIM, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)


class ObsEncoder(nn.Module):
    """Image + 16-D end-effector state -> one context vector."""

    def __init__(
        self,
        state_dim: int = ACTION_DIM,
        vision_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(vision_dim)
        self.proprio_encoder = ProprioEncoder(state_dim, vision_dim)
        self.fuse = nn.Sequential(
            nn.Linear(vision_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.image_encoder(image), self.proprio_encoder(proprio)], dim=-1))


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class EpsilonNet(nn.Module):
    """Predicts the noise on the flattened action chunk at noise level ``t``."""

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        chunk_size: int = 8,
        context_dim: int = 256,
        hidden_dim: int = 512,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        flat_dim = self.action_dim * self.chunk_size
        self.time_embed = TimestepEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(flat_dim + context_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(self, x_t: torch.Tensor, context: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(timesteps)
        return self.net(torch.cat([x_t, context, temb], dim=-1))


# --------------------------------------------------------------------------------------
# diffusion schedule
# --------------------------------------------------------------------------------------

def make_timesteps(total_steps: int, sample_steps: int) -> List[int]:
    """Evenly spaced descending DDIM timesteps, always ending at 0."""

    total_steps = int(total_steps)
    sample_steps = max(1, int(sample_steps))
    if sample_steps == 1:
        return [0]
    indices = torch.linspace(total_steps - 1, 0, sample_steps).round().long().tolist()
    out: List[int] = []
    for index in indices:
        value = int(index)
        if not out or value < out[-1]:
            out.append(value)
    if out[-1] != 0:
        out.append(0)
    return out


def normalize_quaternions(actions: torch.Tensor) -> torch.Tensor:
    """Normalize the (qw, qx, qy, qz) blocks of a 16-D ee action vector."""

    normalized = actions.clone()
    for start in (3, 11):
        quaternion = normalized[..., start : start + 4]
        normalized[..., start : start + 4] = quaternion / quaternion.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
    return normalized


def gaussian_logp(value: torch.Tensor, mean: torch.Tensor, sigma: Any) -> torch.Tensor:
    """Elementwise Gaussian log-density, summed over the last dimension."""

    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(device=mean.device, dtype=mean.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(mean.shape[0], 1)
    else:
        sigma = torch.full((mean.shape[0], 1), float(sigma), device=mean.device, dtype=mean.dtype)
    var = (sigma * sigma).clamp(min=1e-12)
    logp = -0.5 * (((value - mean) ** 2) / var + torch.log(2.0 * math.pi * var))
    return logp.sum(dim=-1)


class DiffusionPolicy(nn.Module):
    """Observation-conditioned epsilon-prediction diffusion policy."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.action_dim = int(cfg_get(config, "action_dim", ACTION_DIM))
        self.state_dim = int(cfg_get(config, "state_dim", ACTION_DIM))
        self.chunk_size = int(cfg_get(config, "chunk_size", 8))
        self.image_size = int(cfg_get(config, "image_size", 96))
        self.diffusion_steps = int(cfg_get(config, "diffusion_steps", 100))
        hidden_dim = int(cfg_get(config, "hidden_dim", 512))
        self.context_dim = int(cfg_get(config, "context_dim", 256))

        beta_start = float(cfg_get(config, "beta_start", 1e-4))
        beta_end = float(cfg_get(config, "beta_end", 0.02))
        betas = torch.linspace(beta_start, beta_end, self.diffusion_steps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float(), persistent=True)
        self.register_buffer("alpha_bars", alpha_bars.float(), persistent=True)

        self.obs_encoder = ObsEncoder(
            state_dim=self.state_dim,
            vision_dim=int(cfg_get(config, "vision_dim", 128)),
            hidden_dim=int(cfg_get(config, "encoder_hidden_dim", 256)),
            out_dim=self.context_dim,
        )
        self.eps_net = EpsilonNet(
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            context_dim=self.context_dim,
            hidden_dim=hidden_dim,
            time_dim=int(cfg_get(config, "time_dim", 128)),
        )

    @property
    def flat_action_dim(self) -> int:
        return self.action_dim * self.chunk_size

    def encode_obs(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.obs_encoder(image, proprio)

    def predict_eps(self, x_t: torch.Tensor, context: torch.Tensor, timesteps: Any) -> torch.Tensor:
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.full((x_t.shape[0],), int(timesteps), device=x_t.device, dtype=torch.long)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(x_t.shape[0])
        return self.eps_net(x_t, context, timesteps)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].to(x0.dtype).unsqueeze(-1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def training_loss(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        batch = image.shape[0]
        flat = action.reshape(batch, -1)
        noise = torch.randn_like(flat)
        timesteps = torch.randint(0, self.diffusion_steps, (batch,), device=flat.device)
        x_t = self.q_sample(flat, timesteps, noise)
        context = self.encode_obs(image, proprio)
        predicted = self.predict_eps(x_t, context, timesteps)
        return F.mse_loss(predicted, noise)

    # -- sampling -----------------------------------------------------------------
    def _ddim_components(
        self,
        x_t: torch.Tensor,
        eps: torch.Tensor,
        t: int,
        t_prev: int,
        sigma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_t = self.alpha_bars[int(t)].to(x_t.dtype)
        alpha_prev = self.alpha_bars[int(t_prev)].to(x_t.dtype)
        x0 = ((x_t - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt()).clamp(-3.0, 3.0)
        sigma_tensor = torch.as_tensor(sigma, device=x_t.device, dtype=x_t.dtype)
        base = (1.0 - alpha_prev - sigma_tensor * sigma_tensor).clamp(min=1e-8).sqrt()
        mean = alpha_prev.sqrt() * x0 + base * eps
        return mean, x0

    def _sigma_for_step(self, t: int, t_prev: int, eta: float, std_min: float, std_max: float) -> float:
        if eta <= 0.0:
            return 0.0
        alpha_t = float(self.alpha_bars[int(t)].item())
        alpha_prev = float(self.alpha_bars[int(t_prev)].item())
        ratio = max((1.0 - alpha_prev) / max(1.0 - alpha_t, 1e-8), 0.0)
        gap = max(1.0 - alpha_t / max(alpha_prev, 1e-8), 0.0)
        sigma = float(eta) * math.sqrt(ratio) * math.sqrt(gap)
        return float(min(max(sigma, std_min), std_max))

    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        sample_steps: int = 10,
        eta: float = 0.0,
        stochastic_steps: int = 0,
        std_min: float = 0.02,
        std_max: float = 0.5,
        generator: Optional[torch.Generator] = None,
        return_chain: bool = False,
    ):
        """Sample a normalized action chunk.

        ``eta > 0`` turns the last ``stochastic_steps`` DDIM transitions into
        Gaussian policy steps.  The returned chain contains only those steps and
        is what DPPO optimizes.
        """

        batch = image.shape[0]
        context = self.encode_obs(image, proprio)
        x = torch.randn(batch, self.flat_action_dim, device=image.device, generator=generator)
        timesteps = make_timesteps(self.diffusion_steps, sample_steps)
        chain: List[Dict[str, Any]] = []
        stochastic_steps = max(0, int(stochastic_steps))

        for index in range(len(timesteps) - 1):
            t = timesteps[index]
            t_prev = timesteps[index + 1]
            eps = self.predict_eps(x, context, t)
            sigma = 0.0
            if stochastic_steps > 0 and index >= len(timesteps) - 1 - stochastic_steps:
                sigma = self._sigma_for_step(t, t_prev, eta, std_min, std_max)
            mean, _x0 = self._ddim_components(x, eps, t, t_prev, sigma)
            if sigma > 0.0:
                noise = torch.randn(batch, self.flat_action_dim, device=x.device, generator=generator)
                x_prev = mean + sigma * noise
                logp = gaussian_logp(x_prev, mean, sigma)
                chain.append(
                    {
                        "t": int(t),
                        "t_prev": int(t_prev),
                        "sigma": float(sigma),
                        "x_t": x.detach().cpu(),
                        "x_prev": x_prev.detach().cpu(),
                        "logp": logp.detach().cpu(),
                    }
                )
                x = x_prev
            else:
                x = mean

        last_t = timesteps[-1]
        eps = self.predict_eps(x, context, last_t)
        alpha = self.alpha_bars[int(last_t)].to(x.dtype)
        actions = ((x - (1.0 - alpha).sqrt() * eps) / alpha.sqrt()).clamp(-3.0, 3.0)
        if not return_chain:
            return actions
        return actions, chain

    # -- PPO support --------------------------------------------------------------
    def chain_logp(
        self,
        x_t: torch.Tensor,
        x_prev: torch.Tensor,
        context: torch.Tensor,
        timesteps: torch.Tensor,
        t_prev: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute the Gaussian log-probability of stored DDIM transitions."""

        eps = self.predict_eps(x_t, context, timesteps)
        alpha_t = self.alpha_bars.to(x_t.dtype)[timesteps].unsqueeze(-1)
        alpha_prev = self.alpha_bars.to(x_t.dtype)[t_prev].unsqueeze(-1)
        x0 = ((x_t - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt()).clamp(-3.0, 3.0)
        sigma_col = sigma.to(x_t.dtype).unsqueeze(-1)
        base = (1.0 - alpha_prev - sigma_col * sigma_col).clamp(min=1e-8).sqrt()
        mean = alpha_prev.sqrt() * x0 + base * eps
        var = (sigma_col * sigma_col).clamp(min=1e-12)
        logp = -0.5 * (((x_prev - mean) ** 2) / var + torch.log(2.0 * math.pi * var))
        return logp.sum(dim=-1)


class Critic(nn.Module):
    """State-value function with its own small vision torso."""

    def __init__(self, state_dim: int = ACTION_DIM, out_dim: int = 128) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(out_dim)
        self.proprio_encoder = ProprioEncoder(state_dim, out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim * 2, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [self.image_encoder(image), self.proprio_encoder(proprio)], dim=-1
        )
        return self.head(features).squeeze(-1)


class EMA:
    """Exponential moving average of a policy's parameters (buffers copied)."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.decay
        for shadow, parameter in zip(self.module.parameters(), model.parameters()):
            shadow.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        for shadow, buffer in zip(self.module.buffers(), model.buffers()):
            shadow.copy_(buffer)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.module.state_dict()


# --------------------------------------------------------------------------------------
# normalization / data
# --------------------------------------------------------------------------------------

@dataclass
class Normalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Normalization":
        return cls(
            state_mean=np.asarray(payload["state_mean"], dtype=np.float32),
            state_std=np.asarray(payload["state_std"], dtype=np.float32),
            action_mean=np.asarray(payload["action_mean"], dtype=np.float32),
            action_std=np.asarray(payload["action_std"], dtype=np.float32),
        )

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.state_mean, device=state.device, dtype=state.dtype)
        std = torch.as_tensor(self.state_std, device=state.device, dtype=state.dtype)
        return (state - mean) / std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.action_mean, device=action.device, dtype=action.dtype)
        std = torch.as_tensor(self.action_std, device=action.device, dtype=action.dtype)
        return (action - mean) / std

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.action_mean, device=action.device, dtype=action.dtype)
        std = torch.as_tensor(self.action_std, device=action.device, dtype=action.dtype)
        return action * std + mean


def _xpolicylab_candidates() -> List[Path]:
    repo = Path(__file__).resolve().parents[3]
    return [
        repo / "third_party" / "XPolicyLab",
        repo / "third_party" / "RoboDojo" / "XPolicyLab",
    ]


def load_image_decoder():
    """Return XPolicyLab's ``decode_image_bit``; fail loudly with instructions."""

    import sys

    candidates = _xpolicylab_candidates()
    for root in candidates:
        if (root / "XPolicyLab" / "utils" / "process_data.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            break
    try:
        from XPolicyLab.utils.process_data import decode_image_bit  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on user's checkout
        raise ImportError(
            "Could not import XPolicyLab.utils.process_data.decode_image_bit. "
            "Run `bash scripts/bootstrap_third_party.sh` in startcode (or add the "
            "XPolicyLab checkout to PYTHONPATH). The demonstrations store legacy "
            "channel-reversed JPEGs; PIL cannot decode them correctly."
        ) from exc
    return decode_image_bit


def _decode_stored_frames(encoded, decoder) -> np.ndarray:
    frames = decoder(encoded)
    array = np.asarray(frames)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(f"Decoded camera buffer has shape {array.shape}, expected (T, H, W, 3).")
    return array.astype(np.uint8)


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install opencv-python to resize demonstration frames.") from exc
    resized = [cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in frames]
    return np.stack(resized, axis=0).astype(np.uint8)


def load_demo_episodes(config: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    """Load ``episode_*.hdf5`` into memory as image/state/action arrays."""

    from starter.data.demos import episode_files, read_episode

    root = Path(str(cfg_get(config, "data_path", "data/RoboDojo/stack_blocks/arx_x5")))
    files = episode_files(root)
    if not files:
        raise FileNotFoundError(
            f"No episode_*.hdf5 found under {root}. Run `bash scripts/download_stack_blocks.sh` "
            "in startcode, then point data_path at data/RoboDojo/stack_blocks/arx_x5."
        )
    limit = int(cfg_get(config, "max_episodes", 0))
    if limit > 0:
        files = files[:limit]
    image_size = int(cfg_get(config, "image_size", 96))
    decoder = load_image_decoder()

    episodes: List[Dict[str, np.ndarray]] = []
    for index, path in enumerate(files):
        episode = read_episode(path)
        states = np.asarray(episode["state"], dtype=np.float32)
        actions = np.asarray(episode["action"], dtype=np.float32)
        frames = _decode_stored_frames(episode["cam_head"], decoder)
        length = min(len(states), len(actions), len(frames))
        if length < 2:
            continue
        frames = _resize_frames(frames[:length], image_size)
        episodes.append(
            {
                "images": frames,
                "state": states[:length],
                "action": actions[:length],
                "path": np.asarray([str(path)]),
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(files):
            print(f"[dp] loaded {index + 1}/{len(files)} episodes", flush=True)
    if not episodes:
        raise RuntimeError("All demonstration episodes were empty or shorter than 2 steps.")
    return episodes


def compute_normalization(episodes: Sequence[Dict[str, np.ndarray]]) -> Normalization:
    states = np.concatenate([episode["state"] for episode in episodes], axis=0)
    actions = np.concatenate([episode["action"] for episode in episodes], axis=0)
    state_mean = states.mean(axis=0).astype(np.float32)
    state_std = np.maximum(states.std(axis=0), 1e-3).astype(np.float32)
    action_mean = actions.mean(axis=0).astype(np.float32)
    action_std = np.maximum(actions.std(axis=0), 1e-3).astype(np.float32)
    return Normalization(state_mean, state_std, action_mean, action_std)


class EpisodeDataset(torch.utils.data.Dataset):
    """(image, proprio, future action chunk) samples from demonstration episodes."""

    def __init__(
        self,
        episodes: Sequence[Dict[str, np.ndarray]],
        indices: Sequence[int],
        normalization: Normalization,
        chunk_size: int,
    ) -> None:
        self.episodes = episodes
        self.normalization = normalization
        self.chunk_size = int(chunk_size)
        self.samples: List[Tuple[int, int]] = []
        for episode_index in indices:
            length = len(self.episodes[episode_index]["action"])
            for step in range(length):
                self.samples.append((int(episode_index), int(step)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        episode_index, step = self.samples[item]
        episode = self.episodes[episode_index]
        images = episode["images"]
        states = episode["state"]
        actions = episode["action"]
        length = len(actions)

        image = torch.from_numpy(images[step]).permute(2, 0, 1).float() / 255.0
        image = (image - IMAGE_MEAN) / IMAGE_STD
        state = torch.from_numpy(states[step]).float()
        state = self.normalization.normalize_state(state)

        chunk = np.stack(
            [actions[min(step + offset, length - 1)] for offset in range(self.chunk_size)],
            axis=0,
        )
        chunk = torch.from_numpy(chunk).float()
        chunk = self.normalization.normalize_action(chunk)
        return {"image": image, "state": state, "action": chunk}


def prepare_image(image: Any, size: int = 96) -> torch.Tensor:
    """RGB HWC array (uint8/float) -> normalized CHW float tensor."""

    if isinstance(image, torch.Tensor):
        tensor = image.detach().float()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
    else:
        tensor = torch.from_numpy(np.asarray(image).astype(np.float32))
    if tensor.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {tuple(tensor.shape)}.")
    if tensor.shape[-1] == 4:
        tensor = tensor[..., :3]
    if tensor.max() > 2.0:
        tensor = tensor / 255.0
    if tensor.shape[0] != size or tensor.shape[1] != size:
        tensor = F.interpolate(
            tensor.permute(2, 0, 1).unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        tensor = tensor.permute(1, 2, 0)
    tensor = tensor.permute(2, 0, 1).contiguous()
    return (tensor - IMAGE_MEAN) / IMAGE_STD


# --------------------------------------------------------------------------------------
# checkpoint helpers
# --------------------------------------------------------------------------------------

def model_from_config(config: Dict[str, Any]) -> DiffusionPolicy:
    return DiffusionPolicy(config)


def save_checkpoint(
    path: Path,
    model: DiffusionPolicy,
    ema: Optional[EMA],
    config: Dict[str, Any],
    normalization: Normalization,
    kind: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "kind": kind,
        "version": 1,
        "config": dict(config),
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "normalization": normalization.to_dict(),
        "chunk_size": model.chunk_size,
        "image_size": model.image_size,
    }
    if extra:
        payload.update(extra)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def torch_load(path: Path, map_location: Any = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location=map_location)


def load_checkpoint_payload(path: str | Path, map_location: Any = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch_load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{path} is not a Diffusion Policy checkpoint.")
    return payload


def build_policy_from_payload(
    payload: Dict[str, Any],
    device: torch.device,
    prefer_ema: bool = True,
) -> Tuple[DiffusionPolicy, Dict[str, Any], Normalization]:
    model = DiffusionPolicy(payload["config"])
    state = payload.get("ema") if prefer_ema else None
    if state is None:
        state = payload["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    normalization = Normalization.from_dict(payload["normalization"])
    return model, payload.get("config", {}), normalization


class DiffusionRuntime:
    """Checkpoint + preprocessing wrapper shared by training-online and serving."""

    def __init__(
        self,
        payload: Dict[str, Any],
        device: torch.device,
        prefer_ema: bool = True,
    ) -> None:
        self.payload = payload
        self.device = device
        self.model, self.config, self.normalization = build_policy_from_payload(
            payload, device, prefer_ema=prefer_ema
        )
        self.chunk_size = int(payload.get("chunk_size", self.model.chunk_size))
        self.image_size = int(payload.get("image_size", self.model.image_size))
        self.sample_steps = int(cfg_get(self.config, "eval_sample_steps", 10))
        self.eval_chunk_size = int(cfg_get(self.config, "eval_chunk_size", self.chunk_size))

    @torch.no_grad()
    def act_chunk_from_arrays(self, image: Any, proprio: Sequence[float] | np.ndarray) -> np.ndarray:
        image_tensor = prepare_image(image, self.image_size).unsqueeze(0).to(self.device)
        proprio_tensor = torch.as_tensor(np.asarray(proprio, dtype=np.float32)).unsqueeze(0).to(self.device)
        proprio_tensor = self.normalization.normalize_state(proprio_tensor)
        normalized = self.model.sample(
            image_tensor,
            proprio_tensor,
            sample_steps=self.sample_steps,
            eta=0.0,
            stochastic_steps=0,
        )
        normalized = normalized.reshape(1, self.model.chunk_size, self.model.action_dim)
        actions = self.normalization.unnormalize_action(normalized)[0]
        actions = actions[: self.eval_chunk_size]
        actions = normalize_quaternions(actions).cpu().numpy()
        actions[:, list(GRIPPER_INDICES)] = np.clip(actions[:, list(GRIPPER_INDICES)], 0.0, 1.0)
        return actions

    def act_chunk_from_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        image, proprio = observation_to_arrays(obs)
        return self.act_chunk_from_arrays(image, proprio)


def observation_to_arrays(obs: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
    """Accept both the starter ``image/proprio`` observation and RoboDojo's raw one."""

    if "image" in obs and "proprio" in obs:
        return obs["image"], np.asarray(obs["proprio"], dtype=np.float32).reshape(-1)[:ACTION_DIM]

    vision = obs.get("vision", {})
    camera = vision.get(HEAD_CAMERA, {})
    image = camera.get("color", camera.get("colors"))
    if image is None:
        raise KeyError("Observation has neither 'image' nor vision.cam_head.color.")
    state = obs.get("state", {})
    values: List[float] = []
    for key, width in zip(ARM_KEYS, ARM_WIDTHS):
        if key not in state:
            raise KeyError(f"Observation state is missing {key}.")
        array = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if array.size < width:
            raise ValueError(f"{key} has {array.size} values, expected {width}.")
        values.extend(array[:width].tolist())
    return image, np.asarray(values, dtype=np.float32)
