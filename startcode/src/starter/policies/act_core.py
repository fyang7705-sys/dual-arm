"""ACT policy core: CVAE + transformer action-chunk predictor, dataset and
checkpoint helpers for the ACT + residual-PPO route.

Import-time dependencies are only ``torch`` and ``numpy``; HDF5 / OpenCV /
XPolicyLab are imported inside the data pipeline so the model can be tested on
a CPU-only machine.
"""

from __future__ import annotations

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


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str = "cuda") -> torch.device:
    requested = str(name or "cuda")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[act] CUDA is not available; falling back to CPU (requested {requested}).", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def cfg_get(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------------------

def normalize_quaternions_np(actions: np.ndarray) -> np.ndarray:
    """Normalize the two (qw, qx, qy, qz) blocks of an (N, 16) action array."""

    normalized = np.array(actions, dtype=np.float64, copy=True)
    for start in (3, 11):
        quaternion = normalized[:, start : start + 4]
        norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
        normalized[:, start : start + 4] = quaternion / np.clip(norm, 1e-8, None)
    return normalized


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, in_channels: int = 3) -> None:
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
    def __init__(self, in_dim: int = ACTION_DIM, out_dim: int = 256) -> None:
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


class VAEEncoder(nn.Module):
    """Maps (observation features, action chunk) to a diagonal Gaussian latent."""

    def __init__(self, feature_dim: int, action_dim: int, chunk_size: int, latent_dim: int = 32) -> None:
        super().__init__()
        input_dim = feature_dim + action_dim * chunk_size
        hidden = max(128, feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = features.shape[0]
        flat = action.reshape(batch, -1)
        hidden = self.trunk(torch.cat([features, flat], dim=-1))
        return self.mu(hidden), self.logvar(hidden).clamp(-8.0, 8.0)


class ActPolicy(nn.Module):
    """Compact ACT: CVAE latent + transformer decoder over action-chunk queries."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.action_dim = int(cfg_get(config, "action_dim", ACTION_DIM))
        self.state_dim = int(cfg_get(config, "state_dim", ACTION_DIM))
        self.chunk_size = int(cfg_get(config, "chunk_size", 16))
        self.image_size = int(cfg_get(config, "image_size", 96))
        self.latent_dim = int(cfg_get(config, "latent_dim", 32))
        self.d_model = int(cfg_get(config, "d_model", 256))
        self.feature_dim = self.d_model * 2

        self.image_encoder = ImageEncoder(self.d_model)
        self.proprio_encoder = ProprioEncoder(self.state_dim, self.d_model)
        self.vae = VAEEncoder(self.feature_dim, self.action_dim, self.chunk_size, self.latent_dim)
        self.latent_proj = nn.Linear(self.latent_dim, self.d_model)
        self.memory_pos = nn.Parameter(torch.zeros(1, 3, self.d_model))
        self.query_pos = nn.Parameter(torch.zeros(1, self.chunk_size, self.d_model))
        nn.init.normal_(self.memory_pos, std=0.02)
        nn.init.normal_(self.query_pos, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(cfg_get(config, "nhead", 4)),
            dim_feedforward=int(cfg_get(config, "dim_feedforward", 512)),
            dropout=float(cfg_get(config, "dropout", 0.1)),
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(cfg_get(config, "decoder_layers", 2))
        )
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_model, self.action_dim),
        )

    # -- observation -------------------------------------------------------------
    def encode_obs(self, image: torch.Tensor, proprio: torch.Tensor):
        image_feature = self.image_encoder(image)
        proprio_feature = self.proprio_encoder(proprio)
        memory = torch.stack([image_feature, proprio_feature], dim=1)
        features = torch.cat([image_feature, proprio_feature], dim=-1)
        return memory, features

    def _decode(self, memory: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        batch = memory.shape[0]
        memory = torch.cat([memory, self.latent_proj(latent).unsqueeze(1)], dim=1)
        memory = memory + self.memory_pos
        queries = self.query_pos.expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.head(decoded)

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        use_latent: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, features = self.encode_obs(image, proprio)
        mu, logvar = self.vae(features, action)
        if use_latent and self.training:
            std = torch.exp(0.5 * logvar)
            latent = mu + std * torch.randn_like(std)
        else:
            latent = mu * 0.0  # zero latent at inference, as in ACT
        prediction = self._decode(memory, latent)
        return prediction, mu, logvar

    @torch.no_grad()
    def act_chunk(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Deterministic normalized action chunk, latent fixed at zero."""

        memory, _features = self.encode_obs(image, proprio)
        latent = torch.zeros(image.shape[0], self.latent_dim, device=image.device, dtype=image.dtype)
        return self._decode(memory, latent)

    def training_loss(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        prediction, mu, logvar = self.forward(image, proprio, action, use_latent=True)
        per_step = F.l1_loss(prediction, action, reduction="none").mean(dim=-1)
        if valid is None:
            l1 = per_step.mean()
        else:
            mask = valid.to(per_step.device, per_step.dtype)
            l1 = (per_step * mask).sum() / mask.sum().clamp(min=1.0)
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        kl_weight = float(cfg_get(self.config, "kl_weight", 10.0))
        loss = l1 + kl_weight * kl
        return loss, {"l1": float(l1.detach().cpu()), "kl": float(kl.detach().cpu())}

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def weight_fingerprint(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()


class ResidualHead(nn.Module):
    """Small Gaussian head over a bounded residual in real action units."""

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        action_dim: int = ACTION_DIM,
        residual_scales: Optional[Sequence[float]] = None,
        initial_std: float = 0.3,
        use_base_action: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.use_base_action = bool(use_base_action)
        if residual_scales is None:
            residual_scales = default_residual_scales()
        scales = torch.as_tensor(list(residual_scales), dtype=torch.float32)
        if scales.numel() != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} residual scales, got {scales.numel()}.")
        self.register_buffer("scales", scales)
        input_dim = feature_dim + self.action_dim if self.use_base_action else feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, self.action_dim),
        )
        self.log_std = nn.Parameter(torch.full((self.action_dim,), math.log(max(initial_std, 1e-3))))

    def forward(self, features: torch.Tensor, base_action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_base_action:
            if base_action is None:
                raise ValueError("ResidualHead was built with use_base_action=True but got no base action")
            features = torch.cat([features, base_action], dim=-1)
        return self.net(features)

    def std(self, std_min: float = 0.05, std_max: float = 0.5) -> torch.Tensor:
        return F.softplus(self.log_std).clamp(min=std_min, max=std_max)

    def sample(
        self,
        features: torch.Tensor,
        base_action: Optional[torch.Tensor] = None,
        std_min: float = 0.05,
        std_max: float = 0.5,
    ):
        mean = self.forward(features, base_action)
        std = self.std(std_min, std_max).expand_as(mean)
        noise = torch.randn_like(mean)
        u = mean + std * noise
        return mean, std, u

    def residual(self, u: torch.Tensor) -> torch.Tensor:
        return self.scales * u.clamp(-1.0, 1.0)

    def deterministic_residual(
        self, features: torch.Tensor, base_action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.residual(self.forward(features, base_action))


def default_residual_scales() -> List[float]:
    """Real-unit bounds: 2 cm / 0.005 quat for poses, 0.05 for grippers."""

    pose = [0.02, 0.02, 0.02, 0.005, 0.005, 0.005, 0.005]
    return pose + [0.05] + pose + [0.05]


# --------------------------------------------------------------------------------------
# normalization and data
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
    import sys

    for root in _xpolicylab_candidates():
        if (root / "XPolicyLab" / "utils" / "process_data.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            break
    try:
        from XPolicyLab.utils.process_data import decode_image_bit  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import XPolicyLab.utils.process_data.decode_image_bit. "
            "Run `bash scripts/bootstrap_third_party.sh` in startcode first; the "
            "demonstration JPEGs use a legacy channel order that PIL gets wrong."
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
    return np.stack(
        [cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in frames], axis=0
    ).astype(np.uint8)


def load_demo_episodes(config: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    from starter.data.demos import episode_files, read_episode

    root = Path(str(cfg_get(config, "data_path", "data/RoboDojo/stack_blocks/arx_x5")))
    files = episode_files(root)
    if not files:
        raise FileNotFoundError(
            f"No episode_*.hdf5 found under {root}. Run `bash scripts/download_stack_blocks.sh` "
            "in startcode first."
        )
    limit = int(cfg_get(config, "max_episodes", 0))
    if limit > 0:
        files = files[:limit]
    image_size = int(cfg_get(config, "image_size", 96))
    decoder = load_image_decoder()
    episodes: List[Dict[str, np.ndarray]] = []
    for index, path in enumerate(files):
        episode = read_episode(path)
        states = canonicalize_quaternions(np.asarray(episode["state"], dtype=np.float32))
        actions = canonicalize_quaternions(np.asarray(episode["action"], dtype=np.float32))
        frames = _decode_stored_frames(episode["cam_head"], decoder)
        length = min(len(states), len(actions), len(frames))
        if length < 2:
            continue
        episodes.append(
            {
                "images": _resize_frames(frames[:length], image_size),
                "state": states[:length],
                "action": actions[:length],
                "path": np.asarray([str(path)]),
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(files):
            print(f"[act] loaded {index + 1}/{len(files)} episodes", flush=True)
    if not episodes:
        raise RuntimeError("All demonstration episodes were empty or shorter than 2 steps.")
    return episodes


def compute_normalization(episodes: Sequence[Dict[str, np.ndarray]]) -> Normalization:
    states = np.concatenate([episode["state"] for episode in episodes], axis=0)
    actions = np.concatenate([episode["action"] for episode in episodes], axis=0)
    return Normalization(
        states.mean(axis=0).astype(np.float32),
        np.maximum(states.std(axis=0), 1e-3).astype(np.float32),
        actions.mean(axis=0).astype(np.float32),
        np.maximum(actions.std(axis=0), 1e-3).astype(np.float32),
    )


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes: Sequence[Dict[str, np.ndarray]],
        indices: Sequence[int],
        normalization: Normalization,
        chunk_size: int,
        augment: bool = False,
    ) -> None:
        self.episodes = episodes
        self.normalization = normalization
        self.chunk_size = int(chunk_size)
        self.augment = bool(augment)
        self.samples: List[Tuple[int, int]] = [
            (int(episode_index), step)
            for episode_index in indices
            for step in range(len(self.episodes[episode_index]["action"]))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        episode_index, step = self.samples[item]
        episode = self.episodes[episode_index]
        length = len(episode["action"])

        image = torch.from_numpy(episode["images"][step]).permute(2, 0, 1).float() / 255.0
        if self.augment:
            image = augment_image(image)
        image = (image - IMAGE_MEAN) / IMAGE_STD
        state = self.normalization.normalize_state(torch.from_numpy(episode["state"][step]).float())
        chunk = np.stack(
            [episode["action"][min(step + offset, length - 1)] for offset in range(self.chunk_size)],
            axis=0,
        )
        valid = np.zeros(self.chunk_size, dtype=np.float32)
        for offset in range(self.chunk_size):
            valid[offset] = 1.0 if step + offset < length else 0.0
        action = self.normalization.normalize_action(torch.from_numpy(chunk).float())
        return {"image": image, "state": state, "action": action, "valid": torch.from_numpy(valid)}


def augment_image(image: torch.Tensor) -> torch.Tensor:
    """Random crop + mild brightness/contrast jitter on a float CHW image in [0,1]."""

    height, width = image.shape[-2:]
    scale = random.uniform(0.85, 0.95)
    crop_h, crop_w = max(1, int(height * scale)), max(1, int(width * scale))
    top = random.randint(0, max(0, height - crop_h))
    left = random.randint(0, max(0, width - crop_w))
    cropped = image[:, top : top + crop_h, left : left + crop_w].unsqueeze(0)
    image = F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
    brightness = random.uniform(0.9, 1.1)
    contrast = random.uniform(0.9, 1.1)
    image = (image * brightness).clamp(0.0, 1.0)
    mean = image.mean(dim=(-2, -1), keepdim=True)
    return ((image - mean) * contrast + mean).clamp(0.0, 1.0)


def canonicalize_quaternions(vector: np.ndarray) -> np.ndarray:
    """Unit-normalize and canonicalize (qw >= 0) both ee quaternion blocks."""

    vector = np.asarray(vector, dtype=np.float32).copy()
    for start in (3, 11):
        quaternion = vector[:, start : start + 4]
        norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
        quaternion = quaternion / np.clip(norm, 1e-8, None)
        sign = np.where(quaternion[:, 0:1] >= 0.0, 1.0, -1.0).astype(np.float32)
        vector[:, start : start + 4] = quaternion * sign
    return vector


def prepare_image(image: Any, size: int = 96) -> torch.Tensor:
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
            tensor.permute(2, 0, 1).unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0).permute(1, 2, 0)
    tensor = tensor.permute(2, 0, 1).contiguous()
    return (tensor - IMAGE_MEAN) / IMAGE_STD


# --------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------

def torch_load(path: Path, map_location: Any = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_act_payload(
    model: ActPolicy,
    config: Dict[str, Any],
    normalization: Normalization,
    kind: str = "act_policy",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "kind": kind,
        "version": 1,
        "config": dict(config),
        "model": model.state_dict(),
        "normalization": normalization.to_dict(),
        "chunk_size": model.chunk_size,
        "image_size": model.image_size,
    }
    if extra:
        payload.update(extra)
    return payload


def load_payload(path: str | Path, map_location: Any = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch_load(path, map_location=map_location)
    if not isinstance(payload, dict) or "kind" not in payload:
        raise ValueError(f"{path} is not an ACT checkpoint.")
    return payload


def build_act_from_payload(payload: Dict[str, Any], device: torch.device) -> ActPolicy:
    model = ActPolicy(payload["config"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    model.freeze()
    return model


class ActRuntime:
    """Frozen ACT base, optionally with the residual PPO head, for evaluation."""

    def __init__(self, payload: Dict[str, Any], device: torch.device) -> None:
        self.payload = payload
        self.device = device
        kind = str(payload.get("kind", "act_policy"))
        self.is_residual = kind.startswith("act_residual")
        if self.is_residual:
            self.base = ActPolicy(payload["base_config"])
            self.base.load_state_dict(payload["base_state"])
            self.base.to(device).eval()
            self.base.freeze()
            config = payload.get("config", {})
            self.residual = ResidualHead(
                feature_dim=int(payload.get("feature_dim", self.base.feature_dim)),
                hidden_dim=int(cfg_get(config, "residual_hidden_dim", 256)),
                action_dim=int(payload.get("action_dim", ACTION_DIM)),
                residual_scales=payload.get("residual_scales"),
                initial_std=float(cfg_get(config, "residual_initial_std", 0.3)),
                use_base_action=bool(cfg_get(config, "residual_use_base_action", False)),
            )
            self.residual.load_state_dict(payload["residual"])
            self.residual.to(device).eval()
            self.config = config
            self.normalization = Normalization.from_dict(payload["normalization"])
            self.eval_chunk_size = int(cfg_get(config, "eval_chunk_size", 1))
            self.std_min = float(cfg_get(config, "std_min", 0.05))
            self.std_max = float(cfg_get(config, "std_max", 0.5))
        else:
            self.base = build_act_from_payload(payload, device)
            self.residual = None
            self.config = payload.get("config", {})
            self.normalization = Normalization.from_dict(payload["normalization"])
            self.eval_chunk_size = int(cfg_get(self.config, "eval_chunk_size", self.base.chunk_size))
        self.image_size = int(payload.get("image_size", self.base.image_size))
        self.chunk_size = int(self.base.chunk_size)
        self.action_dim = int(self.base.action_dim)

    @torch.no_grad()
    def act_chunk_from_arrays(self, image: Any, proprio: Sequence[float] | np.ndarray) -> np.ndarray:
        image_tensor = prepare_image(image, self.image_size).unsqueeze(0).to(self.device)
        proprio_tensor = torch.as_tensor(np.asarray(proprio, dtype=np.float32)).unsqueeze(0).to(self.device)
        proprio_tensor = self.normalization.normalize_state(proprio_tensor)

        normalized_chunk = self.base.act_chunk(image_tensor, proprio_tensor)
        base_chunk = self.normalization.unnormalize_action(normalized_chunk)[0]
        if self.is_residual:
            _memory, features = self.base.encode_obs(image_tensor, proprio_tensor)
            residual = self.residual.deterministic_residual(features, base_chunk[0:1])[0]
            action = base_chunk[0] + residual
            actions = action.unsqueeze(0)
        else:
            actions = base_chunk[: self.eval_chunk_size]
        actions = normalize_quaternions_np(actions.cpu().numpy())
        actions[:, list(GRIPPER_INDICES)] = np.clip(actions[:, list(GRIPPER_INDICES)], 0.0, 1.0)
        return actions

    def act_chunk_from_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        image, proprio = observation_to_arrays(obs)
        return self.act_chunk_from_arrays(image, proprio)


def observation_to_arrays(obs: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
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
