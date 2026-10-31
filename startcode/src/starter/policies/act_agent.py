"""Residual PPO on a frozen ACT base policy.

The base ACT network is loaded from the offline checkpoint, put in eval mode and
frozen.  A small Gaussian head learns a bounded residual in real action units:
``a = a_base + scale * clamp(u)``, with ``u ~ N(mu_theta(obs), sigma^2)``.  PPO's
likelihood is computed on ``u`` only; the base never receives gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starter.rewards.intrinsic import combine_reward

from starter.policies.act_core import (
    ACTION_DIM,
    ActPolicy,
    Normalization,
    ResidualHead,
    cfg_get,
    normalize_quaternions_np,
)


GRIPPER_INDICES = (7, 15)


def _gaussian_logp(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    var = (std * std).clamp(min=1e-12)
    return (-0.5 * (((value - mean) ** 2) / var + torch.log(2.0 * math.pi * var))).sum(dim=-1)


def obs_to_uint8(obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    image = np.asarray(obs["image"], dtype=np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got {image.shape}.")
    image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()
    proprio = torch.as_tensor(np.asarray(obs["proprio"], dtype=np.float32).reshape(-1)[:ACTION_DIM])
    return image_tensor, proprio


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    return images.float().div(255.0).sub(0.5).div(0.5)


class ResidualCritic(nn.Module):
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass
class Transition:
    features: torch.Tensor
    base_action: torch.Tensor
    u: torch.Tensor
    logp: float
    reward: float
    value: float
    terminated: bool
    truncated: bool
    bootstrap: float = 0.0
    advantage: float = 0.0
    returns: float = 0.0


@dataclass
class EpisodeResult:
    score: float = 0.0
    success: bool = False
    steps: int = 0


class ActResidualAgent:
    def __init__(
        self,
        base: ActPolicy,
        residual: ResidualHead,
        critic: ResidualCritic,
        normalization: Normalization,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.base = base
        self.residual = residual
        self.critic = critic
        self.normalization = normalization
        self.config = dict(config)
        self.device = device

        self.gamma = float(cfg_get(config, "gamma", 0.99))
        self.gae_lambda = float(cfg_get(config, "gae_lambda", 0.95))
        self.clip_ratio = float(cfg_get(config, "clip_ratio", 0.2))
        self.ppo_epochs = int(cfg_get(config, "ppo_epochs", 4))
        self.minibatch_size = int(cfg_get(config, "minibatch_size", 64))
        self.value_coef = float(cfg_get(config, "value_coef", 0.5))
        self.entropy_coef = float(cfg_get(config, "entropy_coef", 1e-3))
        self.max_grad_norm = float(cfg_get(config, "max_grad_norm", 0.5))
        self.reward_scale = float(cfg_get(config, "reward_scale", 0.01))
        self.intrinsic_weight = float(cfg_get(config, "intrinsic_weight", 0.0))
        self.std_min = float(cfg_get(config, "std_min", 0.05))
        self.std_max = float(cfg_get(config, "std_max", 0.5))
        self.max_steps = int(cfg_get(config, "rollout_max_steps", 550))

        self.optimizer = torch.optim.AdamW(
            [
                {"params": list(self.residual.parameters()), "lr": float(cfg_get(config, "learning_rate", 3e-4))},
                {"params": list(self.critic.parameters()), "lr": float(cfg_get(config, "critic_learning_rate", 3e-4))},
            ],
            weight_decay=float(cfg_get(config, "weight_decay", 1e-4)),
        )
        self.global_steps = 0

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def _features(self, images: torch.Tensor, proprios: torch.Tensor) -> torch.Tensor:
        images = normalize_images(images.to(self.device))
        proprio = self.normalization.normalize_state(proprios.to(self.device))
        _memory, features = self.base.encode_obs(images, proprio)
        return features

    @torch.no_grad()
    def _base_chunks(self, images: torch.Tensor, proprios: torch.Tensor) -> torch.Tensor:
        images = normalize_images(images.to(self.device))
        proprio = self.normalization.normalize_state(proprios.to(self.device))
        return self.base.act_chunk(images, proprio)

    def _final_actions(self, base_actions: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # base chunks are cached on CPU to save GPU memory; move them next to the
        # residual head before combining.
        actions = base_actions.to(residual.device) + residual
        np_actions = normalize_quaternions_np(actions.cpu().numpy())
        np_actions[:, list(GRIPPER_INDICES)] = np.clip(np_actions[:, list(GRIPPER_INDICES)], 0.0, 1.0)
        return torch.from_numpy(np_actions).to(actions.device)

    def _intrinsic(
        self,
        privileged: Dict[str, Any],
        proprio: torch.Tensor,
        previous_potential: Optional[float],
    ) -> Tuple[float, Optional[float]]:
        if self.intrinsic_weight <= 0.0 or not privileged:
            return 0.0, previous_potential
        blocks = privileged.get("blocks") or {}
        positions: List[np.ndarray] = []
        for info in blocks.values():
            position = np.asarray(info.get("pos", []), dtype=np.float64).reshape(-1)
            if position.size >= 3:
                positions.append(position[:3])
        if not positions:
            return 0.0, previous_potential
        raw = proprio.detach().cpu().numpy().reshape(-1)
        end_effectors = [raw[0:3], raw[8:11]]
        distance = min(
            float(np.linalg.norm(effector - position))
            for effector in end_effectors
            for position in positions
        )
        potential = -distance
        if previous_potential is None:
            return 0.0, potential
        progress = max(min(potential - previous_potential, 1.0), -1.0)
        return self.intrinsic_weight * progress, potential

    # ----------------------------------------------------------------- rollout
    @torch.no_grad()
    def collect_wave(self, env: Any, seeds: Sequence[int]) -> Tuple[List[List[Transition]], List[EpisodeResult]]:
        observations, _infos = env.reset_batch([int(seed) for seed in seeds])
        width = len(observations)
        observations = [obs_to_uint8(obs) for obs in observations]
        trajectories: List[List[Transition]] = [[] for _ in range(width)]
        episodes = [EpisodeResult() for _ in range(width)]
        previous_potential: List[Optional[float]] = [None] * width
        chunks: List[Optional[torch.Tensor]] = [None] * width
        chunk_indices = [0] * width
        active = list(range(width))
        step_index = 0

        while active and step_index < self.max_steps:
            need_chunk = [
                index
                for index in active
                if chunks[index] is None or chunk_indices[index] >= chunks[index].shape[0]
            ]
            if need_chunk:
                images = torch.stack([observations[index][0] for index in need_chunk])
                proprios = torch.stack([observations[index][1] for index in need_chunk])
                new_chunks = self._base_chunks(images, proprios).cpu()
                for position, index in enumerate(need_chunk):
                    chunks[index] = new_chunks[position]
                    chunk_indices[index] = 0

            images = torch.stack([observations[index][0] for index in active])
            proprios = torch.stack([observations[index][1] for index in active])
            features = self._features(images, proprios)
            base_norm = torch.stack(
                [chunks[index][chunk_indices[index]] for index in active], dim=0
            )
            base_actions = self.normalization.unnormalize_action(base_norm).to(features.device)
            mean, std, u = self.residual.sample(features, base_actions, self.std_min, self.std_max)
            residual = self.residual.residual(u)
            logp = _gaussian_logp(u, mean, std)
            values = self.critic(features)
            env_actions = self._final_actions(base_actions, residual)

            results = env.step_batch(env_actions.cpu().tolist(), active)
            still_active: List[int] = []
            for batch_index, env_index in enumerate(active):
                observation, score, terminated, truncated, info = results[batch_index]
                done = bool(terminated or truncated)
                privileged = {}
                if self.intrinsic_weight > 0.0:
                    try:
                        privileged = env.privileged(env_index)
                    except Exception:
                        privileged = {}
                intrinsic, previous_potential[env_index] = self._intrinsic(
                    privileged, observations[env_index][1], previous_potential[env_index]
                )
                reward = combine_reward(float(score), intrinsic) * self.reward_scale
                bootstrap = 0.0
                if truncated and not terminated:
                    next_image, next_proprio = obs_to_uint8(observation)
                    next_features = self._features(
                        next_image.unsqueeze(0), next_proprio.unsqueeze(0)
                    )
                    bootstrap = float(self.critic(next_features).item())

                trajectories[env_index].append(
                    Transition(
                        features=features[batch_index].cpu(),
                        base_action=base_actions[batch_index].cpu(),
                        u=u[batch_index].cpu(),
                        logp=float(logp[batch_index].item()),
                        reward=float(reward),
                        value=float(values[batch_index].item()),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        bootstrap=bootstrap,
                    )
                )
                episodes[env_index].score = max(episodes[env_index].score, float(score))
                episodes[env_index].success = episodes[env_index].success or bool(
                    terminated or info.get("success", False)
                )
                episodes[env_index].steps += 1

                if done:
                    continue
                chunk_indices[env_index] += 1
                observations[env_index] = obs_to_uint8(observation)
                still_active.append(env_index)

            active = still_active
            step_index += 1

        if active:
            for env_index in active:
                if trajectories[env_index]:
                    last = trajectories[env_index][-1]
                    last.truncated = True
                    image, proprio = observations[env_index]
                    last.bootstrap = float(
                        self.critic(self._features(image.unsqueeze(0), proprio.unsqueeze(0))).item()
                    )
        return trajectories, episodes

    # --------------------------------------------------------------------- PPO
    def compute_gae(self, trajectories: Sequence[Sequence[Transition]]) -> None:
        for trajectory in trajectories:
            advantage = 0.0
            for index in range(len(trajectory) - 1, -1, -1):
                transition = trajectory[index]
                next_value = (
                    transition.bootstrap
                    if index == len(trajectory) - 1
                    else trajectory[index + 1].value
                )
                mask = 0.0 if transition.terminated else 1.0
                delta = transition.reward + self.gamma * next_value * mask - transition.value
                advantage = delta + self.gamma * self.gae_lambda * advantage * mask
                transition.advantage = advantage
                transition.returns = advantage + transition.value

    def update(self, trajectories: Sequence[Sequence[Transition]]) -> Dict[str, float]:
        self.compute_gae(trajectories)
        transitions = [transition for trajectory in trajectories for transition in trajectory]
        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "ratio": 1.0}

        features = torch.stack([item.features for item in transitions], dim=0)
        values_target = torch.tensor([item.returns for item in transitions], dtype=torch.float32)
        advantages = torch.tensor([item.advantage for item in transitions], dtype=torch.float32)
        base_actions = torch.stack([item.base_action for item in transitions], dim=0)
        u_target = torch.stack([item.u for item in transitions], dim=0)
        old_logp = torch.tensor([item.logp for item in transitions], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        count = len(transitions)
        policy_losses: List[float] = []
        value_losses: List[float] = []
        ratios: List[float] = []
        entropies: List[float] = []

        for _epoch in range(self.ppo_epochs):
            permutation = torch.randperm(count)
            for start in range(0, count, self.minibatch_size):
                indices = permutation[start : start + self.minibatch_size]
                batch_features = features[indices].to(self.device)
                mean = self.residual.forward(batch_features, base_actions[indices].to(self.device))
                std = self.residual.std(self.std_min, self.std_max).expand_as(mean)
                new_logp = _gaussian_logp(u_target[indices].to(self.device), mean, std)
                old = old_logp[indices].to(self.device)
                ratio = (new_logp - old).clamp(min=-20.0, max=20.0).exp()
                batch_adv = advantages[indices].to(self.device)
                unclipped = ratio * batch_adv
                clipped = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                entropy = (0.5 * math.log(2.0 * math.pi * math.e) + torch.log(std)).sum(dim=-1).mean()

                values = self.critic(batch_features)
                value_loss = F.mse_loss(values, values_target[indices].to(self.device))
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.residual.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.global_steps += 1

                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                ratios.append(float(ratio.detach().mean().cpu()))
                entropies.append(float(entropy.detach().cpu()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "ratio": float(np.mean(ratios)),
            "entropy": float(np.mean(entropies)),
            "transitions": float(len(transitions)),
            "mean_return": float(np.mean([sum(item.reward for item in trajectory) for trajectory in trajectories])),
        }
