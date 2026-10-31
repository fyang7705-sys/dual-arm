"""DPPO: PPO over the last stochastic denoising steps of a Diffusion Policy.

The offline diffusion policy treats a whole action chunk as one sample.  For
online RL we keep the DDIM sampler but make the last ``stochastic_steps``
transitions Gaussian.  Each environment step therefore produces a short chain
of Gaussian "actions"; PPO optimizes the chain log-probability, discounting
earlier denoising steps because they are further from the executed action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starter.rewards.intrinsic import combine_reward

from starter.policies.dp_core import (
    ACTION_DIM,
    Critic,
    normalize_quaternions,
    DiffusionPolicy,
    EMA,
    Normalization,
    cfg_get,
    gaussian_logp,
)


GRIPPER_INDICES = (7, 15)


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def obs_to_uint8(obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    image = np.asarray(obs["image"], dtype=np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got {image.shape}.")
    image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()
    proprio = torch.as_tensor(np.asarray(obs["proprio"], dtype=np.float32).reshape(-1)[:ACTION_DIM])
    return image_tensor, proprio


def stack_observations(observations: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([item[0] for item in observations], dim=0)
    proprios = torch.stack([item[1] for item in observations], dim=0)
    return images, proprios


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """uint8 (B,3,S,S) -> float normalized (B,3,S,S)."""

    return images.float().div(255.0).sub(0.5).div(0.5)


@dataclass
class Transition:
    image: torch.Tensor
    proprio: torch.Tensor
    chain: List[Dict[str, Any]]
    action: torch.Tensor
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
    reward: float = 0.0


class DPPOAgent:
    def __init__(
        self,
        model: DiffusionPolicy,
        ema: EMA,
        critic: Critic,
        normalization: Normalization,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.ema = ema
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
        self.max_grad_norm = float(cfg_get(config, "max_grad_norm", 0.5))
        self.gamma_diff = float(cfg_get(config, "gamma_diff", 0.9))
        self.reward_scale = float(cfg_get(config, "reward_scale", 0.01))
        self.intrinsic_weight = float(cfg_get(config, "intrinsic_weight", 0.0))
        self.sampler_steps = int(cfg_get(config, "sampler_steps", 8))
        self.stochastic_steps = int(cfg_get(config, "stochastic_steps", 4))
        self.eta = float(cfg_get(config, "eta", 0.8))
        self.std_min = float(cfg_get(config, "std_min", 0.02))
        self.std_max = float(cfg_get(config, "std_max", 0.4))
        self.max_steps = int(cfg_get(config, "rollout_max_steps", 550))

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.critic.parameters()),
            lr=float(cfg_get(config, "learning_rate", 5e-5)),
            weight_decay=float(cfg_get(config, "weight_decay", 1e-4)),
        )
        self.global_steps = 0

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act_batch(
        self,
        images: torch.Tensor,
        proprios: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]], torch.Tensor]:
        """Return normalized actions, per-transition denoising chains, values."""

        normalized_images = normalize_images(images.to(self.device))
        proprios = self.normalization.normalize_state(proprios.to(self.device))
        actions, chain = self.model.sample(
            normalized_images,
            proprios,
            sample_steps=self.sampler_steps,
            eta=0.0 if deterministic else self.eta,
            stochastic_steps=0 if deterministic else self.stochastic_steps,
            std_min=self.std_min,
            std_max=self.std_max,
            return_chain=True,
        )
        values = self.critic(normalized_images, proprios)
        per_sample_chain: List[List[Dict[str, Any]]] = [[] for _ in range(images.shape[0])]
        for step in chain:
            for index in range(images.shape[0]):
                per_sample_chain[index].append(
                    {
                        "t": int(step["t"]),
                        "t_prev": int(step["t_prev"]),
                        "sigma": float(step["sigma"]),
                        "x_t": step["x_t"][index].clone(),
                        "x_prev": step["x_prev"][index].clone(),
                        "logp": float(step["logp"][index].item()),
                    }
                )
        return actions.cpu(), per_sample_chain, values.detach().cpu()

    def to_env_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """Unnormalize a sampled chunk and execute its first action (receding horizon)."""

        chunk = normalized_actions.cpu().reshape(
            normalized_actions.shape[0], self.model.chunk_size, self.model.action_dim
        )
        actions = self.normalization.unnormalize_action(chunk)[:, 0, :]
        actions = normalize_quaternions(actions)
        actions[:, list(GRIPPER_INDICES)] = actions[:, list(GRIPPER_INDICES)].clamp(0.0, 1.0)
        return actions

    # --------------------------------------------------------------- rollout
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
        proprio_np = proprio.detach().cpu().numpy().reshape(-1)
        end_effectors = [proprio_np[0:3], proprio_np[8:11]]
        min_distance = min(
            float(np.linalg.norm(effector - position))
            for effector in end_effectors
            for position in positions
        )
        potential = -min_distance
        if previous_potential is None:
            return 0.0, potential
        progress = max(min(potential - previous_potential, 1.0), -1.0)
        return self.intrinsic_weight * progress, potential

    @torch.no_grad()
    def collect_wave(self, env: Any, seeds: Sequence[int]) -> Tuple[List[List[Transition]], List[EpisodeResult]]:
        """Run one wave of ``len(seeds)`` episodes and return per-env trajectories."""

        observations, _infos = env.reset_batch([int(seed) for seed in seeds])
        width = len(observations)
        observations = [obs_to_uint8(obs) for obs in observations]
        trajectories: List[List[Transition]] = [[] for _ in range(width)]
        episodes = [EpisodeResult() for _ in range(width)]
        previous_potential: List[Optional[float]] = [None] * width
        active = list(range(width))
        step_index = 0

        while active and step_index < self.max_steps:
            images, proprios = stack_observations([observations[index] for index in active])
            normalized_actions, chains, values = self.act_batch(images, proprios)
            env_actions = self.to_env_actions(normalized_actions)
            results = env.step_batch(env_actions.tolist(), active)

            still_active: List[int] = []
            for batch_index, env_index in enumerate(active):
                observation, score, terminated, truncated, info = results[batch_index]
                image, proprio = observations[env_index]
                done = bool(terminated or truncated)
                privileged: Dict[str, Any] = {}
                if self.intrinsic_weight > 0.0:
                    try:
                        privileged = env.privileged(env_index)
                    except Exception:
                        privileged = {}
                intrinsic, previous_potential[env_index] = self._intrinsic(
                    privileged, proprio, previous_potential[env_index]
                )
                reward = combine_reward(float(score), intrinsic) * self.reward_scale
                bootstrap = 0.0
                if truncated and not terminated:
                    next_image, next_proprio = obs_to_uint8(observation)
                    bootstrap = float(
                        self.critic(
                            normalize_images(next_image.unsqueeze(0)).to(self.device),
                            self.normalization.normalize_state(next_proprio.unsqueeze(0).to(self.device)),
                        ).item()
                    )

                trajectories[env_index].append(
                    Transition(
                        image=image,
                        proprio=proprio,
                        chain=chains[batch_index],
                        action=normalized_actions[batch_index],
                        reward=float(reward),
                        value=float(values[batch_index]),
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
                episodes[env_index].reward += float(score)

                if done:
                    continue
                observations[env_index] = obs_to_uint8(observation)
                still_active.append(env_index)

            active = still_active
            step_index += 1

        # If the safety horizon stopped a still-running episode, bootstrap it.
        if active:
            for env_index in active:
                if trajectories[env_index]:
                    last = trajectories[env_index][-1]
                    last.truncated = True
                    image, proprio = observations[env_index]
                    bootstrap = float(
                        self.critic(
                            normalize_images(image.unsqueeze(0)),
                            self.normalization.normalize_state(proprio.unsqueeze(0)),
                        ).item()
                    )
                    last.bootstrap = bootstrap
        return trajectories, episodes

    # ------------------------------------------------------------------ PPO
    def compute_gae(self, trajectories: Sequence[Sequence[Transition]]) -> None:
        for trajectory in trajectories:
            advantage = 0.0
            for index in range(len(trajectory) - 1, -1, -1):
                transition = trajectory[index]
                if index == len(trajectory) - 1:
                    next_value = transition.bootstrap
                else:
                    next_value = trajectory[index + 1].value
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

        images = torch.stack([item.image for item in transitions], dim=0)
        proprios = torch.stack([item.proprio for item in transitions], dim=0)
        advantages = torch.tensor([item.advantage for item in transitions], dtype=torch.float32)
        returns = torch.tensor([item.returns for item in transitions], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        chain_length = min(len(item.chain) for item in transitions)
        if chain_length == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "ratio": 1.0}
        x_t = torch.stack(
            [torch.stack([step["x_t"] for step in item.chain[:chain_length]]) for item in transitions]
        )
        x_prev = torch.stack(
            [torch.stack([step["x_prev"] for step in item.chain[:chain_length]]) for item in transitions]
        )
        old_logp = torch.tensor(
            [[step["logp"] for step in item.chain[:chain_length]] for item in transitions],
            dtype=torch.float32,
        )
        timesteps = torch.tensor(
            [[step["t"] for step in item.chain[:chain_length]] for item in transitions],
            dtype=torch.long,
        )
        prev_timesteps = torch.tensor(
            [[step["t_prev"] for step in item.chain[:chain_length]] for item in transitions],
            dtype=torch.long,
        )
        sigmas = torch.tensor(
            [step["sigma"] for step in transitions[0].chain[:chain_length]], dtype=torch.float32
        )
        step_weights = torch.tensor(
            [self.gamma_diff ** (chain_length - 1 - index) for index in range(chain_length)],
            dtype=torch.float32,
        )

        count = len(transitions)
        policy_losses: List[float] = []
        value_losses: List[float] = []
        ratios: List[float] = []

        for _epoch in range(self.ppo_epochs):
            permutation = torch.randperm(count)
            for start in range(0, count, self.minibatch_size):
                indices = permutation[start : start + self.minibatch_size]
                batch_images = normalize_images(images[indices]).to(self.device)
                batch_props = self.normalization.normalize_state(proprios[indices].to(self.device))
                context = self.model.encode_obs(batch_images, batch_props)

                batch = indices.numel()
                context_repeated = context.unsqueeze(1).expand(batch, chain_length, -1).reshape(
                    batch * chain_length, -1
                )
                flat_x = x_t[indices].reshape(batch * chain_length, -1).to(self.device)
                flat_prev = x_prev[indices].reshape(batch * chain_length, -1).to(self.device)
                flat_t = timesteps[indices].reshape(-1).to(self.device)
                flat_t_prev = prev_timesteps[indices].reshape(-1).to(self.device)
                flat_sigma = sigmas.unsqueeze(0).expand(batch, chain_length).reshape(-1).to(self.device)

                new_logp = self.model.chain_logp(
                    flat_x, flat_prev, context_repeated, flat_t, flat_t_prev, flat_sigma
                ).reshape(batch, chain_length)
                old = old_logp[indices].to(self.device)
                log_ratio = (new_logp - old).clamp(min=-20.0, max=20.0)
                ratio = log_ratio.exp()
                batch_adv = advantages[indices].to(self.device).unsqueeze(1) * step_weights.unsqueeze(0).to(self.device)
                unclipped = ratio * batch_adv
                clipped = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                values = self.critic(batch_images, batch_props)
                value_loss = F.mse_loss(values, returns[indices].to(self.device))
                loss = policy_loss + self.value_coef * value_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.ema.update(self.model)
                self.global_steps += 1

                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                ratios.append(float(ratio.detach().mean().cpu()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "ratio": float(np.mean(ratios)) if ratios else 1.0,
            "transitions": float(len(transitions)),
            "mean_return": float(np.mean([sum(item.reward for item in trajectory) for trajectory in trajectories])),
        }
