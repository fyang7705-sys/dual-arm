"""PPO over ACT residual latents or the final diffusion denoising transitions.

    Complete episode waves, bounded RAM buffers, semi-Markov GAE, timeout
    bootstrapping, clipped policy/value objectives, KL stop and checkpoint resume.
"""
from __future__ import annotations

import numpy as np
import torch

from starter.envs.remote import RemoteEnv
from starter.interfaces import ACTION_DIM
from starter.policies.models import Critic, FeatureCritic, ResidualActor
from starter.policies.runtime import (FORMAT, FrameHistory, append_metrics, assert_same_state, build_model,
                                      clone_state, load_payload, observation_arrays, observation_image,
                                      pick_device, project_actions, restore_rng, rng_state, run_directory,
                                      save_checkpoint, seed_everything, tensor_observations)


def gae(records, gamma, lam):
    """Each record spans n physics-control actions; boundaries never leak GAE.

    Success/invalid terminal => bootstrap zero. Timeout/collector cutoff =>
    bootstrap V(last_obs), but stop the trace when resetting the environment.
    """
    carry = 0.0
    for item in reversed(records):
        discount = gamma ** item["duration"]
        delta = item["reward"] + discount * item["next_value"] - item["value"]
        carry = delta + (0 if item["boundary"] else (gamma * lam) ** item["duration"] * carry)
        item["advantage"] = carry
        item["return"] = carry + item["value"]


def clipped_objective(logprob, old_logprob, advantage, clip):
    logratio = (logprob - old_logprob).clamp(-20, 20)
    ratio = logratio.exp()
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    approximate_kl = ((ratio - 1) - logratio).mean()
    return -objective.mean(), approximate_kl


def action_mask(durations, steps, device):
    """Mark action-chunk positions that were actually executed.

    ``execute_steps`` is only an upper bound: an episode can end in the middle
    of a chunk.  Un-executed positions must not enter the PPO likelihood.
    """
    positions = torch.arange(steps, device=device).unsqueeze(0)
    return (positions < durations.to(device).clamp_max(steps).unsqueeze(1))[..., None]


def prefix_step_logprob(dist, value, steps):
    """Per-action log-likelihood of the executed action prefix.

    Returned with shape ``[batch, steps]`` so the update can mask the steps
    that were cut off when an episode ended inside the chunk.  Both the stored
    behaviour log-probability and the recomputed one use the same mask; that
    keeps the PPO ratio exactly one for an unchanged policy.
    """
    return dist.log_prob(value)[..., :steps, :].sum(-1)


def masked_step_sum(values, durations):
    return (values * action_mask(durations, values.shape[1], values.device).squeeze(-1)).sum(-1)


def prefix_entropy(dist, steps, durations=None):
    entropy = dist.entropy()[..., :steps, :]
    if durations is not None:
        mask = action_mask(durations, steps, entropy.device).to(entropy.dtype)
        return (entropy * mask).sum() / mask.sum().clamp_min(1)
    return entropy.mean()


class Learner:
    def __init__(self, kind, config, payload):
        self.kind, self.config, self.payload = kind, config, payload
        self.device, self.cfg, self.stats = pick_device(config), dict(payload["model_config"]), payload["stats"]
        self.model = build_model(kind, self.cfg).to(self.device)
        self.model.load_state_dict(payload.get("ema") or payload["model"])
        self.model.eval()  # dropout is zero; likelihood replay must match sampling
        self.execute_steps = int(self.cfg["execute_steps"])
        self.obs_steps = int(self.cfg.get("obs_steps", 1))
        self.history = FrameHistory(self.obs_steps, self.cfg["image_size"]) if self.obs_steps > 1 else None
        self.base_snapshot = None
        self.diffusion_base = None
        if kind == "act":
            self.model.requires_grad_(False)
            self.base_snapshot = clone_state(self.model)
            self.cfg["residual_std"] = float(config.get("residual_std", 0.3))
            self.actor = ResidualActor(self.cfg).to(self.device)
            if payload.get("residual") is not None:
                self.actor.load_state_dict(payload["residual"])
        else:
            self.actor = self.model
            # The deterministic prefix must stay frozen while PPO optimizes
            # likelihoods only for the suffix transitions.
            self.diffusion_base = build_model(kind, self.cfg).to(self.device).eval()
            self.diffusion_base.load_state_dict(payload.get("diffusion_base") or payload.get("ema") or payload["model"])
            self.diffusion_base.requires_grad_(False)
            # The ImageNet trunk is shared with the frozen prefix and has far
            # more parameters than an online update can estimate.  Freezing it
            # keeps the optimizer parameter list (and resume state) unchanged
            # while avoiding sparse-reward overfitting of the vision features.
            freeze = bool(config.get("freeze_online_backbone", True))
            if payload.get("stage") == "online":
                freeze = bool(payload.get("config", {}).get("freeze_online_backbone", freeze))
            for parameter in self.model.backbone_parameters():
                parameter.requires_grad_(not freeze)
        # Reuse the frozen policy encoder for the value head.  Old online
        # checkpoints carry a standalone Critic whose keys start with
        # "condition."; detect that and keep loading them unchanged.
        critic_keys = payload.get("critic") or {}
        shared = not any(key.startswith("condition.") for key in critic_keys)
        if payload["stage"] != "online":
            shared = bool(config.get("critic_shared", True))
        elif any(key.startswith("condition.") for key in critic_keys):
            shared = False
        else:
            shared = bool(payload.get("config", {}).get("critic_shared", True))
        self.critic_shared = shared
        self.critic = (FeatureCritic(int(self.cfg["width"])) if shared
                       else Critic(self.cfg)).to(self.device)
        # Fresh residual-actor towers start from ImageNet weights; the loaded
        # policy model itself is never re-initialized.
        if self.config.get("pretrained_backbone", True):
            for module in (self.actor, self.critic):
                if module is self.model:
                    continue
                initialize = getattr(module, "initialize_pretrained_backbone", None)
                if initialize is not None:
                    initialize()
        self.scale = float(payload.get("residual_scale", config.get("residual_scale", 0.1)))
        self.stochastic_steps = int(payload.get("finetune_denoising_steps", config.get("finetune_denoising_steps", 4)))
        if kind == "diffusion" and not 1 <= self.stochastic_steps <= len(self.model.schedule):
            raise ValueError("finetune_denoising_steps must be in [1, inference chain length]")
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(config.get("actor_lr", 1e-5)))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(config.get("critic_lr", 1e-4)))
        if payload["stage"] == "online":
            self.critic.load_state_dict(payload["critic"])
            self.actor_opt.load_state_dict(payload["actor_optimizer"])
            self.critic_opt.load_state_dict(payload["critic_optimizer"])

    def _critic_value(self, image, state):
        if not self.critic_shared:
            return self.critic(image, state)
        if self.kind == "act":
            features = self.model(image, state, return_features=True)[2]
        else:
            features = self.diffusion_base.condition_features(image, state)
        return self.critic(features)

    def reset_history(self):
        """Drop per-episode frame history; called before every episode wave."""
        if self.history is not None:
            self.history.reset()

    def observe(self, observations, slots=None):
        """Advance the multi-frame history at control rate without predicting."""
        if self.history is None:
            return
        for index, observation in enumerate(observations):
            self.history.push(slots[index] if slots is not None else index,
                              observation_image(observation))

    @torch.no_grad()
    def sample(self, observations, slots=None):
        images, states = observation_arrays(observations, self.stats, self.cfg["image_size"],
                                            history=self.history, slots=slots)
        image, state = tensor_observations(images, states, self.device)
        if self.kind == "act" and self.critic_shared:
            base, _, features = self.model(image, state, return_features=True)
            values = self.critic(features).cpu().numpy()
        else:
            values = self._critic_value(image, state).cpu().numpy()
        if self.kind == "act":
            base = self.model(image, state)[0] if not self.critic_shared else base
            dist = self.actor.distribution(image, state, base)
            latent = dist.sample()
            action = base + self.scale * latent.tanh()
            traces = [{"base": base[i].cpu(), "latent": latent[i].cpu(),
                       "old_logprob": prefix_step_logprob(dist, latent, self.execute_steps)[i].cpu()}
                      for i in range(len(state))]
        else:
            action, chain = self.model.sample(image, state, self.stochastic_steps,
                                              float(self.config.get("denoising_min_std", 0.05)),
                                              prefix_model=self.diffusion_base,
                                              prefix_steps=self.execute_steps)
            traces = [{"x": torch.stack([step[0][i].cpu() for step in chain]),
                       "next_x": torch.stack([step[1][i].cpu() for step in chain]),
                       "time": torch.stack([step[2][i].cpu() for step in chain]),
                       "old_logprob": torch.stack([step[3][i].cpu() for step in chain]),
                       "prev_time": torch.stack([step[4][i].cpu() for step in chain])}
                      for i in range(len(state))]
        action = action.cpu().numpy() * np.asarray(self.stats["action_std"]) + np.asarray(self.stats["action_mean"])
        fallback = np.asarray([o["proprio"] for o in observations])[:, None, :]
        return project_actions(action, fallback), images, states, values, traces

    @torch.no_grad()
    def values(self, observations, slots=None):
        arrays = observation_arrays(observations, self.stats, self.cfg["image_size"],
                                    history=self.history, slots=slots)
        return self._critic_value(*tensor_observations(*arrays, self.device)).cpu().numpy()

    def update(self, records):
        cfg, device = self.config, self.device
        advantages = torch.tensor([r["advantage"] for r in records], device=device, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        returns = torch.tensor([r["return"] for r in records], device=device, dtype=torch.float32)
        old_values = torch.tensor([r["value"] for r in records], device=device, dtype=torch.float32)
        durations = torch.tensor([min(int(r["duration"]), self.execute_steps) for r in records],
                                 device=device, dtype=torch.float32)
        metrics = {"policy_loss": [], "value_loss": [], "approx_kl": [], "entropy": [], "anchor_loss": []}
        batch_size = int(cfg.get("ppo_batch_size", 32))
        value_clip = float(cfg.get("value_clip", 0.2))
        for _ in range(int(cfg.get("ppo_epochs", 4))):
            stop = False
            for indices in torch.randperm(len(records)).split(batch_size):
                rows = [records[i] for i in indices.tolist()]
                ids = indices.to(device)
                image, state = tensor_observations(np.stack([r["image"] for r in rows]),
                                                   np.stack([r["state"] for r in rows]), device)
                trace = {key: torch.stack([r["trace"][key] for r in rows]).to(device) for key in rows[0]["trace"]}
                anchor = torch.zeros((), device=device)
                if self.kind == "act":
                    dist = self.actor.distribution(image, state, trace["base"])
                    new_steps = prefix_step_logprob(dist, trace["latent"], self.execute_steps)
                    logprob = masked_step_sum(new_steps, durations[ids])
                    old_logprob = masked_step_sum(trace["old_logprob"], durations[ids])
                    entropy = prefix_entropy(dist, self.execute_steps, durations[ids])
                    policy_loss, kl = clipped_objective(logprob, old_logprob, advantages[ids],
                                                       float(cfg.get("ppo_clip", 0.2)))
                    anchor_coef = float(cfg.get("residual_l2", 0.0))
                    if anchor_coef > 0:
                        # Pull the residual toward the frozen base; useful when
                        # sparse rewards give no signal for long stretches.
                        anchor = dist.mean[..., :self.execute_steps, :].square().mean()
                        actor_loss = policy_loss - float(cfg.get("entropy_coef", 0.0)) * entropy + anchor_coef * anchor
                    else:
                        actor_loss = policy_loss - float(cfg.get("entropy_coef", 0.0)) * entropy
                else:
                    k = trace["time"].shape[1]
                    # Each Gaussian transition is a PPO action in the inner MDP.
                    # Earlier inner steps get smaller credit; no gradients through
                    # stored x_t or through the sampling chain itself.
                    repeated_image = image.repeat_interleave(k, 0)
                    repeated_state = state.repeat_interleave(k, 0)
                    repeated_durations = durations[ids].repeat_interleave(k)
                    prev_time = trace["prev_time"].flatten()
                    dist = self.model.distribution(trace["x"].flatten(0, 1), trace["time"].flatten(),
                                                   repeated_image, repeated_state,
                                                   float(cfg.get("denoising_min_std", 0.05)),
                                                   prev_time=prev_time)
                    new_steps = prefix_step_logprob(dist, trace["next_x"].flatten(0, 1), self.execute_steps)
                    logprob = masked_step_sum(new_steps, repeated_durations)
                    old_logprob = masked_step_sum(trace["old_logprob"].flatten(0, 1), repeated_durations)
                    weights = float(cfg.get("denoising_gamma", 0.99)) ** trace["time"].float()
                    advantage = (advantages[ids, None] * weights).flatten()
                    policy_loss, kl = clipped_objective(logprob, old_logprob, advantage,
                                                       float(cfg.get("ppo_clip", 0.2)))
                    entropy = prefix_entropy(dist, self.execute_steps, repeated_durations)
                    anchor_coef = float(cfg.get("suffix_l2", 0.0))
                    if anchor_coef > 0:
                        # Keep the trained suffix close to the frozen prefix.
                        base_dist = self.diffusion_base.distribution(
                            trace["x"].flatten(0, 1), trace["time"].flatten(),
                            repeated_image, repeated_state,
                            float(cfg.get("denoising_min_std", 0.05)),
                            prev_time=prev_time,
                        )
                        anchor = (dist.mean - base_dist.mean)[..., :self.execute_steps, :].square().mean()
                    actor_loss = policy_loss - float(cfg.get("entropy_coef", 0.0)) * entropy + anchor_coef * anchor
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Non-finite PPO actor loss")
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(cfg.get("max_grad_norm", 1)))
                self.actor_opt.step()
                value = self._critic_value(image, state)
                clipped_value = old_values[ids] + (value - old_values[ids]).clamp(-value_clip, value_clip)
                value_loss = 0.5 * torch.maximum((value - returns[ids]).square(),
                                                  (clipped_value - returns[ids]).square()).mean()
                if not torch.isfinite(value_loss):
                    raise FloatingPointError("Non-finite PPO critic loss")
                self.critic_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(cfg.get("max_grad_norm", 1)))
                self.critic_opt.step()
                for key, val in zip(metrics, (policy_loss, value_loss, kl, entropy, anchor)):
                    metrics[key].append(float(val.detach()))
                if float(kl.detach()) > float(cfg.get("target_kl", 0.03)):
                    stop = True
                    break
            if stop:
                break
        if self.base_snapshot is not None:
            assert_same_state(self.model, self.base_snapshot)
        return {key: float(np.mean(values)) for key, values in metrics.items()}


def collect(learner, env, config, update_index):
    width, episodes = int(config.get("num_envs", 1)), int(config.get("finetune_episodes", 4))
    if width < 1 or episodes < 1 or episodes % width:
        raise ValueError("finetune_episodes must be a positive multiple of num_envs")
    gamma, lam = float(config.get("gamma", 0.99)), float(config.get("gae_lambda", 0.95))
    records, episode_metrics = [], []
    execute = int(learner.cfg["execute_steps"])
    cap = int(config.get("max_decisions_per_episode", 550))
    if cap < 1:
        raise ValueError("max_decisions_per_episode must be positive")
    for wave in range(episodes // width):
        seeds = [int(config.get("seed", 0)) + update_index * episodes + wave * width + i for i in range(width)]
        observations, _ = env.reset_batch(seeds)
        learner.reset_history()
        learner.observe(observations, list(range(width)))
        active = list(range(width))
        per_env = [[] for _ in range(width)]
        raw_returns, successes = [0.0] * width, [False] * width
        last_scores, frames = [0.0] * width, [0] * width
        for decision in range(cap):
            selected = [observations[i] for i in active]
            actions, images, states, values, traces = learner.sample(selected, slots=active)
            pending = {i: {"image": images[j], "state": states[j], "value": float(values[j]),
                           "trace": traces[j], "reward": 0.0, "duration": 0,
                           "boundary": False, "terminal": False} for j, i in enumerate(active)}
            action_lookup = {i: actions[j] for j, i in enumerate(active)}
            running = list(active)
            for step in range(execute):
                results = env.step_batch([action_lookup[i][step].tolist() for i in running], running)
                keep = []
                for i, result in zip(running, results):
                    obs, reward, terminated, truncated, info = result
                    observations[i] = obs
                    item = pending[i]
                    # The environment provides the unchanged binary task reward.
                    item["reward"] += gamma ** item["duration"] * float(reward)
                    item["duration"] += 1
                    item["boundary"] = bool(terminated or truncated)
                    item["terminal"] = bool(terminated)
                    raw_returns[i] += float(reward)
                    frames[i] += 1
                    successes[i] = successes[i] or bool(info.get("success", False))
                    last_scores[i] = float(info.get("external_score", 0))
                    if not item["boundary"]:
                        keep.append(i)
                # Advance multi-frame history at control rate, including the
                # terminal observation used for timeout bootstrapping.
                learner.observe([observations[i] for i in running], list(running))
                running = keep
                if not running:
                    break
            next_values = learner.values([observations[i] for i in active], slots=active)
            for j, i in enumerate(active):
                item = pending[i]
                item["next_value"] = 0.0 if item["terminal"] else float(next_values[j])
                item["boundary"] = item["boundary"] or decision + 1 == cap
                per_env[i].append(item)
            active = running
            if not active:
                break
        for i in range(width):
            gae(per_env[i], gamma, lam)
            records.extend(per_env[i])
            episode_metrics.append({"seed": seeds[i], "return": raw_returns[i], "success": successes[i],
                                    "score": last_scores[i], "steps": frames[i],
                                    "collector_cutoff": not per_env[i][-1]["terminal"] and i in active})
    return records, episode_metrics


def train(kind, config):
    if config.get("env_backend", "remote") not in ("remote", "fake"):
        raise ValueError("env_backend must be remote or fake")
    seed_everything(int(config.get("seed", 0)))
    source = config.get("resume") or config.get("offline_checkpoint")
    if not source:
        raise ValueError("Online training requires offline_checkpoint (or resume); random initialization is forbidden")
    payload = load_payload(source)
    if payload["kind"] != kind:
        raise ValueError("Checkpoint policy kind mismatch")
    if config.get("resume") and payload["stage"] != "online":
        raise ValueError("resume must refer to an online checkpoint")
    if not config.get("resume") and payload["stage"] != "offline":
        raise ValueError("offline_checkpoint must refer to an offline checkpoint")
    if int(config.get("ppo_batch_size", 32)) < 1 or int(config.get("ppo_epochs", 4)) < 1:
        raise ValueError("PPO batch size and epochs must be positive")
    learner = Learner(kind, config, payload)
    if payload["stage"] == "online":
        restore_rng(payload.get("rng"))
    start = int(payload["update"]) + 1 if payload["stage"] == "online" else 0
    updates = int(config.get("finetune_updates", 100))
    if updates <= start:
        raise ValueError("finetune_updates must exceed the completed resume update")
    directory = run_directory(config, kind, "online")
    if config.get("env_backend", "remote") == "fake":
        from starter.envs.fake import ToyBatchEnv
        env = ToyBatchEnv(int(config.get("num_envs", 1)), int(config.get("fake_episode_steps", 4)))
    else:
        env = RemoteEnv(str(config.get("env_host", "127.0.0.1")), int(config.get("env_port", 5555)),
                        timeout=float(config.get("env_timeout", 2400)),
                        reset_timeout=float(config.get("reset_timeout", 10800)))
    try:
        for update in range(start, updates):
            records, episodes = collect(learner, env, config, update)
            metrics = learner.update(records)
            metrics.update({"update": update + 1, "decisions": len(records),
                            "return_mean": float(np.mean([e["return"] for e in episodes])),
                            "success_rate": float(np.mean([e["success"] for e in episodes])),
                            "benchmark_score": float(np.mean([e["score"] for e in episodes])),
                            "env_backend": config.get("env_backend", "remote")})
            append_metrics(directory, metrics)
            with (directory / "episodes.jsonl").open("a", encoding="utf-8") as f:
                import json
                for episode in episodes:
                    f.write(json.dumps({"update": update + 1, **episode}) + "\n")
            saved = {"format": FORMAT, "kind": kind, "stage": "online", "action_dim": ACTION_DIM,
                     "model_config": learner.cfg, "stats": learner.stats, "model": clone_state(learner.model),
                     "ema": None, "residual": clone_state(learner.actor) if kind == "act" else None,
                     "diffusion_base": clone_state(learner.diffusion_base) if kind == "diffusion" else None,
                     "finetune_denoising_steps": learner.stochastic_steps,
                     "residual_scale": learner.scale, "critic": clone_state(learner.critic),
                     "actor_optimizer": learner.actor_opt.state_dict(), "critic_optimizer": learner.critic_opt.state_dict(),
                     "update": update, "config": config, "source_checkpoint": str(source), "rng": rng_state()}
            save_checkpoint(directory / "policy_last.ckpt", saved)
            frequency = int(config.get("save_every", 10))
            if frequency > 0 and (update + 1) % frequency == 0:
                save_checkpoint(directory / f"policy_update_{update + 1}.ckpt", saved)
    finally:
        env.close()
    print(f"[{kind}] online checkpoint: {directory / 'policy_last.ckpt'}", flush=True)
    return directory / "policy_last.ckpt"
