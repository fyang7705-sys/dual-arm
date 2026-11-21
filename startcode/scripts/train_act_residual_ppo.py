"""Standalone ACT residual PPO. Add only this file to startcode/scripts/.

运行说明：以下命令均在项目的 startcode 目录执行（先 cd startcode）。

终端一：用项目原有脚本启动仿真环境，训练期间保持运行：
  bash scripts/run_online_env.sh --config configs/act.yaml

终端二：从单路 cam_head 的离线 ACT 权重开始在线训练（将路径替换为真实 checkpoint）：
  python scripts/train_act_residual_ppo.py --set offline_checkpoint=/absolute/path/offline/policy_last.ckpt

恢复训练：保持仿真服务运行，使用上一次在线训练生成的 checkpoint：
  python scripts/train_act_residual_ppo.py --set resume=/absolute/path/online/policy_last.ckpt --set finetune_updates=200

finetune_updates 是训练的总更新次数，不是额外次数，必须大于已完成次数。
恢复时请保持原采样配置和 seed；本文件不保存仿真场景的运行状态。
checkpoint 和训练日志默认写入 startcode/outputs/act/online/ 下的运行目录。

Preserves the project's checkpoint format and residual actor parameter names.
Requires existing startcode/src, torch, torchvision, numpy, h5py and OpenCV.
Does not import the optional Diffusion/XPolicyLab model code.
This file connects to an existing simulator; it does not start Docker or Isaac.
CPU fake-environment tests validate software only, not robotics performance.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Allow direct execution from any working directory after placement in scripts/.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from starter.cli import arguments
from starter.data.demos import resize_rgb
from starter.envs.remote import RemoteEnv
from starter.interfaces import ACTION_DIM, pack_ee_action
from starter.paths import REPO_ROOT
from starter.policies.act_model import ACT

FORMAT = "dual-arm.policy.v1"


def build_model(kind, config):
    if kind != "act":
        raise ValueError("This standalone trainer supports ACT only")
    return ACT(config)

class Vision(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(4, 32), nn.SiLU(),
                                 nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
                                 nn.Conv2d(64, width, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.position = nn.Parameter(torch.randn(1, 16, width) * 0.02)

    def forward(self, image):
        return self.net(image).flatten(2).transpose(1, 2) + self.position

class Condition(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.vision = Vision(width)
        self.proprio = nn.Linear(ACTION_DIM, width)

    def forward(self, image, state):
        return torch.cat((self.vision(image), self.proprio(state).unsqueeze(1)), dim=1)

class ResidualActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        w, h = int(config["width"]), int(config["chunk"])
        self.condition = Condition(w)
        self.mean = nn.Sequential(nn.Linear(w + h * ACTION_DIM, w), nn.Tanh(), nn.Linear(w, h * ACTION_DIM))
        nn.init.zeros_(self.mean[-1].weight)
        nn.init.zeros_(self.mean[-1].bias)
        self.log_std = nn.Parameter(torch.full((h, ACTION_DIM), math.log(float(config.get("residual_std", 0.3)))))
        self.chunk = h

    def distribution(self, image, state, base):
        features = torch.cat((self.condition(image, state).mean(1), base.flatten(1)), -1)
        mean = self.mean(features).reshape(-1, self.chunk, ACTION_DIM)
        return Normal(mean, self.log_std.clamp(-5, 0).exp().expand_as(mean))

class Critic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.condition = Condition(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, image, state):
        if image.ndim == 5:
            image, state = image[:, -1], state[:, -1]
        return self.head(self.condition(image, state).mean(1)).squeeze(-1)

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

def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

def assert_same_state(model, snapshot):
    if any(not torch.equal(value.detach().cpu(), snapshot[key]) for key, value in model.state_dict().items()):
        raise AssertionError("Frozen ACT base changed during residual training")

def validate_config(config):
    """Reject invalid settings before constructing models or opening the simulator."""
    for key, default in (('num_envs', 1), ('finetune_episodes', 4), ('finetune_updates', 100), ('max_decisions_per_episode', 550), ('ppo_epochs', 4), ('ppo_batch_size', 32)):
        value = config.get(key, default)
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config.get('finetune_episodes', 4) % config.get('num_envs', 1):
        raise ValueError('finetune_episodes must be a multiple of num_envs')
    for key, default, low, high in (('gamma', 0.99, 0, 1), ('gae_lambda', 0.95, 0, 1), ('ppo_clip', 0.2, 0, 1)):
        value = float(config.get(key, default))
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be finite and in [{low}, {high}]')
    for key, default in (('actor_lr', 1e-05), ('critic_lr', 0.0001), ('max_grad_norm', 1), ('target_kl', 0.03), ('residual_std', 0.3), ('residual_scale', 0.1)):
        value = float(config.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key, default in (('entropy_coef', 0), ('value_clip', 0.2)):
        value = float(config.get(key, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')

def executed_sum(values, durations):
    """Marginalize independent Gaussian coordinates never sent to the environment.

    Includes early termination within a chunk. No tanh Jacobian is required:
    PPO ratios are evaluated on the stored latent under the same fixed map.
    """
    mask = torch.arange(values.shape[1], device=values.device)[None] < durations[:, None]
    return (values.sum(-1) * mask).sum(-1)

def gae(records, gamma, lam):
    """Each record spans n physics-control actions; boundaries never leak GAE.

    Success/invalid terminal => bootstrap zero. Timeout/collector cutoff =>
    bootstrap V(last_obs), but stop the trace when resetting the environment.
    """
    carry = 0.0
    for item in reversed(records):
        discount = gamma ** item['duration']
        delta = item['reward'] + discount * item['next_value'] - item['value']
        carry = delta + (0 if item['boundary'] else (gamma * lam) ** item['duration'] * carry)
        item['advantage'] = carry
        item['return'] = carry + item['value']

def clipped_objective(logprob, old_logprob, advantage, clip):
    logratio = (logprob - old_logprob).clamp(-20, 20)
    ratio = logratio.exp()
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    approximate_kl = (ratio - 1 - logratio).mean()
    return (-objective.mean(), approximate_kl)

class Learner:

    def __init__(self, kind, config, payload):
        self.kind, self.config, self.payload = (kind, config, payload)
        self.device, self.cfg, self.stats = (pick_device(config), dict(payload['model_config']), payload['stats'])
        if self.cfg.get("camera_names") != ["cam_head"]:
            raise ValueError("ACT residual PPO requires a checkpoint with camera_names=[cam_head]")
        self.model = build_model(kind, self.cfg).to(self.device)
        self.model.load_state_dict(payload.get('ema') or payload['model'])
        self.model.eval()
        self.history = None
        self.base_snapshot = None
        self.diffusion_base = None
        self.model.requires_grad_(False)
        self.base_snapshot = clone_state(self.model)
        self.cfg['residual_std'] = float(config.get('residual_std', 0.3))
        self.actor = ResidualActor(self.cfg).to(self.device)
        if payload.get('residual') is not None:
            self.actor.load_state_dict(payload['residual'])
        self.critic = Critic(int(self.cfg['width'])).to(self.device)
        self.scale = float(payload.get('residual_scale', config.get('residual_scale', 0.1)))
        self.stochastic_steps = int(payload.get('finetune_denoising_steps', config.get('finetune_denoising_steps', 4)))
        if kind == 'diffusion' and (not 1 <= self.stochastic_steps <= self.model.inference_steps):
            raise ValueError('finetune_denoising_steps must be in [1, inference_steps]')
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(config.get('actor_lr', 1e-05)))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(config.get('critic_lr', 0.0001)))
        if payload['stage'] == 'online':
            self.critic.load_state_dict(payload['critic'])
            self.actor_opt.load_state_dict(payload['actor_optimizer'])
            self.critic_opt.load_state_dict(payload['critic_optimizer'])

    @torch.no_grad()
    def sample(self, observations):
        images, states = self.history.arrays(observations) if self.history is not None else observation_arrays(observations, self.stats, self.cfg['image_size'])
        image, state = tensor_observations(images, states, self.device)
        values = self.critic(image, state).cpu().numpy()
        base = self.model(image[:, None], state)[0]
        dist = self.actor.distribution(image, state, base)
        latent = dist.sample()
        action = base + self.scale * latent.tanh()
        traces = [{'base': base[i].cpu(), 'latent': latent[i].cpu(), 'old_logprob': dist.log_prob(latent).flatten(1).sum(-1)[i].cpu(), 'mean': dist.mean[i].cpu(), 'std': dist.stddev[i].cpu()} for i in range(len(state))]
        action = action.cpu().numpy() * np.asarray(self.stats['action_std']) + np.asarray(self.stats['action_mean'])
        fallback = np.asarray([o['proprio'] for o in observations])[:, None, :]
        return (project_actions(action, fallback), images, states, values, traces)

    @torch.no_grad()
    def values(self, observations):
        arrays = observation_arrays(observations, self.stats, self.cfg['image_size'])
        return self.critic(*tensor_observations(*arrays, self.device)).cpu().numpy()

    def update(self, records):
        cfg, device = (self.config, self.device)
        if not records:
            raise ValueError('Cannot update PPO with an empty rollout')
        advantages = torch.tensor([r['advantage'] for r in records], device=device, dtype=torch.float32)
        if not torch.isfinite(advantages).all():
            raise FloatingPointError('Non-finite rollout advantages')
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-06)
        returns = torch.tensor([r['return'] for r in records], device=device, dtype=torch.float32)
        old_values = torch.tensor([r['value'] for r in records], device=device, dtype=torch.float32)
        if not torch.isfinite(returns).all() or not torch.isfinite(old_values).all():
            raise FloatingPointError('Non-finite rollout value targets')
        if any((not 1 <= r['duration'] <= int(self.cfg['execute_steps']) for r in records)):
            raise ValueError('Rollout duration must be between 1 and execute_steps')
        metrics = {'policy_loss': [], 'value_loss': [], 'approx_kl': [], 'entropy': []}
        early_stop = False
        actor_steps = 0
        batch_size = int(cfg.get('ppo_batch_size', 32))
        for _ in range(int(cfg.get('ppo_epochs', 4))):
            stop = False
            for indices in torch.randperm(len(records)).split(batch_size):
                rows = [records[i] for i in indices.tolist()]
                ids = indices.to(device)
                image, state = tensor_observations(np.stack([r['image'] for r in rows]), np.stack([r['state'] for r in rows]), device)
                trace = {key: torch.stack([r['trace'][key] for r in rows]).to(device) for key in rows[0]['trace']}
                dist = self.actor.distribution(image, state, trace['base'])
                durations = torch.tensor([r['duration'] for r in rows], device=device)
                behavior = torch.distributions.Normal(trace['mean'], trace['std'])
                old_logprob = executed_sum(behavior.log_prob(trace['latent']), durations)
                logprob = executed_sum(dist.log_prob(trace['latent']), durations)
                entropy = executed_sum(dist.entropy(), durations).mean()
                policy_loss, _ = clipped_objective(logprob, old_logprob, advantages[ids], float(cfg.get('ppo_clip', 0.2)))
                kl = executed_sum(torch.distributions.kl_divergence(behavior, dist), durations).mean()
                actor_loss = policy_loss - float(cfg.get('entropy_coef', 0.0)) * entropy
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError('Non-finite PPO actor loss')
                if not torch.isfinite(kl):
                    raise FloatingPointError('Non-finite PPO KL')
                if float(kl.detach()) <= float(cfg.get('target_kl', 0.03)):
                    self.actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(cfg.get('max_grad_norm', 1)), error_if_nonfinite=True)
                    self.actor_opt.step()
                    actor_steps += 1
                else:
                    early_stop = True
                value = self.critic(image, state)
                clipped_value = old_values[ids] + (value - old_values[ids]).clamp(-float(cfg.get('value_clip', 0.2)), float(cfg.get('value_clip', 0.2)))
                value_loss = 0.5 * torch.maximum((value - returns[ids]).square(), (clipped_value - returns[ids]).square()).mean()
                if float(cfg.get('value_clip', 0.2)) == 0:
                    value_loss = 0.5 * (value - returns[ids]).square().mean()
                if not torch.isfinite(value_loss):
                    raise FloatingPointError('Non-finite PPO critic loss')
                self.critic_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(cfg.get('max_grad_norm', 1)), error_if_nonfinite=True)
                self.critic_opt.step()
                for key, val in zip(metrics, (policy_loss, value_loss, kl, entropy)):
                    metrics[key].append(float(val.detach()))
                if float(kl.detach()) > float(cfg.get('target_kl', 0.03)):
                    stop = True
                    break
            if stop:
                break
        if self.base_snapshot is not None:
            assert_same_state(self.model, self.base_snapshot)
        result = {key: float(np.mean(values)) for key, values in metrics.items()}
        result.update(actor_steps=actor_steps, early_stop=early_stop, return_std=float(returns.std(unbiased=False)), advantage_std=float(np.std([r['advantage'] for r in records])))
        result['exact_kl'] = result['approx_kl']
        return result

def collect(learner, env, config, update_index):
    width, episodes = (int(config.get('num_envs', 1)), int(config.get('finetune_episodes', 4)))
    if width < 1 or episodes < 1 or episodes % width:
        raise ValueError('finetune_episodes must be a positive multiple of num_envs')
    gamma, lam = (float(config.get('gamma', 0.99)), float(config.get('gae_lambda', 0.95)))
    records, episode_metrics = ([], [])
    execute = int(learner.cfg['execute_steps'])
    cap = int(config.get('max_decisions_per_episode', 550))
    if cap < 1:
        raise ValueError('max_decisions_per_episode must be positive')
    for wave in range(episodes // width):
        seeds = [int(config.get('seed', 0)) + update_index * episodes + wave * width + i for i in range(width)]
        observations, _ = env.reset_batch(seeds)
        if len(observations) != width:
            raise RuntimeError('Environment reset batch size mismatch')
        for i, observation in enumerate(observations):
            observation['env_idx'] = i
        if learner.history is not None:
            learner.history.reset()
            learner.history.update(observations)
        active = list(range(width))
        per_env = [[] for _ in range(width)]
        raw_returns, successes = ([0.0] * width, [False] * width)
        last_scores, frames = ([0.0] * width, [0] * width)
        for decision in range(cap):
            selected = [observations[i] for i in active]
            actions, images, states, values, traces = learner.sample(selected)
            pending = {i: {'image': images[j], 'state': states[j], 'value': float(values[j]), 'trace': traces[j], 'reward': 0.0, 'duration': 0, 'boundary': False, 'terminal': False} for j, i in enumerate(active)}
            action_lookup = {i: actions[j] for j, i in enumerate(active)}
            running = list(active)
            for step in range(execute):
                results = env.step_batch([action_lookup[i][step].tolist() for i in running], running)
                if len(results) != len(running):
                    raise RuntimeError('Environment step batch size mismatch')
                keep = []
                for i, result in zip(running, results):
                    obs, reward, terminated, truncated, info = result
                    if not math.isfinite(float(reward)):
                        raise FloatingPointError('Environment returned non-finite reward')
                    obs['env_idx'] = i
                    observations[i] = obs
                    if learner.history is not None:
                        learner.history.update([obs])
                    item = pending[i]
                    item['reward'] += gamma ** item['duration'] * float(reward)
                    item['duration'] += 1
                    item['boundary'] = bool(terminated or truncated)
                    item['terminal'] = bool(terminated)
                    raw_returns[i] += float(reward)
                    frames[i] += 1
                    successes[i] = successes[i] or bool(info.get('success', False))
                    last_scores[i] = float(info.get('external_score', 0))
                    if not item['boundary']:
                        keep.append(i)
                running = keep
                if not running:
                    break
            bootstrap_ids = [i for i in active if not pending[i]['terminal']]
            next_values = dict(zip(bootstrap_ids, learner.values([observations[i] for i in bootstrap_ids]))) if bootstrap_ids else {}
            for j, i in enumerate(active):
                item = pending[i]
                item['next_value'] = 0.0 if item['terminal'] else float(next_values[i])
                item['boundary'] = item['boundary'] or decision + 1 == cap
                per_env[i].append(item)
            active = running
            if not active:
                break
        for i in range(width):
            gae(per_env[i], gamma, lam)
            records.extend(per_env[i])
            episode_metrics.append({'seed': seeds[i], 'return': raw_returns[i], 'success': successes[i], 'score': last_scores[i], 'steps': frames[i], 'collector_cutoff': not per_env[i][-1]['terminal'] and i in active})
    return (records, episode_metrics)

def train(kind, config):
    if kind != "act":
        raise ValueError("This standalone trainer supports ACT only")
    validate_config(config)
    if config.get('env_backend', 'remote') not in ('remote', 'fake'):
        raise ValueError('env_backend must be remote or fake')
    seed_everything(int(config.get('seed', 0)))
    source = config.get('resume') or config.get('offline_checkpoint')
    if not source:
        raise ValueError('Online training requires offline_checkpoint (or resume); random initialization is forbidden')
    payload = load_payload(source)
    if payload['kind'] != kind:
        raise ValueError('Checkpoint policy kind mismatch')
    if config.get('resume') and payload['stage'] != 'online':
        raise ValueError('resume must refer to an online checkpoint')
    if not config.get('resume') and payload['stage'] != 'offline':
        raise ValueError('offline_checkpoint must refer to an offline checkpoint')
    if int(config.get('ppo_batch_size', 32)) < 1 or int(config.get('ppo_epochs', 4)) < 1:
        raise ValueError('PPO batch size and epochs must be positive')
    learner = Learner(kind, config, payload)
    if payload['stage'] == 'online':
        restore_rng(payload.get('rng'))
    start = int(payload['update']) + 1 if payload['stage'] == 'online' else 0
    updates = int(config.get('finetune_updates', 100))
    if updates <= start:
        raise ValueError('finetune_updates must exceed the completed resume update')
    directory = run_directory(config, kind, 'online')
    if config.get('env_backend', 'remote') == 'fake':
        from starter.envs.fake import ToyBatchEnv
        env = ToyBatchEnv(int(config.get('num_envs', 1)), int(config.get('fake_episode_steps', 4)))
    else:
        env = RemoteEnv(str(config.get('env_host', '127.0.0.1')), int(config.get('env_port', 5555)), timeout=float(config.get('env_timeout', 2400)), reset_timeout=float(config.get('reset_timeout', 10800)))
    try:
        for update in range(start, updates):
            records, episodes = collect(learner, env, config, update)
            metrics = learner.update(records)
            metrics.update({'update': update + 1, 'decisions': len(records), 'return_mean': float(np.mean([e['return'] for e in episodes])), 'success_rate': float(np.mean([e['success'] for e in episodes])), 'benchmark_score': float(np.mean([e['score'] for e in episodes])), 'env_backend': config.get('env_backend', 'remote')})
            append_metrics(directory, metrics)
            with (directory / 'episodes.jsonl').open('a', encoding='utf-8') as f:
                import json
                for episode in episodes:
                    f.write(json.dumps({'update': update + 1, **episode}) + '\n')
            saved = {'format': FORMAT, 'kind': kind, 'stage': 'online', 'action_dim': ACTION_DIM, 'model_config': learner.cfg, 'stats': learner.stats, 'model': clone_state(learner.model), 'ema': None, 'residual': clone_state(learner.actor), 'diffusion_base': None, 'finetune_denoising_steps': learner.stochastic_steps, 'residual_scale': learner.scale, 'critic': clone_state(learner.critic), 'actor_optimizer': learner.actor_opt.state_dict(), 'critic_optimizer': learner.critic_opt.state_dict(), 'update': update, 'config': config, 'source_checkpoint': str(source), 'rng': rng_state()}
            save_checkpoint(directory / 'policy_last.ckpt', saved)
            frequency = int(config.get('save_every', 10))
            if frequency > 0 and (update + 1) % frequency == 0:
                save_checkpoint(directory / f'policy_update_{update + 1}.ckpt', saved)
    finally:
        env.close()
    print(f"[{kind}] online checkpoint: {directory / 'policy_last.ckpt'}", flush=True)
    return directory / 'policy_last.ckpt'

def main():
    argv = sys.argv[1:]
    if "--config" not in argv and not any(a.startswith("--config=") for a in argv):
        argv = ["--config", str(PROJECT / "configs/act.yaml"), *argv]
    _, config = arguments("act", ["online", *argv])
    if "policy_gpu" in config:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["policy_gpu"])
    train("act", config)


if __name__ == "__main__":
    main()
