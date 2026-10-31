"""CPU smoke tests for the Diffusion Policy + DPPO implementation.

They use synthetic tensors, so they run without Isaac, demonstrations or a GPU.
Run with the startcode source on PYTHONPATH::

    PYTHONPATH=/path/to/startcode/src python tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from starter.policies import dp_core as core  # noqa: E402
from starter.policies.dp_agent import DPPOAgent, Transition  # noqa: E402


def tiny_config() -> dict:
    return {
        "action_dim": 16,
        "state_dim": 16,
        "chunk_size": 4,
        "image_size": 32,
        "diffusion_steps": 20,
        "hidden_dim": 64,
        "context_dim": 64,
        "vision_dim": 32,
        "time_dim": 32,
        "beta_start": 1e-4,
        "beta_end": 0.02,
    }


def tiny_agent(model: torch.nn.Module) -> DPPOAgent:
    normalization = core.Normalization(
        np.zeros(16, np.float32), np.ones(16, np.float32), np.zeros(16, np.float32), np.ones(16, np.float32)
    )
    config = tiny_config()
    config.update(
        {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "gamma_diff": 0.9,
            "stochastic_steps": 3,
            "sampler_steps": 5,
            "eta": 0.8,
            "ppo_epochs": 1,
            "minibatch_size": 2,
            "learning_rate": 1e-4,
            "reward_scale": 0.01,
        }
    )
    return DPPOAgent(
        model,
        core.EMA(model, 0.99),
        core.Critic(16, 32),
        normalization,
        config,
        torch.device("cpu"),
    )


class FakeEnv:
    def __init__(self, width: int = 2, length: int = 3) -> None:
        self.width = width
        self.length = length
        self.steps = [0] * width

    def _obs(self):
        return {
            "image": np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
            "proprio": np.zeros(16, dtype=np.float32),
        }

    def reset_batch(self, seeds):
        self.steps = [0] * self.width
        return [self._obs() for _ in range(self.width)], [{} for _ in range(self.width)]

    def step_batch(self, actions, active):
        results = []
        for action, index in zip(actions, active):
            assert len(action) == 16
            self.steps[index] += 1
            done = self.steps[index] >= self.length
            results.append((self._obs(), 0.0, done, False, {"success": False}))
        return results

    def privileged(self, index):
        return {"blocks": {"block_0": {"pos": [0.0, 0.0, 0.0]}}}


class DiffusionModelTest(unittest.TestCase):
    def test_training_loss_and_sampling(self):
        torch.manual_seed(0)
        model = core.DiffusionPolicy(tiny_config())
        images = torch.rand(2, 3, 32, 32)
        proprio = torch.randn(2, 16)
        actions = torch.randn(2, 4, 16)
        loss = model.training_loss(images, proprio, actions)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        sampled = model.sample(images, proprio, sample_steps=5)
        self.assertEqual(tuple(sampled.shape), (2, 64))

    def test_chain_logp_matches_sampling(self):
        torch.manual_seed(1)
        model = core.DiffusionPolicy(tiny_config())
        images = torch.rand(2, 3, 32, 32)
        proprio = torch.randn(2, 16)
        _actions, chain = model.sample(
            images, proprio, sample_steps=5, eta=0.8, stochastic_steps=3, return_chain=True
        )
        self.assertEqual(len(chain), 3)
        context = model.encode_obs(images, proprio)
        batch = images.shape[0]
        x_t = torch.stack([item["x_t"] for item in chain], dim=1)
        x_prev = torch.stack([item["x_prev"] for item in chain], dim=1)
        timesteps = torch.tensor([[item["t"] for item in chain]] * batch)
        prev_timesteps = torch.tensor([[item["t_prev"] for item in chain]] * batch)
        sigma = torch.tensor([item["sigma"] for item in chain])
        repeated = context.unsqueeze(1).expand(batch, len(chain), -1).reshape(-1, 64)
        recomputed = model.chain_logp(
            x_t.reshape(-1, 64),
            x_prev.reshape(-1, 64),
            repeated,
            timesteps.reshape(-1),
            prev_timesteps.reshape(-1),
            sigma.repeat(batch),
        ).reshape(batch, len(chain))
        old = torch.tensor([[float(item["logp"][i]) for item in chain] for i in range(batch)])
        self.assertTrue(torch.allclose(recomputed, old, atol=1e-4))


class DPPOAgentTest(unittest.TestCase):
    def test_update_and_rollout(self):
        torch.manual_seed(2)
        model = core.DiffusionPolicy(tiny_config())
        agent = tiny_agent(model)
        env = FakeEnv(width=2, length=3)
        trajectories, episodes = agent.collect_wave(env, [0, 1])
        self.assertEqual(len(trajectories), 2)
        self.assertEqual([episode.steps for episode in episodes], [3, 3])
        metrics = agent.update(trajectories)
        self.assertIn("policy_loss", metrics)
        self.assertTrue(np.isfinite(metrics["policy_loss"]))

    def test_checkpoint_roundtrip_and_runtime(self):
        torch.manual_seed(3)
        model = core.DiffusionPolicy(tiny_config())
        normalization = core.Normalization(
            np.zeros(16, np.float32), np.ones(16, np.float32), np.zeros(16, np.float32), np.ones(16, np.float32)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ckpt.ckpt"
            core.save_checkpoint(path, model, core.EMA(model), tiny_config(), normalization, "diffusion_policy")
            runtime = core.DiffusionRuntime(core.load_checkpoint_payload(path), torch.device("cpu"))
            observation = {
                "image": np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8),
                "proprio": np.zeros(16, dtype=np.float32),
            }
            chunk = runtime.act_chunk_from_obs(observation)
            self.assertEqual(chunk.shape, (4, 16))
            self.assertTrue(np.all(chunk[:, [7, 15]] >= 0.0))
            self.assertTrue(np.all(chunk[:, [7, 15]] <= 1.0))

    def test_raw_robodojo_observation_conversion(self):
        state = {
            "left_ee_pose": np.zeros(7, dtype=np.float32),
            "left_ee_joint_state": np.zeros(1, dtype=np.float32),
            "right_ee_pose": np.zeros(7, dtype=np.float32),
            "right_ee_joint_state": np.zeros(1, dtype=np.float32),
        }
        observation = {
            "vision": {"cam_head": {"color": np.zeros((32, 32, 3), dtype=np.uint8)}},
            "state": state,
        }
        image, proprio = core.observation_to_arrays(observation)
        self.assertEqual(proprio.shape, (16,))
        self.assertEqual(image.shape, (32, 32, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
