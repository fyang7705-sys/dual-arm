"""CPU smoke tests for ACT + residual PPO.

Synthetic tensors only: no Isaac, no demonstrations, no GPU needed.
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

from starter.policies import act_core as core  # noqa: E402
from starter.policies.act_agent import ActResidualAgent, ResidualCritic  # noqa: E402


def tiny_config() -> dict:
    return {
        "action_dim": 16,
        "state_dim": 16,
        "chunk_size": 4,
        "image_size": 32,
        "latent_dim": 8,
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "decoder_layers": 1,
        "dropout": 0.0,
        "kl_weight": 10.0,
        "residual_hidden_dim": 32,
        "residual_initial_std": 0.3,
        "std_min": 0.05,
        "std_max": 0.5,
        "eval_chunk_size": 1,
    }


def normalization() -> core.Normalization:
    return core.Normalization(
        np.zeros(16, np.float32), np.ones(16, np.float32), np.zeros(16, np.float32), np.ones(16, np.float32)
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


class ActModelTest(unittest.TestCase):
    def test_training_loss_and_chunk(self):
        torch.manual_seed(0)
        model = core.ActPolicy(tiny_config())
        images = torch.rand(2, 3, 32, 32)
        proprio = torch.randn(2, 16)
        actions = torch.randn(2, 4, 16)
        loss, stats = model.training_loss(images, proprio, actions)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        model.eval()
        chunk = model.act_chunk(images, proprio)
        self.assertEqual(tuple(chunk.shape), (2, 4, 16))

    def test_base_stays_frozen_during_residual_update(self):
        torch.manual_seed(1)
        base = core.ActPolicy(tiny_config())
        base.freeze()
        fingerprint = base.weight_fingerprint()
        norm = normalization()
        residual = core.ResidualHead(64, 32, 16, core.default_residual_scales(), 0.3)
        critic = ResidualCritic(64, 32)
        agent = ActResidualAgent(
            base,
            residual,
            critic,
            norm,
            dict(
                tiny_config(),
                gamma=0.99,
                gae_lambda=0.95,
                ppo_epochs=1,
                minibatch_size=2,
                learning_rate=1e-4,
                reward_scale=0.01,
            ),
            torch.device("cpu"),
        )
        env = FakeEnv(width=2, length=3)
        trajectories, episodes = agent.collect_wave(env, [0, 1])
        self.assertEqual([episode.steps for episode in episodes], [3, 3])
        metrics = agent.update(trajectories)
        self.assertTrue(np.isfinite(metrics["policy_loss"]))
        self.assertEqual(base.weight_fingerprint(), fingerprint)

    def test_residual_runtime_and_checkpoint(self):
        torch.manual_seed(2)
        base = core.ActPolicy(tiny_config())
        base.freeze()
        norm = normalization()
        residual = core.ResidualHead(64, 32, 16, core.default_residual_scales(), 0.3)
        config = dict(tiny_config(), residual_hidden_dim=32, eval_chunk_size=1)
        payload = {
            "kind": "act_residual_ppo",
            "base_config": tiny_config(),
            "base_state": base.state_dict(),
            "base_fingerprint": base.weight_fingerprint(),
            "residual": residual.state_dict(),
            "normalization": norm.to_dict(),
            "config": config,
            "feature_dim": 64,
            "action_dim": 16,
            "residual_scales": core.default_residual_scales(),
            "image_size": 32,
            "chunk_size": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "residual.ckpt"
            core.save_checkpoint(path, payload)
            runtime = core.ActRuntime(core.load_payload(path), torch.device("cpu"))
            self.assertTrue(runtime.is_residual)
            observation = {
                "image": np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8),
                "proprio": np.zeros(16, dtype=np.float32),
            }
            chunk = runtime.act_chunk_from_obs(observation)
            self.assertEqual(chunk.shape, (1, 16))
            self.assertTrue(np.all(chunk[:, [7, 15]] >= 0.0))
            self.assertTrue(np.all(chunk[:, [7, 15]] <= 1.0))

    def test_offline_payload_roundtrip(self):
        torch.manual_seed(3)
        model = core.ActPolicy(tiny_config())
        payload = core.build_act_payload(model, tiny_config(), normalization())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.ckpt"
            core.save_checkpoint(path, payload)
            runtime = core.ActRuntime(core.load_payload(path), torch.device("cpu"))
            self.assertFalse(runtime.is_residual)
            self.assertEqual(runtime.eval_chunk_size, 1)

    def test_raw_observation_conversion(self):
        state = {
            "left_ee_pose": np.zeros(7, np.float32),
            "left_ee_joint_state": np.zeros(1, np.float32),
            "right_ee_pose": np.zeros(7, np.float32),
            "right_ee_joint_state": np.zeros(1, np.float32),
        }
        observation = {
            "vision": {"cam_head": {"color": np.zeros((32, 32, 3), np.uint8)}},
            "state": state,
        }
        image, proprio = core.observation_to_arrays(observation)
        self.assertEqual(proprio.shape, (16,))
        self.assertEqual(image.shape, (32, 32, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
