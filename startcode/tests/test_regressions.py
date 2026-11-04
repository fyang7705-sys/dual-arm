"""Regression coverage for the training/serving failures found in the git audit."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from starter.data.demos import episode_files, load_demos, read_episode
from starter.envs.isaac_batch import _controls_remain, _drain_controls
from starter.envs.remote import RemoteEnv
from starter.policies import act, act_core, diffusion, dp_core
from starter.policies.checkpoints import resolve_checkpoint, write_eval_manifest
from starter.policies.preprocessing import canonicalize_quaternions, canonicalize_state, prepare_image, prepare_image_batch
from test_act_smoke import tiny_config as act_config, normalization, FakeEnv as ActEnv
from test_dp_smoke import tiny_config as dp_config, tiny_agent

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


class PreprocessingTest(unittest.TestCase):
    def test_quaternion_sign_is_identical_offline_online_and_at_180_degrees(self):
        state = np.zeros((2, 16), np.float32)
        state[:, 3:7] = [0, -0.6, 0.8, 0]
        state[:, 11:15] = [-0.5, 0.5, 0.5, 0.5]
        opposite = state.copy()
        opposite[:, 3:7] *= -1
        opposite[:, 11:15] *= -1
        expected = canonicalize_quaternions(state)
        np.testing.assert_allclose(expected, canonicalize_quaternions(opposite))
        for core in (act_core, dp_core):
            norm = core.Normalization(np.zeros(16), np.full(16, 0.001), np.zeros(16), np.ones(16))
            torch.testing.assert_close(norm.normalize_state(torch.from_numpy(state)),
                                       norm.normalize_state(torch.from_numpy(opposite)))
        np.testing.assert_allclose(expected, canonicalize_state(torch.from_numpy(state)).numpy())

    def test_zero_quaternions_are_valid_at_the_controller_boundary(self):
        values = np.zeros((1, 16), np.float32)
        result = act_core.normalize_quaternions_np(values)
        np.testing.assert_allclose(result[0, 3:7], [1, 0, 0, 0])
        np.testing.assert_allclose(dp_core.normalize_quaternions(torch.from_numpy(values)).numpy(), result)

    def test_dark_uint8_images_and_rollout_resize_match_serving(self):
        image = np.ones((48, 80, 3), np.uint8)
        single = prepare_image(image, 32)
        batch = prepare_image_batch(torch.from_numpy(image).permute(2, 0, 1)[None], 32)[0]
        self.assertEqual(tuple(single.shape), (3, 32, 32))
        torch.testing.assert_close(single, batch)
        self.assertAlmostEqual(float(single.mean()), 2 / 255 - 1, places=6)

    def test_online_dp_encoder_receives_training_image_resolution(self):
        agent = tiny_agent(dp_core.DiffusionPolicy(dp_config()))
        seen = []
        hook = agent.model.obs_encoder.register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape)))
        agent.act_batch(torch.ones(1, 3, 48, 80, dtype=torch.uint8), torch.zeros(1, 16))
        hook.remove()
        self.assertEqual(seen, [(1, 3, 32, 32)])


class DiffusionScheduleTest(unittest.TestCase):
    def test_cosine_terminal_signal_is_negligible_and_steps_are_valid(self):
        model = dp_core.DiffusionPolicy(dp_config())
        self.assertLess(float(model.alpha_bars[-1]), 1e-4)
        self.assertEqual(dp_core.make_timesteps(20, 1), [19])
        self.assertEqual(len(dp_core.make_timesteps(20, 100)), 20)
        with self.assertRaises(ValueError):
            dp_core.DiffusionPolicy(dict(dp_config(), beta_schedule="invalid"))

    def test_ddpm_reverse_chain_recovers_an_oracle_action(self):
        model = dp_core.DiffusionPolicy(dp_config())
        target = torch.full((1, 64), 0.25)
        def epsilon(x, context, t):
            alpha = model.alpha_bars[t]
            return (x - alpha.sqrt() * target) / (1 - alpha).sqrt()
        with patch.object(model, "predict_eps", side_effect=epsilon) as predictor:
            result = model.sample_ddpm(torch.zeros(1, 3, 32, 32), torch.zeros(1, 16))
        self.assertEqual(predictor.call_count, 20)
        torch.testing.assert_close(result, target, atol=1e-5, rtol=1e-5)

    def test_checkpoint_preserves_legacy_noise_schedule(self):
        config = dict(dp_config(), beta_schedule="linear")
        model = dp_core.DiffusionPolicy(config)
        payload = {"config": dp_config(), "model": model.state_dict(), "normalization": normalization().to_dict()}
        restored, _, _ = dp_core.build_policy_from_payload(payload, torch.device("cpu"))
        torch.testing.assert_close(restored.alpha_bars, model.alpha_bars)
        self.assertEqual(restored.config["beta_schedule"], "linear")


class TrainingRegressionTest(unittest.TestCase):
    def test_act_padding_does_not_change_posterior_or_loss(self):
        model = act_core.ActPolicy(act_config()).train()
        image, state = torch.rand(2, 3, 32, 32), torch.randn(2, 16)
        actions = torch.randn(2, 4, 16)
        changed = actions.clone()
        changed[:, 1:] += 100
        valid = torch.tensor([[1., 0, 0, 0]] * 2)
        torch.manual_seed(10)
        left, _ = model.training_loss(image, state, actions, valid)
        torch.manual_seed(10)
        right, _ = model.training_loss(image, state, changed, valid)
        torch.testing.assert_close(left, right)

    def test_one_transition_ppo_update_is_finite_and_failure_is_not_success(self):
        agent = tiny_agent(dp_core.DiffusionPolicy(dp_config()))
        trajectories, episodes = agent.collect_wave(ActEnv(1, 1), [0])
        metrics = agent.update(trajectories)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertFalse(episodes[0].success)

    def test_act_residual_replans_the_base_each_step(self):
        from starter.policies.act_agent import ActResidualAgent, ResidualCritic
        base = act_core.ActPolicy(act_config())
        base.freeze()
        agent = ActResidualAgent(base, act_core.ResidualHead(64, 32), ResidualCritic(64, 32),
                                 normalization(), dict(act_config(), ppo_epochs=1), torch.device("cpu"))
        with patch.object(base, "act_chunk", wraps=base.act_chunk) as predict:
            trajectories, episodes = agent.collect_wave(ActEnv(1, 3), [0])
        self.assertEqual(predict.call_count, 3)
        self.assertFalse(episodes[0].success)
        single, _ = agent.collect_wave(ActEnv(1, 1), [0])
        self.assertTrue(np.isfinite(agent.update(single)["policy_loss"]))

    def test_real_hdf5_loading_training_checkpoint_and_deployment_for_both_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data = directory / "data"
            data.mkdir()
            for episode in range(2):
                with h5py.File(data / f"episode_{episode:07d}.hdf5", "w") as handle:
                    for kind in ("state", "action"):
                        group = handle.create_group(kind)
                        for key, width in zip(act_core.ARM_KEYS, act_core.ARM_WIDTHS):
                            value = np.zeros((4, width), np.float32)
                            value[:, 0] = 0.3 if width == 7 else 0.5
                            if width == 7:
                                value[:, 3] = -1
                            group.create_dataset(key + "s", data=value)
                    handle.create_dataset("vision/cam_head/colors", data=np.ones((4, 32, 32, 3), np.uint8))
            common = dict(bench_name="RoboDojo", data_task="stack_blocks", env_cfg="arx_x5", action_type="ee",
                          seed=0, data_path=str(data), device="cpu", epochs=1, batch_size=32, num_workers=0,
                          augment=False, warmup_steps=0, save_every=1, run_stamp="test")
            with patch("starter.paths.OUTPUTS_ROOT", directory / "outputs"):
                act.train_offline({**act_config(), **common})
                diffusion.train_offline({**dp_config(), **common, "amp": False})
            act_file = next((directory / "outputs/act").rglob("policy_offline.ckpt"))
            dp_file = next((directory / "outputs/dp").rglob("offline_last.ckpt"))
            observation = {"image": np.ones((48, 80, 3), np.uint8), "proprio": np.zeros(16, np.float32)}
            for runtime in (act.load_runtime(act_file, torch.device("cpu")), diffusion.load_runtime(dp_file, torch.device("cpu"))):
                result = runtime.act_chunk_from_obs(observation)
                self.assertTrue(np.isfinite(result).all())
                self.assertEqual(result.shape[1], 16)
                np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=-1), 1, atol=1e-5)
            class RolloutEnv(ActEnv):
                def close(self):
                    pass
            online = dict(common, run_stamp="online", finetune_updates=1, finetune_episodes=1,
                          num_envs=1, rollout_max_steps=1, ppo_epochs=1, minibatch_size=1)
            with patch("starter.paths.OUTPUTS_ROOT", directory / "outputs"), \
                 patch.object(act, "RemoteEnv", return_value=RolloutEnv(1, 1)), \
                 patch.object(diffusion, "RemoteEnv", return_value=RolloutEnv(1, 1)):
                act.train_online({**act_config(), **online, "offline_checkpoint": str(act_file)})
                diffusion.train_online({**online, "offline_checkpoint": str(dp_file)})
            residual = next((directory / "outputs/act").rglob("residual_last.ckpt"))
            dppo = next((directory / "outputs/dp").rglob("online_last.ckpt"))
            for runtime in (act.load_runtime(residual, torch.device("cpu")), diffusion.load_runtime(dppo, torch.device("cpu"))):
                self.assertEqual(runtime.act_chunk_from_obs(observation).shape, (1, 16))
            self.assertEqual(diffusion.load_runtime(dppo, torch.device("cpu")).sampler, "ddim")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_dppo_safety_horizon_bootstrap_uses_policy_device(self):
        agent = tiny_agent(dp_core.DiffusionPolicy(dp_config()))
        agent.device = torch.device("cuda:0")
        agent.model.to(agent.device)
        agent.critic.to(agent.device)
        agent.max_steps = 1
        trajectories, _ = agent.collect_wave(ActEnv(1, 3), [0])
        self.assertTrue(trajectories[0][-1].truncated)
        self.assertTrue(np.isfinite(trajectories[0][-1].bootstrap))


class InfrastructureTest(unittest.TestCase):
    def test_parallel_control_finishes_longer_queues(self):
        remaining = {0: 1, 1: 3}
        calls = []
        class Manager:
            def get_empty(self, env_idx_list):
                return [i for i in env_idx_list if not remaining[i]]
            def pop(self, active):
                calls.append(list(active))
                for i in active:
                    remaining[i] -= 1
                return None
        task = SimpleNamespace(robot_manager=SimpleNamespace(control_manager=Manager(), control_robot=lambda **kw: None),
                               sim_step=lambda **kw: None)
        _drain_controls(task, [0, 1])
        self.assertEqual(calls, [[0, 1], [1], [1]])
        self.assertFalse(_controls_remain(task, [0, 1]))

    def test_sparse_remote_privileged_state_keeps_global_env_indices(self):
        env = RemoteEnv("localhost", 5555)
        env._privileged = [{"id": 0}, {"id": 1}]
        reply = dict(observations=[{}], scores=[0.0], terminated=[False], truncated=[False],
                     infos=[{}], privileged=[{"id": 10}])
        with patch.object(env, "_request", return_value=reply):
            env.step_batch([[0.] * 16], [1])
        self.assertEqual(env.privileged(0), {"id": 0})
        self.assertEqual(env.privileged(1), {"id": 10})

    def test_checkpoint_ambiguity_is_an_error_and_manifest_identifies_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "outputs/act/run/first/policy_offline.ckpt"
            second = root / "outputs/act/run/second/policy_offline.ckpt"
            for path in (first, second):
                path.parent.mkdir(parents=True)
                path.write_bytes(b"test weights")
            with self.assertRaisesRegex(ValueError, "Multiple runs"):
                resolve_checkpoint(root, "act", "latest_offline")
            self.assertEqual(resolve_checkpoint(root, "act", str(first)), first)
            runtime = SimpleNamespace(payload={"kind": "act_policy", "epoch": 88}, eval_chunk_size=4)
            manifest = write_eval_manifest(first, runtime, "act", 0)
            metadata = json.loads(manifest.read_text())
            self.assertEqual(metadata["epoch"], 88)
            self.assertEqual(len(metadata["sha256"]), 64)

    def test_data_paths_and_missing_data_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0000000.hdf5"
            path.touch()
            self.assertEqual(episode_files(Path(directory)), [path])
        with self.assertRaises(FileNotFoundError):
            load_demos({"data_path": "/missing/real/data"})
        for split in (act._split, diffusion._split_episode_indices):
            with self.assertRaises(ValueError):
                split(1, 0.1, 0)
        for find in (act.find_offline_checkpoint, diffusion.find_offline_checkpoint):
            with self.assertRaises(FileNotFoundError):
                find({"offline_checkpoint": "/missing/requested/checkpoint.ckpt"})

    def test_eval_adapter_batched_request_after_an_environment_finishes(self):
        for route in ("act", "diffusion"):
            spec = importlib.util.spec_from_file_location(f"serve_{route}", ROOT / "policies" / route / "serve_policy.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            runtime = SimpleNamespace(chunk_size=4, eval_chunk_size=1,
                                      act_chunk_from_obs=lambda obs: np.full((1, 16), obs["id"]))
            cls = module.ActEvalModel if route == "act" else module.DiffusionEvalModel
            model = cls(runtime)
            model.update_obs_batch([{"id": 1}])
            result = model.get_action_batch([1])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0][0]["left_ee_pose"][0], 1)


if __name__ == "__main__":
    unittest.main()
