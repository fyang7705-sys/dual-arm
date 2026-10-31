"""Environment and data contracts. Policy training is not implemented."""

from __future__ import annotations

import socket
import threading
import time
import unittest
from pathlib import Path

from starter.config import load_config, parse_simple_yaml
from starter.data.demos import hold_demos
from starter.envs.fake import FakeTask
from starter.envs.stack_blocks import ParallelAdapter, StackBlocksAdapter
from starter.interfaces import ACTION_DIM, pack_ee_action, unpack_ee_action
from starter.policies.act import train_online as act_online
from starter.policies.diffusion import train_offline as diffusion_offline
from starter.policies.diffusion import train_online as diffusion_online
from starter.rewards.intrinsic import combine_reward

ROOT = Path(__file__).resolve().parents[1]


class _BatchFake(FakeTask):
    def __init__(self) -> None:
        super().__init__(step_lim=4, score=0.0)
        self.num_envs = 2
        self.batch_calls = []
        self.reward_manager.get_reward = lambda final_check=False: [0.0, 0.0]
        self.reward_manager.get_score = lambda reward_lst=None: [0.0, 15.0]

        def poses(label):
            del label
            return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]], [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]

        self.reward_manager.func_parser.get_label_pose = poses

    def reset(self, seed=None, options=None) -> None:
        del options
        self.reset_seeds.append(seed)

    def get_obs_batch(self, env_idx_list=None):
        return [self.get_obs() for _ in env_idx_list]

    def take_action_batch(self, actions, env_idx_list=None) -> None:
        self.batch_calls.append((list(env_idx_list), len(actions)))


class ObservationContractTest(unittest.TestCase):
    def test_reset_marks_plain_task_scenes_active(self):
        from starter.envs.stack_blocks import _seed_gripper_state

        class _Robot:
            arm_name = "left_arm"

        class _RobotManager:
            def __init__(self) -> None:
                self.robot_list = [_Robot()]
                self.init_calls = 0

            def set_origin_endpose(self) -> None:
                return None

            def set_robot_init_state(self) -> None:
                self.init_calls += 1

            def get_real_endpose(self, robot):
                del robot
                return [
                    [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                    [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0],
                ]

        class _Reward:
            def __init__(self) -> None:
                self.calls = 0
                self.func_parser = type("Parser", (), {"robot_origin_endpose": [{}, {}]})()

            def init_state(self) -> None:
                self.calls += 1

        class _Task:
            num_envs = 2

            def __init__(self) -> None:
                self.robot_manager = _RobotManager()
                self.reward_manager = _Reward()

        task = _Task()
        _seed_gripper_state(task)
        self.assertEqual(task.success, [True, True])
        self.assertEqual(task.reward_manager.calls, 1)
        task.success[0] = False
        _seed_gripper_state(task)
        self.assertEqual(task.success, [True, True])

        class _EvalTask(_Task):
            def __init__(self) -> None:
                super().__init__()
                self.end_flag = [False, False]
                self.success = [False, True]

        eval_task = _EvalTask()
        _seed_gripper_state(eval_task)
        self.assertEqual(eval_task.success, [False, True])

    def test_policy_view_drops_wrists_and_block_pose(self):
        adapter = StackBlocksAdapter(FakeTask(score=15.0))
        observation, _info = adapter.reset(seed=3)
        self.assertEqual(observation["camera"], "cam_head")
        self.assertEqual(len(observation["proprio"]), ACTION_DIM)
        self.assertNotIn("block_0", observation)
        self.assertNotIn("cam_left_wrist", observation)
        privileged = adapter.get_privileged_state()
        self.assertIn("block_0", privileged["blocks"])
        self.assertNotIn("blocks", observation)

    def test_step_preserves_binary_reward_and_separate_score(self):
        task = FakeTask(step_lim=4, score=15.0)
        adapter = StackBlocksAdapter(task)
        adapter.reset(seed=0)
        action = [0.1] * ACTION_DIM
        action[7] = 1.5
        _obs, reward, terminated, truncated, info = adapter.step(action)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["external_score"], 15.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(task.reward_manager.score, 15.0)
        self.assertEqual(combine_reward(reward, 3.0), 3.0)
        self.assertEqual(task.actions[0]["left_ee_joint_state"], [1.0])

    def test_parallel_step_runs_every_active_scene(self):
        task = _BatchFake()
        adapter = ParallelAdapter(task)
        observations, infos = adapter.reset_batch([3, 4])
        self.assertEqual(len(observations), 2)
        self.assertEqual([item["seed"] for item in infos], [3, 4])
        action = [0.0] * ACTION_DIM
        results = adapter.step_batch([action, action], [0, 1])
        self.assertEqual(results[1][1], 0.0)
        self.assertEqual(results[1][4]["external_score"], 15.0)
        self.assertEqual(task.batch_calls, [([0, 1], 2)])

    def test_success_score_terminates(self):
        task = FakeTask(step_lim=4, score=100.0)
        task.reward_manager.success_reward = 1.0
        adapter = StackBlocksAdapter(task)
        adapter.reset(seed=1)
        _obs, reward, terminated, _truncated, info = adapter.step([0.0] * ACTION_DIM)
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertTrue(info["success"])

    def test_score_is_not_a_substitute_for_binary_success(self):
        adapter = StackBlocksAdapter(FakeTask(score=100.0))
        adapter.reset(seed=0)
        _, reward, terminated, _, info = adapter.step([0.0] * ACTION_DIM)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(info["success"])


class ActionPackTest(unittest.TestCase):
    def test_round_trip(self):
        state = {
            "left_ee_pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            "left_ee_joint_state": [0.4],
            "right_ee_pose": [0.5, 0.6, 0.7, 1.0, 0.0, 0.0, 0.0],
            "right_ee_joint_state": [0.8],
        }
        self.assertEqual(unpack_ee_action(pack_ee_action(state)), state)


class DataAndConfigTest(unittest.TestCase):
    def test_hold_demos_use_the_ee_width(self):
        observation, action = hold_demos(count=2, seed=0)[0]
        self.assertEqual(len(action), ACTION_DIM)
        self.assertEqual(len(observation), ACTION_DIM + 1)

    def test_shipped_configs_match_course_task_and_camera(self):
        env = load_config(ROOT / "configs/env.yaml")
        evaluation = load_config(ROOT / "configs/eval.yaml")
        self.assertEqual(env["rollout_task"], "stack_blocks_random")
        self.assertEqual(env["data_task"], "stack_blocks")
        self.assertEqual(env["eval_task"], "stack_blocks_random")
        self.assertNotEqual(env["policy_gpu"], env["env_gpu"])
        self.assertEqual(evaluation["task"], "stack_blocks_random")
        self.assertEqual(evaluation["action_type"], "ee")
        self.assertEqual(env["camera_names"], ["cam_head"])
        self.assertGreater(env["learning_rate"], 0)
        self.assertGreater(env["ppo_epochs"], 0)
        image_env = (ROOT / "configs/sim_image.env").read_text(encoding="utf-8")
        self.assertIn(
            "SIM_IMAGE_URL=https://huggingface.co/datasets/NantianHe/stack-blocks-images/resolve/main/stack-blocks-sim_isaacsim5.1.tar",
            image_env,
        )
        self.assertIn(
            "POLICY_IMAGE_URL=https://huggingface.co/datasets/NantianHe/stack-blocks-images/resolve/main/stack-blocks-policy_dp-act.tar",
            image_env,
        )

    def test_inline_comment_does_not_join_the_value(self):
        parsed = parse_simple_yaml("num_envs: 4  # at most 5\n")
        self.assertEqual(parsed["num_envs"], 4)


class PolicySlotTest(unittest.TestCase):
    def test_training_requires_real_data_or_an_offline_checkpoint(self):
        for function in (diffusion_offline, diffusion_online, act_online):
            with self.assertRaises((KeyError, ValueError)):
                function({})


class RemoteResetTimeoutTest(unittest.TestCase):
    def test_reset_waits_past_the_step_timeout(self):
        from starter.envs.remote import RemoteEnv

        server, client = socket.socketpair()
        env = RemoteEnv("127.0.0.1", 9, timeout=0.05, reset_timeout=1.0)
        env._socket = client

        def respond() -> None:
            incoming = b""
            while b"\n" not in incoming:
                incoming += server.recv(4096)
            time.sleep(0.2)
            server.sendall(b'{"observations":[{}],"infos":[{}],"privileged":[{}]}\n')

        thread = threading.Thread(target=respond)
        thread.start()
        try:
            observations, _infos = env.reset_batch([0])
        finally:
            thread.join(timeout=2)
            server.close()
            env.close()
        self.assertEqual(observations, [{}])

    def test_reset_timeout_names_the_wait(self):
        from starter.envs.remote import RemoteEnv

        _server, client = socket.socketpair()
        env = RemoteEnv("127.0.0.1", 9, timeout=0.05, reset_timeout=0.15)
        env._socket = client
        try:
            with self.assertRaises(ConnectionError) as caught:
                env.reset_batch([0])
        finally:
            _server.close()
            env.close()
        self.assertIn("did not complete reset within", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
