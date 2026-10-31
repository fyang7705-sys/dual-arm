"""A stand-in task used to exercise the adapter without Isaac Sim."""

from __future__ import annotations

from starter.interfaces import ARM_KEYS, ARM_WIDTHS, BLOCK_LABELS


class FakeParser:
    def __init__(self) -> None:
        self.poses = {
            label: ([0.0, 0.0, 0.01 * (index + 1)], [1.0, 0.0, 0.0, 0.0])
            for index, label in enumerate(BLOCK_LABELS)
        }

    def get_label_pose(self, label: str):
        pos, rot = self.poses[label]
        return [pos], [rot]


class FakeRewardManager:
    def __init__(self, score: float = 0.0, success_reward: float = 0.0) -> None:
        self.score = score
        self.success_reward = success_reward
        self.func_parser = FakeParser()
        self.step_calls = 0

    def get_reward(self, final_check: bool = False):
        del final_check
        return [self.success_reward]

    def get_score(self, reward_lst=None):
        del reward_lst
        return [self.score]

    def step(self, env_idx_list=None) -> None:
        del env_idx_list
        self.step_calls += 1


class FakeTask:
    """Records actions and serves a full RoboDojo-shaped observation."""

    def __init__(self, step_lim: int = 4, score: float = 0.0) -> None:
        self.step_lim = step_lim
        self.reward_manager = FakeRewardManager(score=score)
        self.actions: list[dict] = []
        self.reset_seeds: list = []
        self.end_checks = 0

    def reset(self, seed=None, options=None) -> None:
        del options
        self.reset_seeds.append(seed)
        self.actions.clear()

    def run_reward(self) -> None:
        return None

    def get_score(self) -> None:
        return None

    def is_episode_end(self) -> bool:
        self.end_checks += 1
        return True

    def take_action(self, action: dict) -> None:
        if self.is_episode_end():
            raise RuntimeError("training must suspend is_episode_end before take_action")
        self.actions.append(action)
        self.reward_manager.step()

    def get_obs(self) -> dict:
        color = [[[index, index, index] for index in range(2)] for _ in range(2)]
        state = {key: [0.0] * width for key, width in zip(ARM_KEYS, ARM_WIDTHS)}
        state["left_arm_joint_state"] = [0.0] * 6
        state["block_0"] = [1.0, 2.0, 3.0]
        return {
            "vision": {
                "cam_head": {"color": color},
                "cam_left_wrist": {"color": color},
                "cam_right_wrist": {"color": color},
            },
            "state": state,
            "instruction": "Stack the three blocks with different textures.",
        }


class ToyBatchEnv:
    """Explicit CPU integration-test environment, NOT a robotics benchmark.

    Terminal reward is always one, to exercise credit assignment and IO on a
    tiny test. Its success rate has no meaning for block stacking.
    """
    def __init__(self, num_envs=1, step_limit=4):
        self.num_envs, self.step_limit = num_envs, step_limit

    def _observation(self, index):
        import numpy as np
        pose = [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.5] * 2
        pose[0] = self.steps[index] * 0.01
        return {"image": np.full((24, 24, 3), 40 + self.steps[index], dtype=np.uint8),
                "proprio": pose, "camera": "cam_head"}

    def reset_batch(self, seeds):
        if len(seeds) != self.num_envs:
            raise ValueError("Seed count mismatch")
        self.steps = [0] * self.num_envs
        return [self._observation(i) for i in range(self.num_envs)], [{} for _ in seeds]

    def step_batch(self, actions, active):
        if len(actions) != len(active):
            raise ValueError("Action count mismatch")
        result = []
        for index, action in zip(active, actions):
            if len(action) != 16:
                raise ValueError("Expected a 16-D action")
            self.steps[index] += 1
            done = self.steps[index] >= self.step_limit
            result.append((self._observation(index), float(done), done, False,
                           {"success": done, "external_score": 100.0 if done else 0.0, "test_only": True}))
        return result

    def privileged_batch(self, active):
        return [{} for _ in active]

    def close(self):
        pass
