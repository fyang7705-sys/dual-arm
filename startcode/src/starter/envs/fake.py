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
