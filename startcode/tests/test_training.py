"""CPU integration tests with small synthetic episodes; no Isaac/real demos."""
from __future__ import annotations

import asyncio
import socket
import threading
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from starter.data.demos import ChunkDataset, normalization_stats, split_episodes
from starter.envs.fake import ToyBatchEnv
from starter.envs.remote import RemoteEnv, serve
from starter.interfaces import ARM_KEYS, ARM_WIDTHS
from starter.policies.offline import train as offline
from starter.policies.online import Learner, clipped_objective, gae, train as online
from starter.policies.runtime import Policy, load_payload, project_actions

torch.set_num_threads(1)


@pytest.fixture
def config(tmp_path):
    data = tmp_path / "demos" / "data"
    data.mkdir(parents=True)
    for episode in range(3):
        with h5py.File(data / f"episode_{episode:07d}.hdf5", "w") as f:
            for kind in ("state", "action"):
                group = f.create_group(kind)
                for key, width in zip(ARM_KEYS, ARM_WIDTHS):
                    values = np.zeros((4, width), dtype=np.float32)
                    if width == 7:
                        values[:, 2], values[:, 3] = 0.2, 1.0
                        values[:, 0] = episode * 0.1 + np.arange(4) * 0.01
                    else:
                        values[:] = 0.5
                    group.create_dataset(key + "s", data=values)
            f.create_dataset("vision/cam_head/colors", data=np.full((4, 16, 16, 3), 40 + episode, np.uint8))
    return {"data_path": str(data.parent), "output_root": str(tmp_path / "outputs"),
            "seed": 3, "width": 16, "heads": 2, "layers": 1, "chunk": 2, "latent_dim": 4,
            "execute_steps": 2, "image_size": 16, "diffusion_steps": 4, "epochs": 2,
            "batch_size": 2, "workers": 0, "device": "cpu", "save_every": 1,
            "env_backend": "fake", "num_envs": 2, "finetune_episodes": 2, "finetune_updates": 1,
            "ppo_epochs": 1, "ppo_batch_size": 2, "finetune_denoising_steps": 2,
            "max_decisions_per_episode": 3, "fake_episode_steps": 4}


def changed(left, right):
    return any(not torch.equal(left[key], right[key]) for key in left)


@pytest.mark.parametrize("kind", ["act", "diffusion"])
def test_full_offline_online_save_load_resume(config, kind):
    initial = offline(kind, config)
    before = load_payload(initial)
    assert before["stage"] == "offline"
    assert before["split"]["train"] and set(before["split"]["train"]).isdisjoint(before["split"]["validation"])
    env = ToyBatchEnv(2)
    observations, _ = env.reset_batch([1, 2])
    prediction = Policy(initial).predict(observations)
    assert prediction.shape == (2, 2, 16) and np.isfinite(prediction).all()
    result = online(kind, {**config, "offline_checkpoint": str(initial)})
    after = load_payload(result)
    if kind == "act":
        assert not changed(before["model"], after["model"])
        assert after["residual"] is not None
        assert torch.count_nonzero(after["residual"]["mean.2.weight"]) > 0
    else:
        assert changed(before["ema"], after["model"])
        assert not changed(before["ema"], after["diffusion_base"])
        assert after["ema"] is None  # never deploy stale offline EMA after RL
    assert np.isfinite(Policy(result).predict(observations)).all()
    resumed = online(kind, {**config, "resume": str(result), "finetune_updates": 2})
    assert load_payload(resumed)["update"] == 1
    resumed_bc = offline(kind, {**config, "resume": str(initial), "epochs": 3})
    assert load_payload(resumed_bc)["epoch"] == 2


def test_chunk_padding_never_crosses_episodes_and_stats_only_train(config):
    train, val = split_episodes(config)
    assert set(train).isdisjoint(val)
    stats = normalization_stats(train)
    dataset = ChunkDataset(train, 3, 16, stats)
    boundary = dataset.ends[0] - 1
    sample = dataset[boundary]
    assert sample["valid"].tolist() == [True, False, False]
    np.testing.assert_array_equal(sample["action"][0], sample["action"][2])
    with h5py.File(val[0], "r+") as f:
        f["state/left_ee_poses"][:, 0] = 10000
    assert normalization_stats(train) == stats


def test_missing_data_cannot_silently_use_hold_demos(config):
    with pytest.raises(ValueError, match="real episodes"):
        offline("act", {**config, "data_path": "/definitely/missing"})


def test_gae_timeout_bootstraps_but_terminal_does_not():
    timeout = [{"reward": 1., "value": 2., "next_value": 3., "duration": 2, "boundary": True}]
    gae(timeout, .9, .95)
    assert timeout[0]["return"] == pytest.approx(1 + .9 ** 2 * 3)
    terminal = [{"reward": 1., "value": 2., "next_value": 0., "duration": 2, "boundary": True}]
    gae(terminal, .9, .95)
    assert terminal[0]["return"] == pytest.approx(1.)


def test_ppo_ratio_one_and_clipping():
    old = torch.zeros(2)
    loss, kl = clipped_objective(old.clone().requires_grad_(), old, torch.tensor([1., -1.]), .2)
    assert loss.item() == pytest.approx(0) and kl.item() == pytest.approx(0)
    loss, _ = clipped_objective(torch.log(torch.tensor([2.])), torch.zeros(1), torch.ones(1), .2)
    assert loss.item() == pytest.approx(-1.2)


@pytest.mark.parametrize("kind", ["act", "diffusion"])
def test_exact_behavior_likelihood_replay(config, kind):
    path = offline(kind, {**config, "epochs": 1})
    learner = Learner(kind, config, load_payload(path))
    observations, _ = ToyBatchEnv(2).reset_batch([0, 1])
    _, images, states, _, traces = learner.sample(observations)
    image = torch.from_numpy(images).float() / 255
    state = torch.from_numpy(states)
    trace = {key: torch.stack([t[key] for t in traces]) for key in traces[0]}
    if kind == "act":
        distribution = learner.actor.distribution(image, state, trace["base"])
        replay = distribution.log_prob(trace["latent"]).flatten(1).sum(-1)
    else:
        k = trace["time"].shape[1]
        distribution = learner.model.distribution(trace["x"].flatten(0, 1), trace["time"].flatten(),
                                                   image.repeat_interleave(k, 0), state.repeat_interleave(k, 0), .05)
        replay = distribution.log_prob(trace["next_x"].flatten(0, 1)).flatten(1).sum(-1).reshape(2, k)
    torch.testing.assert_close(replay, trace["old_logprob"], atol=1e-4, rtol=1e-5)


def test_quaternion_projection_and_finite_guard():
    action = np.zeros((2, 3, 16), np.float32)
    action[..., 7] = 2
    projected = project_actions(action)
    assert np.all(projected[..., 3] == 1) and np.all(projected[..., 11] == 1)
    assert np.all(projected[..., 7] == 1)
    with pytest.raises(ValueError):
        project_actions(np.full(16, np.nan))


@pytest.mark.parametrize("kind,script", [("act", "train_act.sh"), ("diffusion", "train_dp.sh")])
def test_shell_entrypoints_offline_and_online(config, tmp_path, kind, script):
    import yaml
    root = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml.safe_dump({**config, "epochs": 1, "policy_gpu": -1}), encoding="utf-8")
    environment = dict(os.environ)
    environment["PATH"] = str(Path(__import__("sys").executable).parent) + os.pathsep + environment["PATH"]
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    first = subprocess.run(["bash", str(root / "scripts" / script), "--config", str(cfg_path)],
                           cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    checkpoint = next((Path(config["output_root"]) / kind / "offline").rglob("policy_last.ckpt"))
    online_script = "train_act_residual.sh" if kind == "act" else "train_dppo.sh"
    second = subprocess.run(["bash", str(root / "scripts" / online_script), "--config", str(cfg_path),
                             "--set", f"offline_checkpoint={checkpoint}"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stderr


def test_native_simulator_launch_uses_random_layouts_and_selected_gpu(tmp_path, monkeypatch):
    import sys
    from starter.launch import simulator_command
    (tmp_path / "task/RoboDojo").mkdir(parents=True)
    (tmp_path / "Assets").mkdir()
    monkeypatch.setenv("ROBODOJO_ROOT", str(tmp_path))
    monkeypatch.setenv("SIM_PYTHON", sys.executable)
    command, environment, _ = simulator_command({"env_launch": "native", "env_gpu": 3,
                                                "rollout_task": "stack_blocks_random", "env_cfg": "arx_x5"})
    assert command[0] == sys.executable
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert command[command.index("--layout-source") + 1] == "random"


def test_encoded_demo_uses_official_rgb_decoder():
    module = pytest.importorskip("XPolicyLab.utils.process_data")
    from starter.data.demos import decode_rgb
    rgb = np.full((16, 16, 3), [210, 30, 70], dtype=np.uint8)
    encoded = module.encode_image_bit(rgb)
    decoded = decode_rgb(encoded)
    assert decoded.shape == rgb.shape
    np.testing.assert_allclose(decoded.astype(float), rgb, atol=5)


def test_remote_reset_step_over_real_tcp():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    thread = threading.Thread(target=serve, args=(ToyBatchEnv(2, 2), "127.0.0.1", port), daemon=True)
    thread.start()
    env = RemoteEnv("127.0.0.1", port, timeout=3, reset_timeout=5)
    try:
        obs, _ = env.reset_batch([0, 1])
        assert len(obs) == 2 and obs[0]["image"].shape == (24, 24, 3)
        assert env.privileged(0) == {}
        results = env.step_batch([o["proprio"] for o in obs], [0, 1])
        assert [r[1] for r in results] == [0., 0.]
        results = env.step_batch([o["proprio"] for o in obs], [0, 1])
        assert all(r[2] and r[1] == 1 for r in results)
    finally:
        env.close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_xpolicylab_official_ws_server_client(config, monkeypatch):
    # Run when the pinned XPolicyLab is bootstrapped. The main suite still
    # runs independently of optional external repositories.
    module = pytest.importorskip("client_server.ws.model_server")
    from client_server.ws.model_client import WsModelClient

    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("eval_server", root / "policies" / "eval_server.py")
    eval_server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_server)

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    path = offline("act", {**config, "epochs": 1})
    model = eval_server.ServePolicy(Path(path), "arx_x5", "ee", "cpu")
    ready = threading.Event()
    info = {}
    loop = asyncio.new_event_loop()
    server = module.PolicyServer(model, module.PolicyServerConfig(host="127.0.0.1", port=0))

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        info["url"] = server.url
        ready.set()
        loop.run_forever()
        loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(10)
    client = WsModelClient(url=info["url"], evaluation_id="cpu-test", trial_id="one", action_case_id="test", repeat_index=0,
                           request_timeout_s=5, handshake_timeout_s=5, max_connect_attempts=1, close_timeout_s=2)
    observations, _ = ToyBatchEnv(2).reset_batch([0, 1])
    try:
        client.call(func_name="reset")
        client.call(func_name="update_obs", obs=observations[0])
        actions = client.call(func_name="get_action")
        assert len(actions) == 2 and len(actions[0]["left_ee_pose"]) == 7
        for index, obs in enumerate(observations):
            obs["env_idx"] = index
        client.call(func_name="update_obs_batch", obs=observations)
        chunks = client.call(func_name="get_action_batch", obs=[1, 0])
        assert len(chunks) == 2
    finally:
        client.close()
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
