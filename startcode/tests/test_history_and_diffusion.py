"""Multi-frame observations, temporal ensembling, DDIM and min-SNR paths."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from starter.data.demos import ChunkDataset, normalization_stats, split_episodes
from starter.envs.fake import ToyBatchEnv
from starter.interfaces import ARM_KEYS, ARM_WIDTHS
from starter.policies.offline import train as offline
from starter.policies.online import Learner, collect, train as online
from starter.policies.runtime import (FrameHistory, Policy, TemporalEnsemble, build_model,
                                      load_payload, model_config)

torch.set_num_threads(1)


def write_demos(tmp_path):
    import h5py

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
            frames = np.zeros((4, 16, 16, 3), np.uint8)
            for step in range(4):
                frames[step] = 40 + episode * 10 + step * 3
            f.create_dataset("vision/cam_head/colors", data=frames)


@pytest.fixture
def config(tmp_path):
    write_demos(tmp_path)
    return {"data_path": str(tmp_path / "demos"), "output_root": str(tmp_path / "outputs"),
            "pretrained_backbone": False, "seed": 3, "width": 16, "heads": 2, "layers": 1,
            "chunk": 2, "latent_dim": 4, "execute_steps": 2, "image_size": 16,
            "diffusion_steps": 4, "epochs": 1, "batch_size": 2, "workers": 0,
            "device": "cpu", "save_every": 1, "env_backend": "fake", "num_envs": 2,
            "finetune_episodes": 2, "finetune_updates": 1, "ppo_epochs": 1,
            "ppo_batch_size": 2, "finetune_denoising_steps": 2,
            "max_decisions_per_episode": 3, "fake_episode_steps": 4}


def test_act_multiframe_forward_and_capacity_options():
    cfg = model_config({"width": 16, "heads": 2, "layers": 1, "chunk": 3, "latent_dim": 4,
                        "execute_steps": 2, "backbone": "resnet18", "obs_steps": 2,
                        "decoder_layers": 2, "dim_feedforward": 32})
    model = build_model("act", cfg)
    assert len(model.decoder) == 2 and model.frame_embed is not None
    assert tuple(model.frame_embed.shape) == (1, 2, 1, 16)
    image = torch.rand(2, 2, 3, 64, 96, requires_grad=True)
    state, action = torch.randn(2, 16), torch.randn(2, 3, 16)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    prediction, kl = model(image, state, action, valid)
    assert prediction.shape == (2, 3, 16) and torch.isfinite(kl)
    (prediction.square().mean() + kl).backward()
    assert image.grad is not None and image.grad.abs().sum() > 0


def test_single_frame_act_keeps_legacy_parameter_set():
    cfg = model_config({"width": 16, "heads": 2, "layers": 2, "chunk": 3, "latent_dim": 4,
                        "execute_steps": 2})
    model = build_model("act", cfg)
    assert "frame_embed" not in dict(model.named_parameters())
    assert len(model.decoder) == 2


def test_model_config_rejects_invalid_history_settings():
    with pytest.raises(ValueError, match="resnet18"):
        model_config({"width": 16, "heads": 2, "layers": 1, "chunk": 2,
                      "execute_steps": 1, "obs_steps": 2, "backbone": "cnn"})
    with pytest.raises(ValueError, match="inference_steps"):
        model_config({"width": 16, "heads": 2, "layers": 1, "chunk": 2,
                      "execute_steps": 1, "diffusion_steps": 4, "inference_steps": 5})


def test_diffusion_multiframe_min_snr_and_ddim():
    base = {"width": 32, "heads": 4, "layers": 2, "chunk": 3, "execute_steps": 2,
            "backbone": "resnet18", "obs_steps": 2, "diffusion_steps": 4}
    plain = build_model("diffusion", model_config(base)).eval()
    weighted = build_model("diffusion", model_config({**base, "min_snr_gamma": 5.0,
                                                      "inference_steps": 2})).eval()
    weighted.load_state_dict(plain.state_dict())
    assert weighted.ddim and len(weighted.schedule) == 2 and weighted.schedule[-1][1] == -1
    image = torch.rand(32, 2, 3, 48, 64)
    state, action = torch.randn(32, 16), torch.randn(32, 3, 16)
    valid = torch.ones(32, 3, dtype=torch.bool)
    torch.manual_seed(0)
    first = plain.training_loss(image, state, action, valid)
    torch.manual_seed(0)
    second = weighted.training_loss(image, state, action, valid)
    assert torch.isfinite(first) and torch.isfinite(second)
    assert not torch.allclose(first, second)
    with torch.no_grad():
        torch.manual_seed(2)
        a, _ = weighted.sample(image, state)
        torch.manual_seed(2)
        b, _ = weighted.sample(image, state)
        x, chain = weighted.sample(image, state, stochastic_steps=2, prefix_steps=2)
    assert a.shape == (32, 3, 16) and torch.equal(a, b)
    assert x.shape == (32, 3, 16) and len(chain) == 2
    assert chain[0][4].shape == (32,) and chain[-1][2].tolist() == [weighted.schedule[-1][0]] * 32


def test_ddim_behavior_likelihood_replay(config):
    settings = {**config, "chunk": 3, "execute_steps": 2, "backbone": "resnet18",
                "diffusion_steps": 6, "inference_steps": 3}
    path = offline("diffusion", settings)
    learner = Learner("diffusion", settings, load_payload(path))
    observations, _ = ToyBatchEnv(2).reset_batch([0, 1])
    _, images, states, _, traces = learner.sample(observations)
    image = torch.from_numpy(images).float() / 255
    state = torch.from_numpy(states)
    trace = {key: torch.stack([item[key] for item in traces]) for key in traces[0]}
    k = trace["time"].shape[1]
    assert k == 2 and trace["prev_time"].shape == trace["time"].shape
    distribution = learner.model.distribution(
        trace["x"].flatten(0, 1), trace["time"].flatten(),
        image.repeat_interleave(k, 0), state.repeat_interleave(k, 0), 0.05,
        prev_time=trace["prev_time"].flatten())
    replay = distribution.log_prob(trace["next_x"].flatten(0, 1))[..., :2, :].sum(-1).reshape(2, k, 2)
    torch.testing.assert_close(replay, trace["old_logprob"], atol=1e-4, rtol=1e-5)


def test_frame_history_pads_oldest_and_resets():
    history = FrameHistory(3, 16)
    frame = lambda value: np.full((16, 16, 3), value, np.uint8)
    stack = history.push("a", frame(50))
    assert stack.shape == (3, 3, 16, 16) and set(stack.reshape(-1)) == {50}
    history.push("a", frame(100))
    np.testing.assert_array_equal(history.stack("a")[:, 0, 0, 0], [50, 50, 100])
    history.push("a", frame(150))
    history.push("a", frame(200))
    np.testing.assert_array_equal(history.stack("a")[:, 0, 0, 0], [100, 150, 200])
    history.reset()
    assert history.slots == {}


def test_temporal_ensemble_uses_official_older_first_weights():
    ensemble = TemporalEnsemble(3, 0.1)
    first = ensemble.predict(0, np.array([[1.0], [2.0], [3.0]], np.float32))
    assert first == pytest.approx(1.0)
    second = ensemble.predict(0, np.array([[10.0], [20.0], [30.0]], np.float32))
    expected = (1.0 * 2.0 + np.exp(-0.1) * 10.0) / (1.0 + np.exp(-0.1))
    assert second == pytest.approx(expected, rel=1e-6)
    ensemble.reset()
    assert ensemble.predict(0, np.array([[10.0], [20.0], [30.0]], np.float32)) == pytest.approx(10.0)


def test_dataset_multiframe_padding_and_frame_dropout(config):
    train, _ = split_episodes(config)
    stats = normalization_stats(train)
    dataset = ChunkDataset(train, 2, 16, stats, obs_steps=2)
    start = dataset[0]
    assert start["image"].shape == (2, 3, 16, 16)
    np.testing.assert_array_equal(start["image"][0], start["image"][1])
    second = dataset[1]
    assert not np.array_equal(second["image"][0], second["image"][1])
    dropped = ChunkDataset(train, 2, 16, stats, obs_steps=2, frame_dropout=1.0)
    np.testing.assert_array_equal(dropped[1]["image"][0], dropped[1]["image"][1])


def test_learner_history_and_policy_single_step(config):
    settings = {**config, "chunk": 3, "execute_steps": 2, "backbone": "resnet18",
                "obs_steps": 2, "temporal_ensemble": True}
    path = offline("act", settings)
    learner = Learner("act", settings, load_payload(path))
    observations, _ = ToyBatchEnv(2).reset_batch([0, 1])
    learner.reset_history()
    learner.observe(observations, [0, 1])
    _, images, _, _, _ = learner.sample(observations, slots=[0, 1])
    assert images.shape == (2, 2, 3, 16, 16)
    records, _ = collect(learner, ToyBatchEnv(2), {**settings, "max_decisions_per_episode": 2}, 0)
    assert records and records[0]["image"].shape == (2, 3, 16, 16)

    policy = Policy(path)
    actions = policy.predict(observations)
    assert actions.shape == (2, 1, 16) and np.isfinite(actions).all()
    policy.predict(observations)
    assert len(policy.history.slots[0]) == 2
    policy.reset()
    assert policy.history.slots == {} and policy.ensemble.slots == {}


@pytest.mark.parametrize("kind", ["act", "diffusion"])
def test_online_multiframe_update_end_to_end(config, kind):
    settings = {**config, "backbone": "resnet18", "obs_steps": 2, "chunk": 3,
                "execute_steps": 2, "diffusion_steps": 4, "inference_steps": 2,
                "finetune_updates": 1, "finetune_episodes": 2,
                "max_decisions_per_episode": 2, "suffix_l2": 1e-3}
    path = offline(kind, settings)
    result = online(kind, {**settings, "offline_checkpoint": str(path)})
    payload = load_payload(result)
    assert payload["stage"] == "online" and payload["model_config"]["obs_steps"] == 2
    observations, _ = ToyBatchEnv(2).reset_batch([0, 1])
    actions = Policy(result).predict(observations)
    assert actions.shape == (2, 1, 16) and np.isfinite(actions).all()


def test_augmentation_parameters_are_shared_across_history():
    from starter.data.demos import augmentation_parameters, augment_rgb

    frame = np.full((24, 32, 3), 128, np.uint8)
    parameters = augmentation_parameters()
    first = augment_rgb(frame, parameters)
    second = augment_rgb(frame.copy(), parameters)
    np.testing.assert_array_equal(first, second)
