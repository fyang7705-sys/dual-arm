"""ResNet18 spatial-token diffusion, photometric augmentation, quaternion canonicalization."""
import numpy as np
import torch

from starter.data.demos import augment_rgb, canonicalize_quaternion
from starter.policies.runtime import build_model, model_config, observation_arrays

torch.set_num_threads(1)


def test_diffusion_transformer_denoiser_gradient_and_shape():
    torch.manual_seed(0)
    cfg = model_config({"width": 32, "heads": 4, "layers": 2, "chunk": 3, "execute_steps": 2,
                        "backbone": "resnet18", "diffusion_steps": 4})
    model = build_model("diffusion", cfg).eval()
    image = torch.rand(2, 3, 64, 96, requires_grad=True)
    state, action = torch.randn(2, 16), torch.randn(2, 3, 16)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    torch.manual_seed(1)
    loss = model.training_loss(image, state, action, valid)
    loss.backward()
    assert torch.isfinite(loss) and image.grad.abs().sum() > 0
    assert list(model.backbone_parameters()) and model.denoiser.vision.backbone[0].weight.grad.abs().sum() > 0
    with torch.no_grad():
        torch.manual_seed(2)
        first, _ = model.sample(image, state)
        torch.manual_seed(2)
        second, _ = model.sample(image, state)
    assert first.shape == (2, 3, 16) and torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_legacy_cnn_backbone_still_builds():
    cfg = model_config({"width": 16, "heads": 2, "layers": 1, "chunk": 2,
                        "execute_steps": 2, "diffusion_steps": 2})
    model = build_model("diffusion", cfg).eval()
    assert list(model.backbone_parameters()) == []
    with torch.no_grad():
        loss = model.training_loss(torch.rand(1, 3, 16, 16), torch.randn(1, 16),
                                   torch.randn(1, 2, 16), torch.ones(1, 2, dtype=torch.bool))
    assert torch.isfinite(loss)


def test_act_dropout_active_only_in_train_mode():
    cfg = model_config({"width": 16, "heads": 2, "layers": 1, "chunk": 3,
                        "latent_dim": 4, "execute_steps": 2, "dropout": 0.5})
    model = build_model("act", cfg)
    image, state = torch.rand(2, 3, 64, 96), torch.randn(2, 16)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(image, state)[0], model(image, state)[0])
    model.train()
    with torch.no_grad():
        assert not torch.allclose(model(image, state)[0], model(image, state)[0])


def test_augment_rgb_photometric_only():
    np.random.seed(0)
    frame = np.full((24, 32, 3), 128, np.uint8)
    augmented = augment_rgb(frame)
    assert augmented.shape == frame.shape and augmented.dtype == np.uint8
    assert not np.array_equal(augmented, frame)


def test_canonicalize_quaternion_flips_negative_w_only():
    value = np.array([[1, 2, 3, -0.5, 0.5, -0.5, 0.5],
                      [4, 5, 6, 0.5, 0.5, 0.5, 0.5]], np.float32)
    result = canonicalize_quaternion(value)
    assert result[0, 3] > 0 and result[1, 3] > 0
    np.testing.assert_allclose(result[0, 4:], -value[0, 4:])
    np.testing.assert_allclose(result[1], value[1])
    np.testing.assert_allclose(result[:, :3], value[:, :3])


def test_observation_state_quaternion_matches_training_canonicalization():
    stats = {"state_mean": [0.0] * 16, "state_std": [1.0] * 16}
    proprio = [0.0] * 16
    proprio[3], proprio[11] = -1.0, -1.0
    _, states = observation_arrays([{"image": np.zeros((16, 16, 3), np.uint8), "proprio": proprio}], stats, 16)
    assert states[0, 3] == 1.0 and states[0, 11] == 1.0
