"""ACT architecture checks, independent of Isaac and downloaded weights."""
import numpy as np
import pytest
import torch

from starter.data.demos import resize_rgb
from starter.policies.runtime import build_model, model_config


@pytest.mark.parametrize("cameras", [["cam_head"], ["cam_head", "cam_right_wrist", "cam_left_wrist"]])
def test_act_visual_gradient_padding_and_zero_query(cameras):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = model_config({"width": 16, "heads": 2, "layers": 1, "chunk": 3,
                        "latent_dim": 4, "execute_steps": 2, "camera_names": cameras})
    model = build_model("act", cfg).eval()
    image = torch.rand(2, len(cameras), 3, 64, 96, requires_grad=True)
    state, action = torch.randn(2, 16), torch.randn(2, 3, 16)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    captured = []
    projected, memory = [], []
    handle = model.decoder[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    projection_handle = model.image_projection.register_forward_hook(
        lambda module, args, result: projected.append(result.detach()))
    encoder_handle = model.encoder[0].register_forward_pre_hook(lambda module, args: memory.append(args))
    torch.manual_seed(9)
    prediction, kl = model(image, state, action, valid)
    assert len(projected) == len(cameras)
    # Official ACT folds cameras into feature-map width before flattening.
    expected = torch.cat(projected, dim=3).flatten(2).transpose(1, 2)
    torch.testing.assert_close(memory[0][0][:, 2:], expected)
    positions = memory[0][1][:, 2:].reshape(1, 2, len(cameras), 3, 16)
    for camera in range(1, len(cameras)):
        torch.testing.assert_close(positions[:, :, camera], positions[:, :, 0])
    projection_handle.remove()
    encoder_handle.remove()
    altered = action.clone()
    altered[~valid] = 1000
    torch.manual_seed(9)
    masked_prediction, masked_kl = model(image, state, altered, valid)
    torch.testing.assert_close(prediction, masked_prediction)
    torch.testing.assert_close(kl, masked_kl)
    assert prediction.shape == (2, 3, 16) and torch.isfinite(kl)
    assert torch.count_nonzero(captured[0]) == 0
    handle.remove()
    (prediction.square().mean() + .01 * kl).backward()
    assert image.grad is not None
    assert (image.grad.abs().sum(dim=(0, 2, 3, 4)) > 0).all()
    assert model.backbone[0].weight.grad.abs().sum() > 0
    with torch.no_grad():
        baseline, zero_kl = model(image, state)
        repeated, _ = model(image, state)
        for camera in range(len(cameras)):
            altered_image = image.detach().clone()
            altered_image[:, camera] = 0
            changed, _ = model(altered_image, state)
            assert not torch.allclose(baseline, changed)
    torch.testing.assert_close(baseline, repeated)
    assert zero_kl == 0
    with pytest.raises(ValueError, match="ACT requires"):
        model(image[:, 0], state)


def test_rectangular_resize():
    assert resize_rgb(np.zeros((48, 64, 3), np.uint8), [24, 32]).shape == (24, 32, 3)
