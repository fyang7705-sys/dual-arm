"""ACT architecture checks, independent of Isaac and downloaded weights."""
import numpy as np
import torch

from starter.data.demos import resize_rgb
from starter.policies.runtime import build_model, model_config


def test_act_visual_gradient_padding_and_zero_query():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = model_config({"width": 16, "heads": 2, "layers": 1, "chunk": 3,
                        "latent_dim": 4, "execute_steps": 2})
    model = build_model("act", cfg).eval()
    image = torch.rand(2, 3, 64, 96, requires_grad=True)
    state, action = torch.randn(2, 16), torch.randn(2, 3, 16)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    captured = []
    handle = model.decoder[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    torch.manual_seed(9)
    prediction, kl = model(image, state, action, valid)
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
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert model.backbone[0].weight.grad.abs().sum() > 0
    with torch.no_grad():
        baseline, zero_kl = model(image, state)
        changed, _ = model(torch.zeros_like(image), state)
        repeated, _ = model(image, state)
    assert not torch.allclose(baseline, changed)
    torch.testing.assert_close(baseline, repeated)
    assert zero_kl == 0


def test_rectangular_resize():
    assert resize_rgb(np.zeros((48, 64, 3), np.uint8), [24, 32]).shape == (24, 32, 3)
