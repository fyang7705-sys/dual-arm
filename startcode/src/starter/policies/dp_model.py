"""ResNet18 + XPolicyLab temporal U-Net, DDPM sampling and DPPO transitions."""
import sys

import torch
from torch import nn
from torch.distributions import Normal
from diffusers import DDPMScheduler

from starter.interfaces import ACTION_DIM
from starter.paths import REPO_ROOT

# Upstream uses absolute diffusion_policy imports internally.
DP_ROOT = str(REPO_ROOT / "third_party/XPolicyLab/policy/DP")
if DP_ROOT not in sys.path:
    sys.path.insert(0, DP_ROOT)
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.model_getter import get_resnet


class Denoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.obs_steps = int(config["n_obs_steps"])
        size = config["image_size"]
        height, width = (size, size) if isinstance(size, int) else size
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta={"obs": {"head_cam": {"shape": [3, height, width], "type": "rgb"},
                                "agent_pos": {"shape": [ACTION_DIM], "type": "low_dim"}}},
            rgb_model=get_resnet("resnet18", weights=None), use_group_norm=True, imagenet_norm=True)
        self.unet = ConditionalUnet1D(
            input_dim=ACTION_DIM, global_cond_dim=self.obs_steps * (512 + ACTION_DIM),
            diffusion_step_embed_dim=int(config["diffusion_step_embed_dim"]),
            down_dims=list(config["down_dims"]), kernel_size=5, n_groups=8, cond_predict_scale=True)

    def forward(self, action, timestep, image, state):
        if image.ndim != 5 or state.ndim != 3 or image.shape[1] != self.obs_steps:
            raise ValueError("DP requires [B,n_obs_steps,3,H,W] images and [B,n_obs_steps,16] states")
        batch = image.shape[0]
        features = self.obs_encoder({"head_cam": image.flatten(0, 1), "agent_pos": state.flatten(0, 1)})
        condition = features.reshape(batch, -1)
        return self.unet(action, timestep, global_cond=condition)


class Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk, self.steps = int(config["chunk"]), int(config["diffusion_steps"])
        self.inference_steps = int(config["inference_steps"])
        divisor = 2 ** (len(config["down_dims"]) - 1)
        if self.chunk < divisor or self.chunk % divisor:
            raise ValueError(f"DP chunk must be a positive multiple of {divisor}")
        if not 1 <= self.inference_steps <= self.steps or self.steps < 2:
            raise ValueError("Require 1 <= inference_steps <= diffusion_steps and diffusion_steps >=2")
        if int(config["n_obs_steps"]) < 1:
            raise ValueError("n_obs_steps must be positive")
        self.denoiser = Denoiser(config)
        self.scheduler = DDPMScheduler(num_train_timesteps=self.steps, beta_schedule="squaredcos_cap_v2",
                                       variance_type="fixed_small", clip_sample=True, prediction_type="epsilon")
        self.scheduler.set_timesteps(self.inference_steps)
        self.register_buffer("abar", self.scheduler.alphas_cumprod.clone())
        previous = torch.ones(self.steps)
        for time in self.scheduler.timesteps.tolist():
            prev = int(self.scheduler.previous_timestep(time))
            previous[time] = self.abar[prev] if prev >= 0 else 1
        self.register_buffer("previous", previous)

    def training_loss(self, image, state, action, valid):
        time = torch.randint(self.steps, (len(action),), device=action.device)
        noise = torch.randn_like(action)
        alpha = self.abar[time, None, None]
        noisy = alpha.sqrt() * action + (1 - alpha).sqrt() * noise
        prediction = self.denoiser(noisy, time, image, state)
        return ((prediction - noise).square().mean(-1) * valid).sum() / valid.sum().clamp_min(1)

    def distribution(self, x, time, image, state, min_std):
        """DDPMScheduler posterior mean/variance, with explicit DPPO std floor.

        The previous alpha uses the actual inference timetable, including skips.
        Sampling and PPO replay both use this exact distribution.
        """
        eps = self.denoiser(x, time, image, state)
        alpha, prev = self.abar[time, None, None], self.previous[time, None, None]
        beta = 1 - alpha / prev
        clean = ((x - (1 - alpha).sqrt() * eps) / alpha.sqrt()).clamp(-1, 1)
        mean = prev.sqrt() * beta / (1 - alpha) * clean + (alpha / prev).sqrt() * (1 - prev) / (1 - alpha) * x
        std = (beta * (1 - prev) / (1 - alpha)).clamp_min(1e-20).sqrt().clamp_min(min_std)
        return Normal(mean, std.expand_as(mean))

    @torch.no_grad()
    def sample(self, image, state, stochastic_steps=0, min_std=.05, prefix_model=None, finetune_steps=None):
        if not 0 <= stochastic_steps <= self.inference_steps:
            raise ValueError("stochastic_steps must not exceed inference_steps")
        x = torch.randn(len(state), self.chunk, ACTION_DIM, device=state.device)
        trace = []
        suffix = stochastic_steps if finetune_steps is None else finetune_steps
        for index, scalar_time in enumerate(self.scheduler.timesteps):
            remaining = self.inference_steps - index
            predictor = prefix_model if prefix_model is not None and remaining > suffix else self
            time = torch.full((len(state),), int(scalar_time), device=state.device, dtype=torch.long)
            if remaining <= stochastic_steps:
                dist = predictor.distribution(x, time, image, state, min_std)
                nxt = dist.sample()
                trace.append((x.clone(), nxt.clone(), time, dist.log_prob(nxt).flatten(1).sum(-1)))
            else:
                eps = predictor.denoiser(x, time, image, state)
                nxt = self.scheduler.step(eps, int(scalar_time), x).prev_sample
            x = nxt
        return x, trace
