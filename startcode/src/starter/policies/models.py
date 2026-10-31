"""Compact ACT-CVAE, conditional diffusion, residual actor and visual critic.

    Independently implemented task baselines; no pretrained backbone downloads.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from starter.interfaces import ACTION_DIM


class Vision(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(4, 32), nn.SiLU(),
                                 nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
                                 nn.Conv2d(64, width, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.position = nn.Parameter(torch.randn(1, 16, width) * 0.02)

    def forward(self, image):
        return self.net(image).flatten(2).transpose(1, 2) + self.position


class Condition(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.vision = Vision(width)
        self.proprio = nn.Linear(ACTION_DIM, width)

    def forward(self, image, state):
        return torch.cat((self.vision(image), self.proprio(state).unsqueeze(1)), dim=1)


class ACT(nn.Module):
    def __init__(self, config):
        super().__init__()
        w, h, z = int(config["width"]), int(config["chunk"]), int(config["latent_dim"])
        heads, layers = int(config["heads"]), int(config["layers"])
        self.chunk, self.latent_dim = h, z
        self.condition = Condition(w)
        enc = nn.TransformerEncoderLayer(w, heads, 4 * w, dropout=0, batch_first=True)
        self.posterior = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.action_embed = nn.Linear(ACTION_DIM, w)
        self.cls = nn.Parameter(torch.zeros(1, 1, w))
        self.posterior_pos = nn.Parameter(torch.randn(1, h + 2, w) * 0.02)
        self.posterior_out = nn.Linear(w, z * 2)
        self.latent_embed = nn.Linear(z, w)
        self.queries = nn.Parameter(torch.randn(1, h, w) * 0.02)
        dec = nn.TransformerDecoderLayer(w, heads, 4 * w, dropout=0, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, layers)
        self.output = nn.Linear(w, ACTION_DIM)

    def forward(self, image, state, actions=None, valid=None):
        batch = len(state)
        memory = self.condition(image, state)
        kl = state.new_zeros(())
        latent = state.new_zeros(batch, self.latent_dim)
        if actions is not None:
            tokens = torch.cat((self.cls.expand(batch, -1, -1), memory[:, -1:], self.action_embed(actions)), 1)
            padding = torch.cat((torch.zeros(batch, 2, device=state.device, dtype=torch.bool), ~valid), 1)
            encoded = self.posterior(tokens + self.posterior_pos, src_key_padding_mask=padding)[:, 0]
            mu, logvar = self.posterior_out(encoded).chunk(2, -1)
            logvar = logvar.clamp(-10, 10)
            latent = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
            kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
        memory = torch.cat((memory, self.latent_embed(latent).unsqueeze(1)), 1)
        prediction = self.output(self.decoder(self.queries.expand(batch, -1, -1), memory))
        return prediction, kl


class Denoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        w, h = int(config["width"]), int(config["chunk"])
        self.condition = Condition(w)
        self.chunk = h
        self.input = nn.Linear(h * ACTION_DIM + 2 * w, 4 * w)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(4 * w), nn.Linear(4 * w, 4 * w), nn.SiLU(),
                                                nn.Linear(4 * w, 4 * w)) for _ in range(int(config["layers"])))
        self.out = nn.Linear(4 * w, h * ACTION_DIM)
        self.width = w

    def forward(self, x, timestep, image, state):
        half = self.width // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / max(half - 1, 1))
        phase = timestep[:, None].float() * freqs[None, :]
        time = torch.cat((phase.sin(), phase.cos()), -1)
        context = self.condition(image, state).mean(1)
        hidden = torch.nn.functional.silu(self.input(torch.cat((x.flatten(1), context, time), -1)))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.out(hidden).reshape(-1, self.chunk, ACTION_DIM)


class Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.denoiser = Denoiser(config)
        self.chunk, self.steps = int(config["chunk"]), int(config["diffusion_steps"])
        if self.steps < 2:
            raise ValueError("diffusion_steps must be >=2")
        # Cosine schedule is well behaved even for short CPU smoke schedules.
        t = torch.linspace(0, self.steps, self.steps + 1)
        abar = torch.cos(((t / self.steps + 0.008) / 1.008) * math.pi / 2).square()
        abar = abar / abar[0]
        beta = (1 - abar[1:] / abar[:-1]).clamp(1e-5, 0.999)
        alpha = 1 - beta
        cumulative = torch.cumprod(alpha, 0)
        previous = torch.cat((torch.ones(1), cumulative[:-1]))
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)
        self.register_buffer("abar", cumulative)
        self.register_buffer("previous", previous)

    def training_loss(self, image, state, action, valid):
        time = torch.randint(self.steps, (len(action),), device=action.device)
        noise = torch.randn_like(action)
        abar = self.abar[time, None, None]
        noisy = abar.sqrt() * action + (1 - abar).sqrt() * noise
        prediction = self.denoiser(noisy, time, image, state)
        return (((prediction - noise).square().mean(-1) * valid).sum() / valid.sum().clamp_min(1))

    def distribution(self, x, time, image, state, min_std):
        eps = self.denoiser(x, time, image, state)
        a, b = self.abar[time, None, None], self.previous[time, None, None]
        beta = self.beta[time, None, None]
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-5, 5)
        mean = beta * b.sqrt() / (1 - a) * x0 + (1 - b) * self.alpha[time, None, None].sqrt() / (1 - a) * x
        std = (beta * (1 - b) / (1 - a)).sqrt().clamp_min(min_std)
        return Normal(mean, std.expand_as(mean))

    @torch.no_grad()
    def sample(self, image, state, stochastic_steps=0, min_std=0.05, prefix_model=None, finetune_steps=None):
        x = torch.randn(len(state), self.chunk, ACTION_DIM, device=state.device)
        trace = []
        for step in reversed(range(self.steps)):
            time = torch.full((len(state),), step, dtype=torch.long, device=state.device)
            threshold = stochastic_steps if finetune_steps is None else finetune_steps
            predictor = prefix_model if prefix_model is not None and step >= threshold else self
            dist = predictor.distribution(x, time, image, state, min_std)
            if step < stochastic_steps:
                nxt = dist.sample()
                trace.append((x.clone(), nxt.clone(), time, dist.log_prob(nxt).flatten(1).sum(-1)))
            else:
                nxt = dist.mean
            x = nxt
        return x, trace


class ResidualActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        w, h = int(config["width"]), int(config["chunk"])
        self.condition = Condition(w)
        self.mean = nn.Sequential(nn.Linear(w + h * ACTION_DIM, w), nn.Tanh(), nn.Linear(w, h * ACTION_DIM))
        nn.init.zeros_(self.mean[-1].weight)
        nn.init.zeros_(self.mean[-1].bias)
        self.log_std = nn.Parameter(torch.full((h, ACTION_DIM), math.log(float(config.get("residual_std", 0.3)))))
        self.chunk = h

    def distribution(self, image, state, base):
        features = torch.cat((self.condition(image, state).mean(1), base.flatten(1)), -1)
        mean = self.mean(features).reshape(-1, self.chunk, ACTION_DIM)
        return Normal(mean, self.log_std.clamp(-5, 0).exp().expand_as(mean))


class Critic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.condition = Condition(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, image, state):
        return self.head(self.condition(image, state).mean(1)).squeeze(-1)
