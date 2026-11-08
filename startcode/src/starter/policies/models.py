"""Compact ACT-CVAE, conditional diffusion, residual actor and visual critic.

    ACT uses a pretrained ResNet34 on fresh offline runs.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from starter.interfaces import ACTION_DIM
from starter.policies.act_model import ACT
from starter.policies.dp_model import Diffusion


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
        if image.ndim == 5:
            image, state = image[:, -1], state[:, -1]
        return self.head(self.condition(image, state).mean(1)).squeeze(-1)
