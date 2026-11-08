"""Single-camera ACT: ResNet18, positional-attention Transformer and CVAE.

Architecture follows https://github.com/tonyzhaozh/act, adapted to EE16.
Positions enter attention queries/keys; values carry content only.  With
``obs_steps > 1`` the shared ResNet18 encodes a short frame history and a
learned per-frame embedding preserves temporal order.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from starter.interfaces import ACTION_DIM


def sine_position(length, width, device):
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device) * (-math.log(10000) / width))
    phase = position * frequency
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)


def spatial_position(height, width, channels, device):
    # DETR-style normalized 2-D sine encoding; no fixed spatial pooling.
    frequency = 10000 ** (2 * (torch.arange(channels // 2, device=device) // 2) / (channels // 2))
    y = torch.arange(1, height + 1, device=device) * (2 * math.pi / height)
    x = torch.arange(1, width + 1, device=device) * (2 * math.pi / width)
    def encode(coordinate):
        phase = coordinate[:, None] / frequency
        return torch.stack((phase[:, 0::2].sin(), phase[:, 1::2].cos()), -1).flatten(-2)
    return torch.cat((encode(y)[:, None].expand(-1, width, -1),
                      encode(x)[None].expand(height, -1, -1)), -1).reshape(1, height * width, channels)


class EncoderLayer(nn.Module):
    def __init__(self, width, heads, dropout=0.0, dim_feedforward=None):
        super().__init__()
        hidden = int(dim_feedforward) if dim_feedforward else 4 * width
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden, width), nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)

    def forward(self, content, position, padding=None):
        query = content + position
        update = self.attention(query, query, content, key_padding_mask=padding, need_weights=False)[0]
        content = self.norm1(content + update)
        return self.norm2(content + self.ff(content))


class DecoderLayer(nn.Module):
    def __init__(self, width, heads, dropout=0.0, dim_feedforward=None):
        super().__init__()
        hidden = int(dim_feedforward) if dim_feedforward else 4 * width
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden, width), nn.Dropout(dropout))
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(width) for _ in range(3))

    def forward(self, content, query_position, memory, memory_position):
        query = content + query_position
        content = self.norm1(content + self.self_attention(query, query, content, need_weights=False)[0])
        update = self.cross_attention(content + query_position, memory + memory_position,
                                      memory, need_weights=False)[0]
        content = self.norm2(content + update)
        return self.norm3(content + self.ff(content))


class ACT(nn.Module):
    def __init__(self, config):
        super().__init__()
        from torchvision.models import resnet18
        from torchvision.ops.misc import FrozenBatchNorm2d

        width, heads, layers = (int(config[key]) for key in ("width", "heads", "layers"))
        dropout = float(config.get("dropout", 0.1))
        obs_steps = int(config.get("obs_steps", 1))
        decoder_layers = int(config.get("decoder_layers", 0)) or layers
        feedforward = int(config.get("dim_feedforward", 0)) or None
        if width % 4:
            raise ValueError("ACT width must be divisible by 4 for 2-D position encoding")
        if obs_steps < 1 or decoder_layers < 1:
            raise ValueError("obs_steps must be positive and decoder_layers at least one")
        self.chunk, self.latent_dim = int(config["chunk"]), int(config["latent_dim"])
        self.obs_steps = obs_steps
        # Only multi-frame models own this parameter, so single-frame checkpoints
        # written before history support still load bit-exactly.
        self.frame_embed = (nn.Parameter(torch.randn(1, obs_steps, 1, width) * 0.02)
                            if obs_steps > 1 else None)
        backbone = resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.image_projection = nn.Conv2d(512, width, 1)
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.state_projection = nn.Linear(ACTION_DIM, width)
        self.posterior_state = nn.Linear(ACTION_DIM, width)
        self.action_embed = nn.Linear(ACTION_DIM, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.posterior = nn.ModuleList(EncoderLayer(width, heads, dropout, feedforward) for _ in range(layers))
        self.posterior_out = nn.Linear(width, 2 * self.latent_dim)
        self.latent_projection = nn.Linear(self.latent_dim, width)
        self.extra_position = nn.Parameter(torch.randn(1, 2, width) * .02)
        self.queries = nn.Parameter(torch.randn(1, self.chunk, width) * .02)
        self.encoder = nn.ModuleList(EncoderLayer(width, heads, dropout, feedforward) for _ in range(layers))
        self.decoder = nn.ModuleList(DecoderLayer(width, heads, dropout, feedforward)
                                     for _ in range(decoder_layers))
        self.decoder_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, ACTION_DIM)

    def initialize_pretrained_backbone(self):
        # Only fresh offline training calls this. Loading checkpoints is offline-safe.
        from torchvision.models import ResNet18_Weights, resnet18
        from torchvision.ops.misc import FrozenBatchNorm2d
        source = resnet18(weights=ResNet18_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)
        self.backbone.load_state_dict(nn.Sequential(*list(source.children())[:-2]).state_dict())

    def backbone_parameters(self):
        return self.backbone.parameters()

    def forward(self, image, state, actions=None, valid=None, return_features=False):
        batch, width = len(state), self.queries.shape[-1]
        latent = state.new_zeros(batch, self.latent_dim)
        kl = state.new_zeros(())
        if actions is not None:
            if valid is None:
                raise ValueError("ACT training requires a valid action mask")
            tokens = torch.cat((self.cls.expand(batch, -1, -1), self.posterior_state(state)[:, None],
                                self.action_embed(actions)), 1)
            padding = torch.cat((torch.zeros(batch, 2, device=state.device, dtype=torch.bool), ~valid), 1)
            position = sine_position(tokens.shape[1], width, state.device)[None]
            for layer in self.posterior:
                tokens = layer(tokens, position, padding)
            mu, logvar = self.posterior_out(tokens[:, 0]).chunk(2, -1)
            logvar = logvar.clamp(-10, 10)
            latent = mu + (.5 * logvar).exp() * torch.randn_like(mu)
            kl = (-.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
        if image.dim() == 5:
            frames = int(image.shape[1])
            if self.frame_embed is None or frames != self.obs_steps:
                raise ValueError(f"ACT with obs_steps={self.obs_steps} expects {self.obs_steps} frames, got {frames}")
            flat = image.reshape(batch * frames, *image.shape[2:])
            feature = self.image_projection(self.backbone((flat - self.image_mean) / self.image_std))
            grid = int(feature.shape[-2] * feature.shape[-1])
            tokens = feature.flatten(2).transpose(1, 2).reshape(batch, frames, grid, width)
            visual = (tokens + self.frame_embed).reshape(batch, frames * grid, width)
            position = spatial_position(*feature.shape[-2:], width, image.device).repeat(1, frames, 1)
        elif image.dim() == 4:
            if self.obs_steps != 1:
                raise ValueError(f"ACT with obs_steps={self.obs_steps} expects a [B,T,3,H,W] image history")
            feature = self.image_projection(self.backbone((image - self.image_mean) / self.image_std))
            visual = feature.flatten(2).transpose(1, 2)
            position = spatial_position(*feature.shape[-2:], width, image.device)
        else:
            raise ValueError("ACT image must be [B,3,H,W] or [B,obs_steps,3,H,W]")
        memory = torch.cat((self.latent_projection(latent)[:, None], self.state_projection(state)[:, None],
                            visual), 1)
        position = torch.cat((self.extra_position, position), 1)
        for layer in self.encoder:
            memory = layer(memory, position)
        content = state.new_zeros(batch, self.chunk, width)
        for layer in self.decoder:
            content = layer(content, self.queries, memory, position)
        action = self.output(self.decoder_norm(content))
        if return_features:
            # Frozen encoder summary for the online value head.
            return action, kl, memory.mean(1)
        return action, kl
