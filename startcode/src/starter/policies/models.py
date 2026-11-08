"""Compact ACT-CVAE, conditional diffusion, residual actor and visual critic.

    ACT uses a pretrained ResNet18 on fresh offline runs. ``backbone: resnet18``
    upgrades diffusion/critic vision to the same ImageNet ResNet18 spatial
    tokens and switches the denoiser to a Transformer attending over them.
    ``backbone: cnn`` keeps the original compact architecture so checkpoints
    trained before the upgrade still load.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from starter.interfaces import ACTION_DIM
from starter.policies.act_model import ACT, sine_position, spatial_position


class Vision(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(4, 32), nn.SiLU(),
                                 nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
                                 nn.Conv2d(64, width, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.position = nn.Parameter(torch.randn(1, 16, width) * 0.02)

    def forward(self, image):
        return self.net(image).flatten(2).transpose(1, 2) + self.position


class SpatialVision(nn.Module):
    """ResNet18 trunk; keeps the full stride-32 spatial grid as tokens."""

    def __init__(self, width, obs_steps=1):
        super().__init__()
        from torchvision.models import resnet18
        from torchvision.ops.misc import FrozenBatchNorm2d

        backbone = resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
        self.width = width
        self.obs_steps = int(obs_steps)
        if self.obs_steps < 1:
            raise ValueError("obs_steps must be positive")
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Conv2d(512, width, 1)
        # A learned per-frame embedding distinguishes history order; the spatial
        # position encoding itself is shared by every frame.
        self.frame_embed = (nn.Parameter(torch.randn(1, self.obs_steps, 1, width) * 0.02)
                            if self.obs_steps > 1 else None)
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, image):
        if image.dim() == 5:
            batch, frames = int(image.shape[0]), int(image.shape[1])
            if self.frame_embed is None or frames != self.obs_steps:
                raise ValueError(f"vision with obs_steps={self.obs_steps} got {frames} frames")
            flat = image.reshape(batch * frames, *image.shape[2:])
            feature = self.projection(self.backbone((flat - self.image_mean) / self.image_std))
            grid = int(feature.shape[-2] * feature.shape[-1])
            tokens = feature.flatten(2).transpose(1, 2).reshape(batch, frames, grid, self.width)
            tokens = (tokens + self.frame_embed).reshape(batch, frames * grid, self.width)
            position = spatial_position(*feature.shape[-2:], self.width, image.device)
            return tokens + position.repeat(1, frames, 1)
        if image.dim() != 4:
            raise ValueError("vision image must be [B,3,H,W] or [B,obs_steps,3,H,W]")
        if self.obs_steps != 1:
            raise ValueError(f"vision with obs_steps={self.obs_steps} expects a [B,T,3,H,W] history")
        feature = self.projection(self.backbone((image - self.image_mean) / self.image_std))
        position = spatial_position(*feature.shape[-2:], self.width, image.device)
        return feature.flatten(2).transpose(1, 2) + position

    def initialize_pretrained_backbone(self):
        # Only fresh training calls this. Loading checkpoints is offline-safe.
        from torchvision.models import ResNet18_Weights, resnet18
        from torchvision.ops.misc import FrozenBatchNorm2d

        source = resnet18(weights=ResNet18_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)
        self.backbone.load_state_dict(nn.Sequential(*list(source.children())[:-2]).state_dict())

    def backbone_parameters(self):
        return self.backbone.parameters()


class TokenPool(nn.Module):
    """Learned-query cross-attention pooling over spatial tokens."""

    def __init__(self, width, heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, tokens):
        query = self.query.expand(len(tokens), -1, -1)
        return self.attention(query, tokens, tokens, need_weights=False)[0][:, 0]


class Condition(nn.Module):
    def __init__(self, width, backbone="cnn", heads=4, obs_steps=1):
        super().__init__()
        if backbone not in ("cnn", "resnet18"):
            raise ValueError(f"Unknown vision backbone: {backbone}")
        self.obs_steps = int(obs_steps)
        if backbone == "cnn" and self.obs_steps > 1:
            raise ValueError("obs_steps > 1 requires backbone: resnet18")
        self.spatial = backbone == "resnet18"
        self.vision = SpatialVision(width, self.obs_steps) if self.spatial else Vision(width)
        self.pool = TokenPool(width, heads) if self.spatial else None
        self.proprio = nn.Linear(ACTION_DIM, width)

    def forward(self, image, state):
        return torch.cat((self.vision(image), self.proprio(state).unsqueeze(1)), dim=1)

    def pooled(self, image, state):
        if self.spatial:
            return self.pool(self.vision(image)) + self.proprio(state)
        return self.forward(image, state).mean(1)

    def initialize_pretrained_backbone(self):
        if self.spatial:
            self.vision.initialize_pretrained_backbone()


class Block(nn.Module):
    """Self-attention encoder block; positions are baked into the content."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(4 * width, width), nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)

    def forward(self, content):
        update = self.attention(content, content, content, need_weights=False)[0]
        content = self.norm1(content + update)
        return self.norm2(content + self.ff(content))


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


class TransformerDenoiser(nn.Module):
    """DiT-style denoiser: time/state/action tokens attend to spatial image tokens."""

    def __init__(self, config):
        super().__init__()
        w, h = int(config["width"]), int(config["chunk"])
        heads, layers = int(config["heads"]), int(config["layers"])
        dropout = float(config.get("dropout", 0.1))
        if w % heads or w % 4:
            raise ValueError("width must be divisible by heads and by 4 for 2-D position encoding")
        self.chunk, self.width = h, w
        self.vision = SpatialVision(w, int(config.get("obs_steps", 1)))
        self.time_token = nn.Linear(w, w)
        self.state_token = nn.Linear(ACTION_DIM, w)
        self.action_embed = nn.Linear(ACTION_DIM, w)
        self.extra_position = nn.Parameter(torch.randn(1, 2, w) * .02)
        self.layers = nn.ModuleList(Block(w, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(w)
        self.out = nn.Linear(w, ACTION_DIM)

    def forward(self, x, timestep, image, state, visual=None):
        half = self.width // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / max(half - 1, 1))
        phase = timestep[:, None].float() * freqs[None, :]
        time = torch.cat((phase.sin(), phase.cos()), -1)
        prefix = torch.stack((self.time_token(time), self.state_token(state)), 1) + self.extra_position
        actions = self.action_embed(x) + sine_position(self.chunk, self.width, x.device)[None]
        # ``visual`` lets DDIM reuse the frozen image tokens across denoising
        # steps; PPO still recomputes gradients through every stored step.
        tokens = self.vision(image) if visual is None else visual
        content = torch.cat((prefix, actions, tokens), 1)
        for layer in self.layers:
            content = layer(content)
        return self.out(self.norm(content[:, 2:2 + self.chunk]))

    def initialize_pretrained_backbone(self):
        self.vision.initialize_pretrained_backbone()

    def backbone_parameters(self):
        return self.vision.backbone_parameters()


class Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone = str(config.get("backbone", "cnn"))
        if backbone not in ("cnn", "resnet18"):
            raise ValueError(f"Unknown vision backbone: {backbone}")
        if backbone == "cnn" and int(config.get("obs_steps", 1)) > 1:
            raise ValueError("obs_steps > 1 requires backbone: resnet18")
        self.denoiser = TransformerDenoiser(config) if backbone == "resnet18" else Denoiser(config)
        self.chunk, self.steps = int(config["chunk"]), int(config["diffusion_steps"])
        if self.steps < 2:
            raise ValueError("diffusion_steps must be >=2")
        # Min-SNR-gamma loss weighting (arXiv:2303.09556). 0 keeps plain MSE.
        self.min_snr_gamma = float(config.get("min_snr_gamma", 0.0))
        if self.min_snr_gamma < 0:
            raise ValueError("min_snr_gamma must be nonnegative")
        # DDIM (arXiv:2010.02502) inference over a strided subset of the training
        # schedule; 0 keeps the legacy DDPM loop for old checkpoints.
        requested = int(config.get("inference_steps", 0) or self.steps)
        self.inference_steps = min(max(requested, 1), self.steps)
        self.ddim = self.inference_steps < self.steps
        self.schedule = self._schedule()
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

    def _schedule(self):
        """Descending ``(timestep, previous_timestep)`` pairs; -1 means x0."""
        if not getattr(self, "ddim", False):
            return [(step, step - 1) for step in reversed(range(self.steps))]
        values = torch.linspace(self.steps - 1, 0, self.inference_steps).round().long().tolist()
        values = list(dict.fromkeys(values))
        return [(step, values[index + 1] if index + 1 < len(values) else -1)
                for index, step in enumerate(values)]

    def initialize_pretrained_backbone(self):
        initialize = getattr(self.denoiser, "initialize_pretrained_backbone", None)
        if initialize is not None:
            initialize()

    def backbone_parameters(self):
        parameters = getattr(self.denoiser, "backbone_parameters", None)
        return parameters() if parameters is not None else []

    def training_loss(self, image, state, action, valid):
        time = torch.randint(self.steps, (len(action),), device=action.device)
        noise = torch.randn_like(action)
        abar = self.abar[time, None, None]
        noisy = abar.sqrt() * action + (1 - abar).sqrt() * noise
        prediction = self.denoiser(noisy, time, image, state)
        loss = (prediction - noise).square().mean(-1)
        if self.min_snr_gamma > 0:
            snr = self.abar[time] / (1 - self.abar[time]).clamp_min(1e-8)
            weights = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma)) / snr
            loss = loss * weights[:, None]
        return (loss * valid).sum() / valid.sum().clamp_min(1)

    def condition_features(self, image, state):
        """Frozen visual/proprioceptive context for the online value head."""
        denoiser = self.denoiser
        if hasattr(denoiser, "condition"):
            return denoiser.condition.pooled(image, state)
        tokens = denoiser.vision(image)
        return tokens.mean(1) + denoiser.state_token(state)

    def distribution(self, x, time, image, state, min_std, prev_time=None):
        if self.ddim and prev_time is not None:
            return self.ddim_distribution(x, time, prev_time, image, state, min_std)
        eps = self.denoiser(x, time, image, state)
        a, b = self.abar[time, None, None], self.previous[time, None, None]
        beta = self.beta[time, None, None]
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-5, 5)
        mean = beta * b.sqrt() / (1 - a) * x0 + (1 - b) * self.alpha[time, None, None].sqrt() / (1 - a) * x
        std = (beta * (1 - b) / (1 - a)).sqrt().clamp_min(min_std)
        return Normal(mean, std.expand_as(mean))

    def ddim_distribution(self, x, time, prev_time, image, state, min_std, visual=None):
        """eta=0 DDIM transition to ``prev_time`` with a minimum exploration std.

        The mean is the deterministic DDIM step, so deployment samples converge
        to x0; the PPO suffix still gets a proper Gaussian log-density.
        """
        eps = (self.denoiser(x, time, image, state, visual=visual)
               if visual is not None else self.denoiser(x, time, image, state))
        a = self.abar[time, None, None]
        safe = prev_time.clamp_min(0)
        previous = self.abar[safe, None, None]
        previous = torch.where(prev_time[:, None, None] < 0, torch.ones_like(previous), previous)
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-5, 5)
        mean = previous.sqrt() * x0 + (1 - previous).sqrt() * eps
        return Normal(mean, torch.full_like(mean, max(float(min_std), 1e-6)))

    @torch.no_grad()
    def sample(self, image, state, stochastic_steps=0, min_std=0.05, prefix_model=None, finetune_steps=None,
               prefix_steps=None):
        if prefix_steps is not None and not 1 <= int(prefix_steps) <= self.chunk:
            raise ValueError("prefix_steps must be in [1, chunk]")
        x = torch.randn(len(state), self.chunk, ACTION_DIM, device=state.device)
        trace = []
        count = len(self.schedule)
        threshold = stochastic_steps if finetune_steps is None else finetune_steps
        threshold = max(0, min(int(threshold), count))
        visuals = {}
        for index, (step, previous) in enumerate(self.schedule):
            time = torch.full((len(state),), int(step), dtype=torch.long, device=state.device)
            predictor = prefix_model if prefix_model is not None and index < count - threshold else self
            prev_time = torch.full((len(state),), int(previous), dtype=torch.long, device=state.device)
            if self.ddim:
                # DDIM conditions on the same observation at every step; encode
                # the image tokens once per participating model.
                vision = getattr(predictor.denoiser, "vision", None)
                visual = None
                if vision is not None:
                    visual = visuals.get(id(predictor))
                    if visual is None:
                        visual = vision(image)
                        visuals[id(predictor)] = visual
                dist = predictor.ddim_distribution(x, time, prev_time, image, state, min_std, visual=visual)
            else:
                dist = predictor.distribution(x, time, image, state, min_std)
            if index >= count - stochastic_steps:
                nxt = dist.sample()
                logprob = dist.log_prob(nxt)
                if prefix_steps is not None:
                    # PPO only owns the executed action prefix; the unexecuted
                    # suffix never reaches the simulator.  Keep one value per
                    # prefix step so the update can mask early episode ends.
                    logprob = logprob[..., :int(prefix_steps), :].sum(-1)
                else:
                    logprob = logprob.flatten(1).sum(-1)
                trace.append((x.clone(), nxt.clone(), time, logprob, prev_time))
            else:
                nxt = dist.mean
            x = nxt
        return x, trace


class ResidualActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        w, h = int(config["width"]), int(config["chunk"])
        self.condition = Condition(w, str(config.get("backbone", "cnn")), int(config.get("heads", 4)),
                                   int(config.get("obs_steps", 1)))
        self.mean = nn.Sequential(nn.Linear(w + h * ACTION_DIM, w), nn.Tanh(), nn.Linear(w, h * ACTION_DIM))
        nn.init.zeros_(self.mean[-1].weight)
        nn.init.zeros_(self.mean[-1].bias)
        self.log_std = nn.Parameter(torch.full((h, ACTION_DIM), math.log(float(config.get("residual_std", 0.3)))))
        self.chunk = h

    def distribution(self, image, state, base):
        features = torch.cat((self.condition.pooled(image, state), base.flatten(1)), -1)
        mean = self.mean(features).reshape(-1, self.chunk, ACTION_DIM)
        return Normal(mean, self.log_std.clamp(-5, 0).exp().expand_as(mean))

    def initialize_pretrained_backbone(self):
        initialize = getattr(self.condition, "initialize_pretrained_backbone", None)
        if initialize is not None:
            initialize()


class Critic(nn.Module):
    """Legacy independent visual critic (kept for old online checkpoints)."""

    def __init__(self, config):
        super().__init__()
        width = int(config["width"])
        self.condition = Condition(width, str(config.get("backbone", "cnn")), int(config.get("heads", 4)),
                                   int(config.get("obs_steps", 1)))
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, image, state):
        return self.head(self.condition.pooled(image, state)).squeeze(-1)

    def initialize_pretrained_backbone(self):
        initialize = getattr(self.condition, "initialize_pretrained_backbone", None)
        if initialize is not None:
            initialize()


class FeatureCritic(nn.Module):
    """Value head on top of a frozen policy encoder.

    Sparse 0/1 rewards make a from-scratch visual critic the main sample
    bottleneck.  ACT/Diffusion already learned task features offline; this
    head only maps those frozen features to a value.
    """

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, features):
        return self.head(self.norm(features)).squeeze(-1)
