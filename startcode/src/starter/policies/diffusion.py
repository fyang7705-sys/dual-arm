"""Visual diffusion cloning followed by denoising-chain PPO."""


def train_offline(config):
    from starter.policies.offline import train
    return train("diffusion", config)


def train_online(config):
    from starter.policies.online import train
    return train("diffusion", config)


def load_policy(checkpoint_path, device="cpu"):
    from starter.policies.runtime import Policy
    return Policy(checkpoint_path, device)
