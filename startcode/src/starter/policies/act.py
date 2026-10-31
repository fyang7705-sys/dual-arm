"""ACT CVAE cloning, followed by PPO residuals on a frozen ACT base."""


def train_offline(config):
    from starter.policies.offline import train
    return train("act", config)


def train_online(config):
    from starter.policies.online import train
    return train("act", config)


def load_policy(checkpoint_path, device="cpu"):
    from starter.policies.runtime import Policy
    return Policy(checkpoint_path, device)
