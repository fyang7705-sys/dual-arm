"""Installed into XPolicyLab adapters; this file has no policy/GPU dependencies."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        if not actions:
            raise RuntimeError("Policy returned an empty action chunk")
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        active = TASK_ENV.get_running_env_idx_list()
        observations = TASK_ENV.get_obs_batch(active)
        for observation, index in zip(observations, active):
            observation["env_idx"] = index
        model_client.call(func_name="update_obs_batch", obs=observations)
        chunks = model_client.call(func_name="get_action_batch", obs=active)
        if len(chunks) != len(active) or not chunks or not chunks[0]:
            raise RuntimeError("Policy returned malformed batch chunks")
        for step in range(len(chunks[0])):
            TASK_ENV.take_action_batch([chunk[step] for chunk in chunks], active)
            if TASK_ENV.is_episode_end():
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            kept = [(index, chunk) for index, chunk in zip(active, chunks) if index in running]
            active = [item[0] for item in kept]
            chunks = [item[1] for item in kept]
            if not active:
                break
