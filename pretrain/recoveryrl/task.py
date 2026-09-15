"""Task SAC portion of source-domain Recovery RL pretraining."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def update_task(agent, replay) -> dict[str, float]:
    """Update only the task actor, reward critics, targets, and entropy."""

    config = agent.config
    batch = replay.sample(config.batch_size, agent.device)
    observation = agent.task_view(batch["obs"])
    with torch.no_grad():
        next_observation = agent.task_view(batch["next_obs"])
        next_action, next_log_probability = agent.task_actor.sample(next_observation)
        target = batch["reward"] + config.gamma * (1 - batch["terminated"]) * (
            torch.minimum(*agent.reward_target(next_observation, next_action))
            - agent.alpha * next_log_probability
        )

    q1, q2 = agent.reward_q(observation, batch["proposed_action"])
    metrics = {
        "reward_q_loss": agent._step(
            "reward_q", F.mse_loss(q1, target) + F.mse_loss(q2, target)
        )
    }
    agent.reward_q.requires_grad_(False)
    try:
        action, log_probability = agent.task_actor.sample(observation)
        metrics["task_actor_loss"] = agent._step(
            "task_actor",
            (
                agent.alpha * log_probability
                - torch.minimum(*agent.reward_q(observation, action))
            ).mean(),
        )
    finally:
        agent.reward_q.requires_grad_(True)

    if config.automatic_entropy_tuning:
        metrics.update(agent._update_entropy(log_probability))
    else:
        metrics.update(
            alpha=float(agent.alpha), entropy=float((-log_probability).mean())
        )
    agent._soft_update(agent.reward_q, agent.reward_target, config.tau)
    agent.task_updates += 1
    return metrics
