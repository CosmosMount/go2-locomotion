"""Safety critic portion of source-domain Recovery RL pretraining."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .core import safety_bellman_target


def update_safety(agent, replay) -> dict[str, float]:
    """Update only the sigmoid twin Safety-Q and its target network."""

    config = agent.config
    batch = replay.sample(
        config.batch_size,
        agent.device,
        positive_fraction=config.safety_positive_fraction,
    )
    observation = agent.safety_view(batch["obs"])
    with torch.no_grad():
        next_task_observation = agent.task_view(batch["next_obs"])
        next_task_action, _ = agent.task_actor.sample(next_task_observation)
        next_safety_observation = agent.safety_view(batch["next_obs"])
        next_risk = torch.maximum(
            *agent.safety_target(next_safety_observation, next_task_action)
        )
        target = safety_bellman_target(
            batch["cost"], batch["terminated"], next_risk, config.gamma_safe
        )

    q1, q2 = agent.safety_q(observation, batch["executed_action"])
    loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    value = agent._step("safety_q", loss)
    agent._soft_update(agent.safety_q, agent.safety_target, config.tau_safe)
    agent.safety_updates += 1
    return {"safety_q_loss": value}
