"""MF recovery actor portion of source-domain Recovery RL pretraining."""

from __future__ import annotations

import torch


def freeze_dependencies(agent) -> None:
    """Freeze task and safety learners, leaving only recovery trainable."""

    for module in (
        agent.task_actor,
        agent.reward_q,
        agent.reward_target,
        agent.safety_q,
        agent.safety_target,
    ):
        module.eval().requires_grad_(False)
    agent.entropy_coefficient.eval().requires_grad_(False)
    agent.task_normalizer.frozen = True
    agent.safety_normalizer.frozen = True
    agent.recovery_actor.train().requires_grad_(True)


def update_recovery(agent, replay) -> dict[str, float]:
    """Optimize only recovery actions against the frozen worst-case risk."""

    batch = replay.sample(
        agent.config.batch_size,
        agent.device,
        positive_fraction=agent.config.safety_positive_fraction,
    )
    observation = agent.safety_view(batch["obs"])
    action, _ = agent.recovery_actor.sample(observation)
    loss = torch.maximum(*agent.safety_q(observation, action)).mean()
    value = agent._step("recovery_actor", loss)
    agent.recovery_updates += 1
    return {"recovery_loss": value}
