"""Minimal on-policy runner around the repository's reusable PPO core."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Normal

from common.checkpoint import checkpoint_metadata, load_checkpoint, save_checkpoint
from common.observation import PPO_OBSERVATION_ABI
from common.output import prepare_output_dir
from rl.ppo.pytorch.batch import Batch
from rl.ppo.pytorch.network import build_mlp
from rl.ppo.pytorch.ppo import PPO, compute_gae


class GaussianActor(nn.Module):
    """Diagonal Gaussian actor compatible with ``rl.ppo.pytorch.PPO``."""

    def __init__(self, observation_dim: int, action_dim: int, cfg: dict):
        super().__init__()
        self.policy_mean = build_mlp(
            observation_dim, action_dim, cfg["actor_hidden_dims"], cfg["activation"], output_gain=0.01
        )
        self.policy_logstd = nn.Parameter(torch.full((1, action_dim), float(cfg["initial_std"])).log())

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.policy_mean(observation)
        return Normal(mean, self.policy_logstd.exp().expand_as(mean))

    def get_action_logprob(self, observation: torch.Tensor, clip_actions: float):
        distribution = self.distribution(observation)
        raw_action = distribution.sample()
        environment_action = raw_action.clamp(-clip_actions, clip_actions)
        log_probability = distribution.log_prob(raw_action).sum(dim=-1)
        return raw_action, environment_action, log_probability

    def get_logprob_entropy(self, observation: torch.Tensor, raw_action: torch.Tensor):
        distribution = self.distribution(observation)
        return distribution.log_prob(raw_action).sum(dim=-1), distribution.entropy().sum(dim=-1)


class ValueCritic(nn.Module):
    def __init__(self, observation_dim: int, cfg: dict):
        super().__init__()
        self.critic = build_mlp(
            observation_dim, 1, cfg["critic_hidden_dims"], cfg["activation"], output_gain=1.0
        )

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation)


def _ppo_config(cfg: dict) -> SimpleNamespace:
    return SimpleNamespace(
        min_std_dev=float(cfg["min_std"]),
        max_std_dev=float(cfg["max_std"]),
        adaptive_learning_rate=bool(cfg["adaptive_learning_rate"]),
        target_kl=float(cfg["target_kl"]),
        kl_stop_multiplier=float(cfg["kl_stop_multiplier"]),
        min_learning_rate=float(cfg["min_learning_rate"]),
        max_learning_rate=float(cfg["max_learning_rate"]),
        update_epochs=int(cfg["update_epochs"]),
        num_minibatches=int(cfg["num_minibatches"]),
        clip_ratio=float(cfg["clip_ratio"]),
        value_clip=float(cfg["value_clip"]),
        value_loss_coefficient=float(cfg["value_loss_coefficient"]),
        entropy_coefficient=float(cfg["entropy_coefficient"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
    )


def _flatten(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, *value.shape[2:]) if value.ndim > 2 else value.reshape(-1)


def _metadata(cfg: dict) -> dict:
    observation = cfg["observation"]
    return checkpoint_metadata(
        task="pretrain-recovery",
        policy_view=PPO_OBSERVATION_ABI,
        policy_observation_size=cfg["environment"]["actor_observation_dim"],
        default_joint_position=cfg["robot"]["default_joint_position"],
        action_scale=cfg["robot"]["action_scale"],
        observation_scales={
            "angular_velocity": observation["angular_velocity_scale"],
            "joint_velocity": observation["joint_velocity_scale"],
            "command": observation["command_scale"],
            "height": observation["height_scale"],
        },
    )


def _save(path: Path, *, actor, critic, optimizer, iteration: int, cfg: dict) -> None:
    save_checkpoint(
        path,
        metadata=_metadata(cfg),
        state={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": int(iteration),
        },
    )


def train(env, cfg: dict) -> None:
    """Collect vectorized rollouts and update a 45D/260D asymmetric PPO."""

    device = torch.device(cfg["device"])
    env_cfg, ppo_cfg = cfg["environment"], cfg["ppo"]
    actor = GaussianActor(env_cfg["actor_observation_dim"], env_cfg["action_dim"], ppo_cfg).to(device)
    critic = ValueCritic(env_cfg["critic_observation_dim"], ppo_cfg).to(device)
    optimizer = torch.optim.Adam(
        tuple(actor.parameters()) + tuple(critic.parameters()), lr=float(ppo_cfg["learning_rate"])
    )
    updater = PPO(_ppo_config(ppo_cfg), actor, critic, optimizer)

    start_iteration = 0
    if cfg.get("checkpoint"):
        payload = load_checkpoint(cfg["checkpoint"], expected_task="pretrain-recovery", map_location=device)
        actor.load_state_dict(payload["state"]["actor"])
        critic.load_state_dict(payload["state"]["critic"])
        optimizer.load_state_dict(payload["state"]["optimizer"])
        start_iteration = int(payload["state"]["iteration"])
        updater._bound_policy_std()

    output_dir = prepare_output_dir(
        cfg["output_dir"], temp_prefix="go2-pretrain-recovery-"
    )
    observations, _ = env.reset(seed=int(cfg["seed"]))
    policy_observation = observations["policy"]
    critic_observation = observations["critic"]
    steps = int(ppo_cfg["steps_per_env"])
    clip_actions = float(env_cfg["clip_actions"])

    for iteration in range(start_iteration, int(ppo_cfg["max_iterations"])):
        env.training_iteration = iteration
        rollout = {name: [] for name in (
            "policy", "critic", "next_policy", "action", "reward", "value", "terminated", "truncated",
            "log_probability", "old_mean", "old_std", "next_value",
        )}
        for _ in range(steps):
            with torch.no_grad():
                distribution = actor.distribution(policy_observation)
                raw_action = distribution.sample()
                environment_action = raw_action.clamp(-clip_actions, clip_actions)
                log_probability = distribution.log_prob(raw_action).sum(dim=-1)
                value = critic.get_value(critic_observation).squeeze(-1)

            next_observations, reward, terminated, truncated, _ = env.step(environment_action)
            next_policy = next_observations["policy"]
            next_critic = next_observations["critic"]
            bootstrap_critic = torch.where(
                truncated.unsqueeze(-1), env.transition_critic_observation, next_critic
            )
            with torch.no_grad():
                next_value = critic.get_value(bootstrap_critic).squeeze(-1)

            for name, value_to_store in (
                ("policy", policy_observation), ("critic", critic_observation),
                ("next_policy", next_policy), ("action", raw_action), ("reward", reward),
                ("value", value), ("terminated", terminated), ("truncated", truncated),
                ("log_probability", log_probability), ("old_mean", distribution.mean),
                ("old_std", distribution.stddev), ("next_value", next_value),
            ):
                rollout[name].append(value_to_store.detach())
            policy_observation, critic_observation = next_policy, next_critic

        stacked = {name: torch.stack(values) for name, values in rollout.items()}
        advantages, returns = compute_gae(
            stacked["reward"], stacked["value"], stacked["next_value"],
            stacked["terminated"], stacked["truncated"],
            float(ppo_cfg["gamma"]), float(ppo_cfg["gae_lambda"]),
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
        batch = Batch(
            states=_flatten(stacked["policy"]),
            next_states=_flatten(stacked["next_policy"]),
            actions=_flatten(stacked["action"]),
            rewards=_flatten(stacked["reward"]),
            values=_flatten(stacked["value"]),
            terminations=_flatten(stacked["terminated"]),
            truncations=_flatten(stacked["truncated"]),
            log_probs=_flatten(stacked["log_probability"]),
            advantages=_flatten(advantages),
            returns=_flatten(returns),
            env_actions=None,
            old_mean=_flatten(stacked["old_mean"]),
            old_std=_flatten(stacked["old_std"]),
            critic_states=_flatten(stacked["critic"]),
        )
        policy_loss, value_loss, entropy = updater._ppo_update(batch)
        completed = iteration + 1
        print(
            f"iteration={completed} reward={stacked['reward'].mean().item():.5f} "
            f"policy_loss={policy_loss:.5f} value_loss={value_loss:.5f} entropy={entropy:.5f} "
            f"kl={updater.update_kl:.6f}"
        )
        if completed % int(ppo_cfg["save_interval"]) == 0:
            _save(output_dir / "latest.pt", actor=actor, critic=critic, optimizer=optimizer,
                  iteration=completed, cfg=cfg)

    _save(output_dir / "final.pt", actor=actor, critic=critic, optimizer=optimizer,
          iteration=int(ppo_cfg["max_iterations"]), cfg=cfg)
