"""PPO rollout, update, and checkpoint loop for locomotion pretraining."""

from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

from rl import ActionSpaceType, ObservationSpaceType
from rl.ppo.pytorch.batch import Batch
from rl.ppo.pytorch.critic import get_critic
from rl.ppo.pytorch.policy import get_policy
from rl.ppo.pytorch.ppo import PPO, compute_gae
from common.output import prepare_output_dir

from .config import LocomotionConfig


class LocomotionTrainer(PPO):
    """Own the reusable PPO models while the environment owns task semantics."""

    def __init__(self, config: LocomotionConfig):
        config.validate(check_assets=True)
        self.config = config
        self.device = config.device
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.env = None
        self.policy = None
        self.critic = None
        self.optimizer = None

    def run(self) -> None:
        from .environment import Go2LocomotionEnv

        self.env = Go2LocomotionEnv(self.config)
        try:
            observations = self.env.reset()[0]
            observation = observations["policy"]
            self.critic_observation = observations["critic"]
            self._validate_spaces(observation)
            self._build_models()
            output_dir = prepare_output_dir(
                self.config.output_dir, temp_prefix="go2-pretrain-locomotion-"
            )
            log_dir = (
                output_dir
                / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ).resolve()
            log_dir.mkdir(parents=True)
            start_time = time.monotonic()
            for iteration in range(1, self.config.max_iterations + 1):
                observation, batch, mean_reward = self._collect_rollout(observation)
                policy_loss, value_loss, entropy = self._ppo_update(batch)
                elapsed = time.monotonic() - start_time
                fps = int(
                    iteration * self.config.horizon * self.config.num_envs / elapsed
                )
                print(
                    f"iteration={iteration:04d}/{self.config.max_iterations} "
                    f"fps={fps} reward={mean_reward:+.4f} "
                    f"policy={policy_loss:+.4f} value={value_loss:.4f} "
                    f"entropy={entropy:.4f} "
                    f"std_max={self.policy.policy_logstd.detach().exp().max().item():.3f} "
                    f"saturation={self.action_saturation:.1%} "
                    f"kl={self.update_kl:.4f} kl_stop={self.kl_stopped} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e} "
                    f"metrics={self.rollout_metrics}",
                    flush=True,
                )
                if (
                    iteration % self.config.checkpoint_interval == 0
                    or iteration == self.config.max_iterations
                ):
                    self._save_checkpoint(log_dir, iteration)
        finally:
            if self.env is not None:
                self.env.close()

    def _validate_spaces(self, observation) -> None:
        expected = (self.config.num_envs, self.config.observation_size)
        if observation.shape != expected:
            raise RuntimeError(f"invalid policy observation: expected {expected}, got {observation.shape}")
        if self.critic_observation.shape != (
            self.config.num_envs,
            self.config.critic_observation_size,
        ):
            raise RuntimeError(f"invalid critic observation: {self.critic_observation.shape}")
        if self.env.single_action_space.shape != (self.config.action_size,):
            raise RuntimeError(f"invalid action space: {self.env.single_action_space.shape}")

    def _build_models(self) -> None:
        model_config = SimpleNamespace(algorithm=self.config)
        model_env = SimpleNamespace(
            general_properties=SimpleNamespace(
                action_space_type=ActionSpaceType.CONTINUOUS,
                observation_space_type=ObservationSpaceType.FLAT_VALUES,
            ),
            single_observation_space=self.env.single_observation_space,
            single_action_space=self.env.single_action_space,
        )
        policy = get_policy(model_config, model_env, self.device)
        model_env.single_observation_space = self.env.single_critic_observation_space
        critic = get_critic(model_config, model_env, self.device)
        optimizer = torch.optim.Adam(
            tuple(policy.parameters()) + tuple(critic.parameters()),
            lr=self.config.learning_rate,
        )
        super().__init__(self.config, policy, critic, optimizer)

    def _collect_rollout(self, observation):
        states, next_states, actions, env_actions = [], [], [], []
        rewards, values, terminations, truncations, log_probs = [], [], [], [], []
        means, stds, critic_states = [], [], []
        metrics = {}
        for _ in range(self.config.horizon):
            with torch.no_grad():
                action, env_action, log_prob = self.policy.get_action_logprob(observation)
                value = self.critic.get_value(self.critic_observation).squeeze(-1)
                distribution = self.policy.distribution(observation)
                critic_states.append(self.critic_observation)
                means.append(distribution.mean.clone())
                stds.append(distribution.stddev.clone())
            next_observations, reward, terminated, truncated, info = self.env.step(env_action)
            for name, metric in info["metrics"].items():
                metrics[name] = metrics.get(name, 0) + metric.detach()
            states.append(observation)
            next_states.append(info["final_critic_observation"])
            actions.append(action)
            env_actions.append(env_action)
            rewards.append(reward)
            values.append(value)
            terminations.append(terminated.clone())
            truncations.append(truncated.clone())
            log_probs.append(log_prob)
            observation = next_observations["policy"]
            self.critic_observation = next_observations["critic"]

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        values = torch.stack(values)
        terminations = torch.stack(terminations)
        truncations = torch.stack(truncations)
        with torch.no_grad():
            next_values = self.critic.get_value(next_states).squeeze(-1)
            advantages, returns = compute_gae(
                rewards,
                values,
                next_values,
                terminations,
                truncations,
                self.config.gamma,
                self.config.gae_lambda,
            )
        flat_advantages = advantages.flatten()
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std(unbiased=False) + 1.0e-8
        )
        batch = Batch(
            states.flatten(0, 1),
            next_states.flatten(0, 1),
            torch.stack(actions).flatten(0, 1),
            rewards.flatten(),
            values.flatten(),
            terminations.flatten(),
            torch.stack(log_probs).flatten(),
            flat_advantages,
            returns.flatten(),
            truncations.flatten(),
            torch.stack(env_actions).flatten(0, 1),
            torch.stack(means).flatten(0, 1),
            torch.stack(stds).flatten(0, 1),
            torch.stack(critic_states).flatten(0, 1),
        )
        self.rollout_metrics = {
            name: round((value / self.config.horizon).item(), 5)
            for name, value in metrics.items()
        }
        self.action_saturation = (batch.actions.abs() >= 1.0).float().mean().item()
        return observation, batch, rewards.mean().item()

    def _save_checkpoint(self, log_dir, iteration: int) -> None:
        from common.checkpoint import checkpoint_metadata, save_checkpoint

        metadata = checkpoint_metadata(
            task=self.config.name,
            policy_view=self.config.policy_view,
            policy_observation_size=self.config.observation_size,
            default_joint_position=self.config.default_joint_pos,
            action_scale=(self.config.action_scale,) * self.config.action_size,
            observation_scales={
                "angular_velocity": self.config.angular_velocity_scale,
                "joint_velocity": self.config.joint_velocity_scale,
                "command": list(self.config.command_scale),
                "critic_torque": self.config.torque_scale,
            },
        )
        save_checkpoint(
            log_dir / f"checkpoint_{iteration:05d}.pt",
            metadata=metadata,
            state={
                "iteration": iteration,
                "config": self.config.to_mapping(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
        )
