"""Minimal SAC task learner with frozen Recovery RL safety components."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common.checkpoint import checkpoint_metadata, load_checkpoint, save_checkpoint
from common.observation import (
    RAW_OBSERVATION_SIZE,
    prepare_task_observation,
)


POLICY_VIEW = "yaw_invariant_46d"


@dataclass
class AgentConfig:
    obs_dim: int = 46
    action_dim: int = 12
    hidden: int = 256
    lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    gamma_safe: float = 0.9607894391523232
    epsilon: float = 0.2
    tau: float = 0.005
    tau_safe: float = 0.0002
    safety_continuation: str = "task"
    safety_pos_fraction: float = 0.25
    alpha: float = 1.0
    automatic_entropy_tuning: bool = True
    target_entropy: float | None = None
    seed: int = 3701
    task_observation_profile: str = "yaw_invariant"

    @classmethod
    def from_checkpoint(cls, values: dict, *, seed: int, batch_size: int,
                        epsilon: float) -> "AgentConfig":
        known = {field.name for field in fields(cls)}
        config = {key: value for key, value in values.items() if key in known}
        config.update(seed=int(seed), batch_size=int(batch_size), epsilon=float(epsilon))
        result = cls(**config)
        if (result.obs_dim, result.action_dim) != (46, 12):
            raise ValueError("source-safe checkpoint must use a 46D observation and 12D action")
        if result.task_observation_profile != "yaw_invariant":
            raise ValueError("source-safe checkpoint must use yaw_invariant task observations")
        return result


class RunningNormalizer(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self.frozen = True

    def forward(self, value):
        return ((value - self.mean) / torch.sqrt(self.var + 1e-6)).clamp(-10, 10)


def _mlp(input_size: int, output_size: int, hidden: int):
    return nn.Sequential(
        nn.Linear(input_size, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, output_size),
    )


class Actor(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.net = _mlp(config.obs_dim, config.action_dim * 2, config.hidden)

    def sample(self, observation, deterministic=False):
        mean, log_std = self.net(observation).chunk(2, -1)
        distribution = torch.distributions.Normal(mean, log_std.clamp(-20, 2).exp())
        latent = mean if deterministic else distribution.rsample()
        action = latent.tanh()
        log_probability = (
            distribution.log_prob(latent) - torch.log(1 - action.square() + 1e-6)
        ).sum(-1, keepdim=True)
        return action, log_probability


class TwinQ(nn.Module):
    def __init__(self, config: AgentConfig, *, safety=False):
        super().__init__()
        size = config.obs_dim + config.action_dim
        self.q1 = _mlp(size, 1, config.hidden)
        self.q2 = _mlp(size, 1, config.hidden)
        self.safety = bool(safety)

    def forward(self, observation, action):
        value = torch.cat((observation, action), -1)
        q1, q2 = self.q1(value), self.q2(value)
        if self.safety:
            return q1.sigmoid(), q2.sigmoid()
        return q1, q2


class EntropyCoefficient(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor([np.log(alpha)], dtype=torch.float32))


class ReplayBuffer:
    """Task-only replay; recovery intervention transitions are never added."""

    FIELDS = (
        ("obs", 46), ("next_obs", 46), ("proposed_action", 12),
        ("executed_action", 12), ("reward", 1), ("cost", 1),
        ("terminated", 1), ("truncated", 1),
    )

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.position = self.size = 0
        self.data = {name: np.zeros((self.capacity, width), np.float32)
                     for name, width in self.FIELDS}

    def __len__(self):
        return self.size

    def add(self, observation, proposed_action, executed_action, reward, cost,
            next_observation, terminated, truncated):
        values = {
            "obs": observation, "next_obs": next_observation,
            "proposed_action": proposed_action, "executed_action": executed_action,
            "reward": reward, "cost": cost, "terminated": terminated,
            "truncated": truncated,
        }
        for name, value in values.items():
            self.data[name][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        if self.size < 1:
            raise ValueError("cannot sample an empty replay")
        indices = np.random.randint(self.size, size=int(batch_size))
        return {name: torch.as_tensor(value[indices], device=device)
                for name, value in self.data.items()}

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "position": self.position,
            "size": self.size,
            "data": {name: value[:self.size].copy()
                     for name, value in self.data.items()},
        }


def validate_source_state(bundle: dict, metadata: dict, *, policy_view: str,
                          default_joint_position, action_scale):
    required_top = {"config", "modules"}
    missing = sorted(required_top - bundle.keys())
    if missing:
        raise ValueError(f"source-safe checkpoint is missing fields: {missing}")
    if metadata.get("policy_view") != policy_view:
        raise ValueError("source-safe checkpoint must use the yaw-invariant 46D task view")
    if metadata.get("policy_observation_size") != RAW_OBSERVATION_SIZE:
        raise ValueError("source-safe checkpoint policy observation must contain 46 values")
    if metadata.get("checkpoint_purpose") != "source_only_safety_pretrain_for_online_adaptation":
        raise ValueError("checkpoint is not a source-only safety pretrain handoff")
    if metadata.get("safety_policy_view") != "raw":
        raise ValueError("source-safe checkpoint must use the raw46 safety view")
    if metadata.get("policy_action_dim") != 12:
        raise ValueError("source-safe checkpoint action dimension must be 12")
    training = bundle.get("training") or {}
    if not all(training.get(name) is True for name in (
        "formal", "normalizers_frozen", "safety_stack_frozen", "transfer_ready"
    )):
        raise ValueError("source-safe checkpoint is not a formal transfer-ready checkpoint")
    for name, expected in (
        ("default_joint_position", default_joint_position),
        ("action_scale", action_scale),
    ):
        actual = np.asarray(metadata.get(name), dtype=float)
        if actual.shape != (12,) or not np.allclose(
            actual, np.asarray(expected, dtype=float), rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"source-safe checkpoint has incompatible {name}")
    required_modules = {
        "task_actor", "recovery_actor", "safety_q", "safety_target",
        "task_normalizer", "safety_normalizer", "entropy_coefficient",
    }
    missing_modules = sorted(required_modules - bundle["modules"].keys())
    if missing_modules:
        raise ValueError(f"source-safe checkpoint is missing modules: {missing_modules}")
    return metadata


class RecoveryAgent:
    """Online task actor/critic learner with immutable source safety modules."""

    def __init__(self, config: AgentConfig, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.task_actor = Actor(config).to(self.device)
        self.recovery_actor = Actor(config).to(self.device)
        self.reward_q = TwinQ(config).to(self.device)
        self.safety_q = TwinQ(config, safety=True).to(self.device)
        self.reward_target = deepcopy(self.reward_q).to(self.device)
        self.safety_target = deepcopy(self.safety_q).to(self.device)
        self.task_normalizer = RunningNormalizer(config.obs_dim).to(self.device)
        self.safety_normalizer = RunningNormalizer(config.obs_dim).to(self.device)
        self.entropy_coefficient = EntropyCoefficient(config.alpha).to(self.device)
        self.modules = {
            name: getattr(self, name) for name in (
                "task_actor", "recovery_actor", "reward_q", "safety_q",
                "reward_target", "safety_target", "task_normalizer",
                "safety_normalizer", "entropy_coefficient",
            )
        }
        self.optimizers = {
            "task_actor": torch.optim.Adam(self.task_actor.parameters(), lr=config.lr),
            "reward_q": torch.optim.Adam(self.reward_q.parameters(), lr=config.lr),
        }
        self.reward_target.requires_grad_(False)
        self.steps = 0
        self.warmup_updates = 0

    @classmethod
    def from_source_checkpoint(cls, path, *, source_task, policy_view,
                               default_joint_position, action_scale, seed,
                               batch_size, epsilon, device="cpu"):
        payload = load_checkpoint(path, expected_task=source_task, map_location="cpu")
        bundle = payload["state"]
        metadata = validate_source_state(
            bundle, payload["metadata"], policy_view=policy_view,
            default_joint_position=default_joint_position,
            action_scale=action_scale,
        )
        config = AgentConfig.from_checkpoint(bundle["config"], seed=seed,
                                             batch_size=batch_size,
                                             epsilon=epsilon)
        agent = cls(config, device)
        # The target-domain reward critic and its Adam state intentionally start
        # fresh and receive the explicit 1000-update warmup. Only the source
        # actor and immutable safety/recovery stack transfer across domains.
        for name in (
            "task_actor", "recovery_actor", "safety_q", "safety_target",
            "task_normalizer", "safety_normalizer",
        ):
            agent.modules[name].load_state_dict(bundle["modules"][name], strict=True)
        if "entropy_coefficient" in bundle["modules"]:
            agent.entropy_coefficient.load_state_dict(
                bundle["modules"]["entropy_coefficient"], strict=True
            )
        agent.freeze_source_modules()
        return agent, metadata

    @property
    def alpha(self):
        return self.entropy_coefficient.log_alpha.exp().detach()

    def freeze_source_modules(self):
        for module in (
            self.safety_q, self.safety_target, self.recovery_actor,
            self.task_normalizer, self.safety_normalizer,
            self.entropy_coefficient,
        ):
            module.eval().requires_grad_(False)
        self.task_normalizer.frozen = self.safety_normalizer.frozen = True

    def task_observation(self, raw_observation):
        value = torch.as_tensor(raw_observation, dtype=torch.float32, device=self.device)
        return self.task_normalizer(prepare_task_observation(value, "yaw_invariant"))

    def safety_observation(self, raw_observation):
        value = torch.as_tensor(raw_observation, dtype=torch.float32, device=self.device)
        return self.safety_normalizer(value)

    @torch.no_grad()
    def act(self, observation, *, deterministic=False, use_recovery=True):
        value = np.asarray(observation, np.float32).reshape(1, RAW_OBSERVATION_SIZE)
        proposed, _ = self.task_actor.sample(self.task_observation(value), deterministic)
        safety_observation = self.safety_observation(value)
        risk = torch.maximum(*self.safety_q(safety_observation, proposed)).squeeze(-1)
        recovered = bool(use_recovery and risk.item() > self.config.epsilon)
        if recovered:
            executed = self.recovery_actor.sample(safety_observation, deterministic)[0]
        else:
            executed = proposed
        return {
            "proposed_action": proposed[0].cpu().numpy(),
            "executed_action": executed[0].cpu().numpy(),
            "risk": float(risk.item()),
            "recovered": recovered,
        }

    def _optimizer_step(self, name: str, loss):
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"{name} loss is non-finite")
        loss.backward()
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f"{name} gradient is non-finite")
        optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def _update_reward_target(self):
        for source, target in zip(self.reward_q.parameters(), self.reward_target.parameters()):
            target.lerp_(source, self.config.tau)

    def _reward_target_value(self, batch):
        with torch.no_grad():
            next_observation = self.task_observation(batch["next_obs"])
            next_action, log_probability = self.task_actor.sample(next_observation)
            next_q = torch.minimum(*self.reward_target(next_observation, next_action))
            # Truncations bootstrap; only a physical task failure is terminal.
            return batch["reward"] + self.config.gamma * (1 - batch["terminated"]) * (
                next_q - self.alpha * log_probability
            )

    def update_reward_critic(self, replay: ReplayBuffer):
        batch = replay.sample(self.config.batch_size, self.device)
        observation = self.task_observation(batch["obs"])
        q1, q2 = self.reward_q(observation, batch["proposed_action"])
        target = self._reward_target_value(batch)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        value = self._optimizer_step("reward_q", loss)
        self._update_reward_target()
        self.warmup_updates += 1
        return {"reward_q_loss": value, "alpha": float(self.alpha)}

    def update_task(self, replay: ReplayBuffer):
        batch = replay.sample(self.config.batch_size, self.device)
        observation = self.task_observation(batch["obs"])
        q1, q2 = self.reward_q(observation, batch["proposed_action"])
        target = self._reward_target_value(batch)
        critic_loss = self._optimizer_step(
            "reward_q", F.mse_loss(q1, target) + F.mse_loss(q2, target)
        )
        self.reward_q.requires_grad_(False)
        try:
            action, log_probability = self.task_actor.sample(observation)
            actor_loss = self._optimizer_step(
                "task_actor",
                (self.alpha * log_probability
                 - torch.minimum(*self.reward_q(observation, action))).mean(),
            )
        finally:
            self.reward_q.requires_grad_(True)
        self._update_reward_target()
        self.steps += 1
        return {
            "reward_q_loss": critic_loss,
            "task_actor_loss": actor_loss,
            "alpha": float(self.alpha),
            "entropy": float(-log_probability.detach().mean()),
        }

    def frozen_source_digest(self):
        digest = hashlib.sha256()
        for name in (
            "safety_q", "safety_target", "recovery_actor",
            "task_normalizer", "safety_normalizer", "entropy_coefficient",
        ):
            for key, value in sorted(self.modules[name].state_dict().items()):
                digest.update(f"{name}/{key}".encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(str((
            self.task_normalizer.frozen,
            self.safety_normalizer.frozen,
            tuple(parameter.requires_grad for module in (
                self.safety_q, self.safety_target, self.recovery_actor,
                self.entropy_coefficient,
            ) for parameter in module.parameters()),
        )).encode())
        return digest.hexdigest()

    def save(self, path, *, replay: ReplayBuffer, source_sha256: str,
             counters: dict, default_joint_position, action_scale):
        state = {
            "config": asdict(self.config),
            "modules": {name: module.state_dict() for name, module in self.modules.items()},
            "optimizers": {name: optimizer.state_dict()
                           for name, optimizer in self.optimizers.items()},
            "replay": replay.state_dict(),
            "steps": self.steps,
            "warmup_updates": self.warmup_updates,
            "source_sha256": source_sha256,
            "frozen_source_sha256": self.frozen_source_digest(),
            "counters": dict(counters),
        }
        metadata = checkpoint_metadata(
            task="recovery-rl", policy_view=POLICY_VIEW,
            policy_observation_size=RAW_OBSERVATION_SIZE,
            default_joint_position=default_joint_position,
            action_scale=action_scale,
            observation_scales={},
        )
        save_checkpoint(path, state=state, metadata=metadata)
