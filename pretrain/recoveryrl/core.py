"""Self-contained staged model-free Recovery RL components."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common.checkpoint import load_checkpoint
from common.observation import prepare_task_observation


POLICY_VIEW = "yaw_invariant_46d"


@dataclass(frozen=True)
class AgentConfig:
    obs_dim: int = 46
    action_dim: int = 12
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    gamma_safe: float = 0.9607894391523232
    tau: float = 0.005
    tau_safe: float = 0.0002
    safety_positive_fraction: float = 0.25
    risk_threshold_epsilon: float = 0.1
    alpha_init: float = 1.0
    automatic_entropy_tuning: bool = True
    target_entropy: float = -12.0
    safety_continuation: str = "task"
    task_observation_view: str = "yaw_invariant"
    seed: int = 1701

    def __post_init__(self):
        if self.obs_dim != 46 or self.action_dim != 12:
            raise ValueError("Go2 MF safety requires raw46/action12")
        if self.task_observation_view != "yaw_invariant":
            raise ValueError("task observation must be yaw_invariant")
        if self.safety_continuation != "task":
            raise ValueError("source safety target must continue with the task policy")
        if not 0 <= self.safety_positive_fraction <= 1:
            raise ValueError("safety_positive_fraction must lie in [0, 1]")


class RunningNormalizer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self.frozen = False

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        if self.frozen or value.numel() == 0:
            return
        count = value.shape[0]
        batch_mean = value.mean(0)
        batch_var = value.var(0, unbiased=False)
        total = self.count + count
        delta = batch_mean - self.mean
        self.var.copy_(
            (self.var * self.count + batch_var * count
             + delta.square() * self.count * count / total) / total
        )
        self.mean.add_(delta * count / total)
        self.count.copy_(total)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return ((value - self.mean) / torch.sqrt(self.var + 1e-6)).clamp(-10, 10)


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SquashedGaussianActor(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.net = _mlp(config.obs_dim, 2 * config.action_dim, config.hidden_dim)

    def sample(self, observation: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.net(observation).chunk(2, dim=-1)
        distribution = torch.distributions.Normal(mean, log_std.clamp(-20, 2).exp())
        latent = mean if deterministic else distribution.rsample()
        action = latent.tanh()
        log_probability = (
            distribution.log_prob(latent) - torch.log(1 - action.square() + 1e-6)
        ).sum(-1, keepdim=True)
        return action, log_probability


class TwinQ(nn.Module):
    def __init__(self, config: AgentConfig, *, sigmoid_output: bool):
        super().__init__()
        width = config.obs_dim + config.action_dim
        self.q1 = _mlp(width, 1, config.hidden_dim)
        self.q2 = _mlp(width, 1, config.hidden_dim)
        self.sigmoid_output = sigmoid_output

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        value = torch.cat((observation, action), dim=-1)
        first, second = self.q1(value), self.q2(value)
        if self.sigmoid_output:
            first, second = first.sigmoid(), second.sigmoid()
        return first, second


class EntropyCoefficient(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.tensor([np.log(alpha)], dtype=torch.float32)
        )


class ReplayBuffer:
    """Raw46 replay: task Q uses proposed action; Safety Q uses executed action."""

    def __init__(
        self, capacity: int, *, obs_dim: int = 46, action_dim: int = 12, seed: int = 0
    ):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)
        self.data = {
            key: np.empty((self.capacity, width), dtype=np.float32)
            for key, width in (
                ("obs", obs_dim),
                ("next_obs", obs_dim),
                ("proposed_action", action_dim),
                ("executed_action", action_dim),
                ("reward", 1),
                ("cost", 1),
                ("terminated", 1),
                ("truncated", 1),
            )
        }

    def __len__(self):
        return self.size

    def add_batch(
        self,
        obs,
        proposed_action,
        executed_action,
        reward,
        cost,
        next_obs,
        terminated,
        truncated,
    ) -> None:
        values = {
            "obs": obs,
            "next_obs": next_obs,
            "proposed_action": proposed_action,
            "executed_action": executed_action,
            "reward": np.asarray(reward).reshape(-1, 1),
            "cost": np.asarray(cost).reshape(-1, 1),
            "terminated": np.asarray(terminated).reshape(-1, 1),
            "truncated": np.asarray(truncated).reshape(-1, 1),
        }
        batch_size = len(values["obs"])
        if batch_size > self.capacity:
            raise ValueError("a replay batch cannot exceed capacity")
        indices = (self.position + np.arange(batch_size)) % self.capacity
        for key, value in values.items():
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (batch_size, self.data[key].shape[1]):
                raise ValueError(f"invalid replay field {key}: {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite replay field {key}")
            self.data[key][indices] = value
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    @property
    def positive_count(self) -> int:
        return int(np.count_nonzero(self.data["cost"][: self.size, 0] > 0))

    def sample(self, batch_size: int, device, *, positive_fraction=None):
        if self.size == 0:
            raise ValueError("cannot sample an empty replay")
        if positive_fraction is None:
            indices = self.rng.integers(self.size, size=batch_size)
        else:
            positives = np.flatnonzero(self.data["cost"][: self.size, 0] > 0)
            negatives = np.flatnonzero(self.data["cost"][: self.size, 0] <= 0)
            count = round(batch_size * float(positive_fraction))
            if len(positives) == 0 or len(negatives) == 0:
                indices = self.rng.integers(self.size, size=batch_size)
            else:
                indices = np.concatenate(
                    (
                        self.rng.choice(positives, count, replace=True),
                        self.rng.choice(negatives, batch_size - count, replace=True),
                    )
                )
                self.rng.shuffle(indices)
        return {
            key: torch.as_tensor(value[indices], device=device)
            for key, value in self.data.items()
        }


def safety_bellman_target(cost, terminated, next_risk, gamma_safe):
    """True terminals and violations stop risk return; timeouts bootstrap."""

    return cost + (1 - cost) * gamma_safe * (1 - terminated) * next_risk


class RecoveryAgent:
    """Task SAC plus reward Q, sigmoid Safety Q, and a recovery actor."""

    def __init__(self, config: AgentConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.task_actor = SquashedGaussianActor(config).to(self.device)
        self.recovery_actor = SquashedGaussianActor(config).to(self.device)
        self.reward_q = TwinQ(config, sigmoid_output=False).to(self.device)
        self.safety_q = TwinQ(config, sigmoid_output=True).to(self.device)
        self.reward_target = deepcopy(self.reward_q).requires_grad_(False)
        self.safety_target = deepcopy(self.safety_q).requires_grad_(False)
        self.task_normalizer = RunningNormalizer(config.obs_dim).to(self.device)
        self.safety_normalizer = RunningNormalizer(config.obs_dim).to(self.device)
        self.entropy_coefficient = EntropyCoefficient(config.alpha_init).to(self.device)
        self.modules = {
            name: getattr(self, name)
            for name in (
                "task_actor",
                "recovery_actor",
                "reward_q",
                "safety_q",
                "reward_target",
                "safety_target",
                "task_normalizer",
                "safety_normalizer",
                "entropy_coefficient",
            )
        }
        self.optimizers = {
            "task_actor": torch.optim.Adam(
                self.task_actor.parameters(), lr=config.learning_rate
            ),
            "recovery_actor": torch.optim.Adam(
                self.recovery_actor.parameters(), lr=config.learning_rate
            ),
            "reward_q": torch.optim.Adam(
                self.reward_q.parameters(), lr=config.learning_rate
            ),
            "safety_q": torch.optim.Adam(
                self.safety_q.parameters(), lr=config.learning_rate
            ),
            "alpha": torch.optim.Adam(
                self.entropy_coefficient.parameters(), lr=config.learning_rate
            ),
        }
        self.stage1_updates = 0
        self.task_updates = 0
        self.safety_updates = 0
        self.recovery_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.entropy_coefficient.log_alpha.exp().detach()

    def _tensor(self, observation) -> torch.Tensor:
        value = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if value.ndim != 2 or value.shape[1] != self.config.obs_dim:
            raise ValueError("expected raw observations shaped [N, 46]")
        return value

    def task_view(self, raw_observation: torch.Tensor) -> torch.Tensor:
        return self.task_normalizer(
            prepare_task_observation(raw_observation, "yaw_invariant")
        )

    def safety_view(self, raw_observation: torch.Tensor) -> torch.Tensor:
        return self.safety_normalizer(raw_observation)

    @torch.no_grad()
    def observe(self, raw_observation) -> None:
        raw = self._tensor(raw_observation)
        self.task_normalizer.update(
            prepare_task_observation(raw, "yaw_invariant")
        )
        self.safety_normalizer.update(raw)

    @torch.no_grad()
    def act_task(self, raw_observation, deterministic: bool = False) -> np.ndarray:
        raw = self._tensor(raw_observation)
        action, _ = self.task_actor.sample(self.task_view(raw), deterministic)
        return action.cpu().numpy()

    @torch.no_grad()
    def act_composite(self, raw_observation, deterministic: bool = False):
        raw = self._tensor(raw_observation)
        proposed, _ = self.task_actor.sample(self.task_view(raw), deterministic)
        safety = self.safety_view(raw)
        risk = torch.maximum(*self.safety_q(safety, proposed)).squeeze(-1)
        recovered = risk > self.config.risk_threshold_epsilon
        recovery, _ = self.recovery_actor.sample(safety, deterministic)
        executed = torch.where(recovered[:, None], recovery, proposed)
        return {
            "proposed_action": proposed.cpu().numpy(),
            "executed_action": executed.cpu().numpy(),
            "risk": risk.cpu().numpy(),
            "recovered": recovered.cpu().numpy(),
        }

    def _step(self, name: str, loss: torch.Tensor) -> float:
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"{name} loss is non-finite")
        loss.backward()
        gradients = [
            torch.isfinite(parameter.grad).all()
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if gradients and not bool(torch.stack(gradients).all()):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"{name} gradient is non-finite")
        optimizer.step()
        return float(loss.detach())

    @staticmethod
    @torch.no_grad()
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            target_parameter.lerp_(source_parameter, tau)

    def _update_entropy(self, log_probability: torch.Tensor):
        entropy = -log_probability.detach()
        loss = self.entropy_coefficient.log_alpha.exp() * (
            entropy - self.config.target_entropy
        )
        alpha_loss = self._step("alpha", loss.mean())
        return {
            "alpha": float(self.alpha),
            "alpha_loss": alpha_loss,
            "entropy": float(entropy.mean()),
        }

    def update_stage1(self, replay: ReplayBuffer) -> dict[str, float]:
        """Compatibility wrapper for one paired task/safety source update."""

        metrics = self.update_task(replay)
        metrics.update(self.update_safety(replay))
        self.stage1_updates += 1
        return metrics

    def update_task(self, replay: ReplayBuffer) -> dict[str, float]:
        """Run one Task SAC update through the task pretraining component."""

        from .task import update_task

        return update_task(self, replay)

    def update_safety(self, replay: ReplayBuffer) -> dict[str, float]:
        """Run one Safety-Q update through the safety pretraining component."""

        from .safety import update_safety

        return update_safety(self, replay)

    def freeze_stage1(self) -> None:
        """Freeze task/safety dependencies before MF recovery pretraining."""

        from .recovery import freeze_dependencies

        freeze_dependencies(self)

    def update_recovery(self, replay: ReplayBuffer) -> dict[str, float]:
        """Run one MF recovery update through the recovery component."""

        from .recovery import update_recovery

        return update_recovery(self, replay)

    def prepare_for_adaptation(self) -> None:
        """Set transfer-time permissions: task learner mutable, safety stack frozen."""

        for module in (self.task_actor, self.reward_q):
            module.train().requires_grad_(True)
        for module in (
            self.reward_target,
            self.safety_q,
            self.safety_target,
            self.recovery_actor,
            self.entropy_coefficient,
        ):
            module.eval().requires_grad_(False)
        self.task_normalizer.frozen = True
        self.safety_normalizer.frozen = True

    def checkpoint_state(self, training_state: dict) -> dict:
        """Return the state inside the shared common checkpoint envelope."""

        transfer_config = {
            "obs_dim": self.config.obs_dim,
            "action_dim": self.config.action_dim,
            "hidden": self.config.hidden_dim,
            "lr": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "gamma": self.config.gamma,
            "gamma_safe": self.config.gamma_safe,
            "epsilon": self.config.risk_threshold_epsilon,
            "tau": self.config.tau,
            "tau_safe": self.config.tau_safe,
            "safety_continuation": self.config.safety_continuation,
            "safety_pos_fraction": self.config.safety_positive_fraction,
            "alpha": self.config.alpha_init,
            "automatic_entropy_tuning": self.config.automatic_entropy_tuning,
            "target_entropy": self.config.target_entropy,
            "seed": self.config.seed,
            "task_observation_profile": self.config.task_observation_view,
        }
        return {
            "config": transfer_config,
            "modules": {name: module.state_dict() for name, module in self.modules.items()},
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
            "training": {
                **training_state,
                "stage1_updates": self.stage1_updates,
                "task_updates": self.task_updates,
                "safety_updates": self.safety_updates,
                "recovery_updates": self.recovery_updates,
                "normalizers_frozen": True,
                "safety_stack_frozen": True,
                "transfer_ready": True,
            },
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

    @classmethod
    def load_for_adaptation(
        cls, checkpoint: str | Path, device: str | torch.device = "cpu"
    ) -> tuple["RecoveryAgent", dict]:
        """Load a source checkpoint with only the target task learner unfrozen."""

        payload = load_checkpoint(
            checkpoint, expected_task="safe", map_location=device
        )
        bundle = payload["state"]
        config = bundle["config"]
        agent_config = AgentConfig(
            obs_dim=config["obs_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config["hidden"],
            learning_rate=config["lr"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            gamma_safe=config["gamma_safe"],
            tau=config["tau"],
            tau_safe=config["tau_safe"],
            safety_positive_fraction=config["safety_pos_fraction"],
            risk_threshold_epsilon=config["epsilon"],
            alpha_init=config["alpha"],
            automatic_entropy_tuning=config["automatic_entropy_tuning"],
            target_entropy=config["target_entropy"],
            safety_continuation=config["safety_continuation"],
            task_observation_view=config["task_observation_profile"],
            seed=config["seed"],
        )
        agent = cls(agent_config, device)
        for name, state in bundle["modules"].items():
            agent.modules[name].load_state_dict(state)
        for name, state in bundle.get("optimizers", {}).items():
            if name in agent.optimizers:
                agent.optimizers[name].load_state_dict(state)
        stage1_updates = int(bundle["training"]["stage1_updates"])
        agent.stage1_updates = stage1_updates
        agent.task_updates = int(bundle["training"].get("task_updates", stage1_updates))
        agent.safety_updates = int(
            bundle["training"].get("safety_updates", stage1_updates)
        )
        agent.recovery_updates = int(bundle["training"]["recovery_updates"])
        agent.prepare_for_adaptation()
        return agent, payload
