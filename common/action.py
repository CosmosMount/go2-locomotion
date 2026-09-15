"""Backend-neutral normalized action to absolute joint-target conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validated_vector(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (12,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain twelve finite values")
    return result


def target_from_normalized_action(action, default_joint_position, action_scale):
    action = _validated_vector(action, "action")
    default = _validated_vector(default_joint_position, "default_joint_position")
    scale = _validated_vector(action_scale, "action_scale")
    if np.any(scale <= 0):
        raise ValueError("action_scale must be positive")
    return default + scale * np.clip(action, -1.0, 1.0)


def normalized_action_from_target(q_target, default_joint_position, action_scale):
    target = _validated_vector(q_target, "q_target")
    default = _validated_vector(default_joint_position, "default_joint_position")
    scale = _validated_vector(action_scale, "action_scale")
    if np.any(scale <= 0):
        raise ValueError("action_scale must be positive")
    return np.clip((target - default) / scale, -1.0, 1.0).astype(np.float32)


@dataclass
class ActionMapper:
    default_joint_position: np.ndarray
    action_scale: np.ndarray
    control_dt: float = 0.02
    max_target_rate: float | None = None
    joint_lower_limit: np.ndarray | None = None
    joint_upper_limit: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.default_joint_position = _validated_vector(self.default_joint_position, "default_joint_position")
        self.action_scale = _validated_vector(self.action_scale, "action_scale")
        if (self.joint_lower_limit is None) != (self.joint_upper_limit is None):
            raise ValueError("joint limits must be supplied together")
        if self.joint_lower_limit is not None:
            self.joint_lower_limit = _validated_vector(self.joint_lower_limit, "joint_lower_limit")
            self.joint_upper_limit = _validated_vector(self.joint_upper_limit, "joint_upper_limit")
        self.previous_joint_target = self.default_joint_position.copy()

    def reset(self, previous_joint_target=None) -> None:
        self.previous_joint_target = _validated_vector(
            self.default_joint_position if previous_joint_target is None else previous_joint_target,
            "previous_joint_target",
        ).copy()

    def apply(self, action):
        target = target_from_normalized_action(action, self.default_joint_position, self.action_scale)
        if self.joint_lower_limit is not None:
            target = np.clip(target, self.joint_lower_limit, self.joint_upper_limit)
        if self.max_target_rate is not None:
            delta = float(self.max_target_rate) * float(self.control_dt)
            target = np.clip(target, self.previous_joint_target - delta, self.previous_joint_target + delta)
        applied = normalized_action_from_target(target, self.default_joint_position, self.action_scale)
        self.previous_joint_target = target.astype(np.float32, copy=True)
        return applied, self.previous_joint_target.copy()
