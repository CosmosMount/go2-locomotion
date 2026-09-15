"""The raw observation ABI and task-specific projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .types import RobotState


RAW_OBSERVATION_ABI = "go2_raw_46d"
RAW_OBSERVATION_SIZE = 46
PPO_OBSERVATION_ABI = "go2_ppo_45d"
TASK_OBSERVATION_PROFILES = ("raw", "yaw_invariant")
GO2_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)


@dataclass(frozen=True)
class RawObservationSpec:
    abi: str = RAW_OBSERVATION_ABI
    size: int = RAW_OBSERVATION_SIZE
    joint_position: slice = field(default_factory=lambda: slice(0, 12))
    joint_velocity: slice = field(default_factory=lambda: slice(12, 24))
    angular_velocity: slice = field(default_factory=lambda: slice(24, 27))
    estimated_body_velocity: slice = field(default_factory=lambda: slice(27, 30))
    base_quaternion_wxyz: slice = field(default_factory=lambda: slice(30, 34))
    previous_joint_target: slice = field(default_factory=lambda: slice(34, 46))
    joint_order: tuple[str, ...] = GO2_JOINT_NAMES
    quaternion_order: str = "WXYZ"
    angular_velocity_frame: str = "body"
    estimated_velocity_frame: str = "body"


RAW_OBSERVATION_SPEC = RawObservationSpec()


def _normalized_joint_name(name: str) -> str:
    name = str(name)
    return name if name.endswith("_joint") else name + "_joint"


def joint_order_indices(source_names: Sequence[str]) -> np.ndarray:
    """Gather indices that convert a backend order to the canonical order."""

    normalized = [_normalized_joint_name(name) for name in source_names]
    duplicate = {name for name in normalized if normalized.count(name) > 1}
    if duplicate:
        raise ValueError(f"duplicate joint names: {sorted(duplicate)}")
    missing = [name for name in GO2_JOINT_NAMES if name not in normalized]
    if missing:
        raise ValueError(f"missing Go2 joints: {missing}")
    return np.asarray([normalized.index(name) for name in GO2_JOINT_NAMES], dtype=np.int64)


def reorder_joints(values, source_names: Sequence[str]):
    """Reorder the final dimension of a NumPy array or Torch tensor."""

    indices = joint_order_indices(source_names)
    if _is_torch(values):
        import torch

        return values.index_select(-1, torch.as_tensor(indices, device=values.device))
    return np.take(np.asarray(values), indices, axis=-1)


def continuous_quaternion_wxyz(quaternion, previous=None):
    """Normalize WXYZ quaternion(s) and choose a temporally continuous sign."""

    if _is_torch(quaternion):
        import torch

        q = torch.as_tensor(quaternion)
        if q.shape[-1] != 4:
            raise ValueError("quaternion must end in four WXYZ values")
        norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        if not torch.isfinite(norm).all() or (norm < 1.0e-8).any():
            raise ValueError("invalid quaternion")
        q = q / norm
        if previous is None:
            reference = q[..., :1]
        else:
            previous = torch.as_tensor(previous, device=q.device, dtype=q.dtype)
            reference = (q * previous).sum(-1, keepdim=True)
        return torch.where(reference < 0, -q, q)
    # Preserve float64 sensor precision on MuJoCo/control paths while keeping
    # float32 inputs (the usual policy-training representation) as float32.
    # The ABI fixes layout and semantics, not a backend-specific scalar type.
    dtype = np.result_type(np.asarray(quaternion).dtype, np.float32)
    q = np.asarray(quaternion, dtype=dtype)
    if q.shape[-1] != 4:
        raise ValueError("quaternion must end in four WXYZ values")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norm)) or np.any(norm < 1.0e-8):
        raise ValueError("invalid quaternion")
    q = q / norm
    reference = q[..., :1] if previous is None else np.sum(
        q * np.asarray(previous, dtype=q.dtype), axis=-1, keepdims=True
    )
    return np.where(reference < 0, -q, q).astype(dtype, copy=False)


def build_raw_observation(state: RobotState, previous_quaternion=None):
    """Build one or a batch of canonical NumPy observations."""

    fields = (
        state.joint_position,
        state.joint_velocity,
        state.angular_velocity,
        state.estimated_body_velocity,
        state.base_quaternion_wxyz,
        state.previous_joint_target,
    )
    dtype = np.result_type(*(np.asarray(value).dtype for value in fields), np.float32)
    values = (
        np.asarray(state.joint_position, dtype=dtype),
        np.asarray(state.joint_velocity, dtype=dtype),
        np.asarray(state.angular_velocity, dtype=dtype),
        np.asarray(state.estimated_body_velocity, dtype=dtype),
        continuous_quaternion_wxyz(state.base_quaternion_wxyz, previous_quaternion),
        np.asarray(state.previous_joint_target, dtype=dtype),
    )
    observation = np.concatenate(values, axis=-1).astype(dtype, copy=False)
    _validate_observation(observation)
    return observation


def build_raw_observation_tensor(
    joint_position, joint_velocity, angular_velocity, estimated_body_velocity,
    base_quaternion_wxyz, previous_joint_target, previous_quaternion=None,
):
    """Torch equivalent of :func:`build_raw_observation`."""

    import torch

    q = continuous_quaternion_wxyz(base_quaternion_wxyz, previous_quaternion)
    observation = torch.cat(
        (joint_position, joint_velocity, angular_velocity,
         estimated_body_velocity, q, previous_joint_target), dim=-1,
    )
    if observation.shape[-1] != RAW_OBSERVATION_SIZE or not torch.isfinite(observation).all():
        raise ValueError(f"raw observation must end in {RAW_OBSERVATION_SIZE} finite values")
    return observation


def projected_gravity(quaternion_wxyz):
    """World-down unit vector expressed in the body frame."""

    q = continuous_quaternion_wxyz(quaternion_wxyz)
    if _is_torch(q):
        import torch

        w, x, y, z = q.unbind(-1)
        return torch.stack(
            (2 * (w * y - x * z), -2 * (w * x + y * z),
             -(1 - 2 * (x.square() + y.square()))), dim=-1,
        )
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (2 * (w * y - x * z), -2 * (w * x + y * z),
         -(1 - 2 * (x * x + y * y))), axis=-1,
    ).astype(np.float32, copy=False)


def ppo_observation(
    raw_observation, command, *, default_joint_position, action_scale,
    angular_velocity_scale: float = 0.2, joint_velocity_scale: float = 0.05,
    command_scale=(1.0, 1.0, 1.0), clip: float = 100.0,
):
    """Project canonical raw46 plus a command context to the PPO 45D view."""

    spec = RAW_OBSERVATION_SPEC
    if _is_torch(raw_observation):
        import torch

        raw = raw_observation
        if raw.shape[-1] != RAW_OBSERVATION_SIZE:
            raise ValueError("raw observation must end in 46 values")
        default = torch.as_tensor(default_joint_position, dtype=raw.dtype, device=raw.device)
        scale = torch.as_tensor(action_scale, dtype=raw.dtype, device=raw.device)
        command = torch.as_tensor(command, dtype=raw.dtype, device=raw.device)
        command_scale = torch.as_tensor(command_scale, dtype=raw.dtype, device=raw.device)
        view = torch.cat(
            (angular_velocity_scale * raw[..., spec.angular_velocity],
             projected_gravity(raw[..., spec.base_quaternion_wxyz]),
             command * command_scale,
             raw[..., spec.joint_position] - default,
             joint_velocity_scale * raw[..., spec.joint_velocity],
             ((raw[..., spec.previous_joint_target] - default) / scale).clamp(-1, 1)),
            dim=-1,
        )
        return view.clamp(-clip, clip)
    raw = np.asarray(raw_observation, dtype=np.float32)
    _validate_observation(raw)
    default = np.asarray(default_joint_position, dtype=np.float32)
    scale = np.asarray(action_scale, dtype=np.float32)
    view = np.concatenate(
        (angular_velocity_scale * raw[..., spec.angular_velocity],
         projected_gravity(raw[..., spec.base_quaternion_wxyz]),
         np.asarray(command, dtype=np.float32) * np.asarray(command_scale, dtype=np.float32),
         raw[..., spec.joint_position] - default,
         joint_velocity_scale * raw[..., spec.joint_velocity],
         np.clip((raw[..., spec.previous_joint_target] - default) / scale, -1, 1)),
        axis=-1,
    ).astype(np.float32, copy=False)
    if view.shape[-1] != 45 or not np.all(np.isfinite(view)):
        raise ValueError("PPO observation must end in 45 finite values")
    return np.clip(view, -clip, clip)


def prepare_task_observation(observation, profile: str = "yaw_invariant"):
    """Return a task-policy view while leaving replay and safety data raw."""

    if profile not in TASK_OBSERVATION_PROFILES:
        raise ValueError(f"unknown task observation profile: {profile}")
    if profile == "raw":
        return observation
    if observation.shape[-1] != RAW_OBSERVATION_SIZE:
        raise ValueError("yaw_invariant requires canonical raw46")
    q_slice = RAW_OBSERVATION_SPEC.base_quaternion_wxyz
    if _is_torch(observation):
        import torch

        q = continuous_quaternion_wxyz(observation[..., q_slice])
        w, x, y, z = q.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        c, s = torch.cos(yaw / 2), torch.sin(yaw / 2)
        no_yaw = torch.stack(
            (c * w + s * z, c * x + s * y, c * y - s * x, c * z - s * w), dim=-1,
        )
        return torch.cat((observation[..., :q_slice.start], no_yaw,
                          observation[..., q_slice.stop:]), dim=-1)
    raw = np.asarray(observation, dtype=np.float32)
    q = continuous_quaternion_wxyz(raw[..., q_slice])
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw / 2), np.sin(yaw / 2)
    no_yaw = np.stack(
        (c * w + s * z, c * x + s * y, c * y - s * x, c * z - s * w), axis=-1,
    )
    return np.concatenate((raw[..., :q_slice.start], no_yaw,
                           raw[..., q_slice.stop:]), axis=-1).astype(np.float32)


def _validate_observation(observation) -> None:
    if observation.shape[-1] != RAW_OBSERVATION_SIZE or not np.all(np.isfinite(observation)):
        raise ValueError(f"raw observation must end in {RAW_OBSERVATION_SIZE} finite values")


def _is_torch(value) -> bool:
    return type(value).__module__.startswith("torch")
