"""Simulator-independent recovery reward terms migrated from FR-Net."""

from __future__ import annotations

import torch


STAND_HIGH = ([-0.05, 0.8, -1.5, 0.05, 0.8, -1.5]
              + [-0.05, 1.0, -1.5, 0.05, 1.0, -1.5])
STAND_LOW = [0.0, 1.5, -2.4] * 4


def recovery_reward_terms(
    *, projected_gravity: torch.Tensor, base_height: torch.Tensor,
    foot_contacts: torch.Tensor, joint_position: torch.Tensor,
    joint_velocity: torch.Tensor, joint_acceleration: torch.Tensor,
    applied_torque: torch.Tensor, action: torch.Tensor,
    previous_action: torch.Tensor, previous_previous_action: torch.Tensor,
    soft_joint_limits: torch.Tensor, base_contacts: torch.Tensor,
    foot_contact_forces: torch.Tensor, iteration: int,
    stand_curriculum_iterations: int,
) -> dict[str, torch.Tensor]:
    """Compute unscaled terms; penalties are returned as positive magnitudes."""

    device, dtype = joint_position.device, joint_position.dtype
    target_gravity = torch.tensor((0.0, 0.0, -1.0), device=device, dtype=dtype)
    orientation = torch.sum((projected_gravity - target_gravity).square(), dim=-1)
    orientation_pos = torch.exp(-((projected_gravity[:, 2] + 1.0).square()) / (2.0 * 0.15**2))
    base_height_pos = torch.exp(-(0.45 - base_height).square())
    foot_contact_pos = foot_contacts.to(dtype).sum(dim=-1)

    progress = min(float(iteration) / float(stand_curriculum_iterations), 1.0)
    low = torch.tensor(STAND_LOW, device=device, dtype=dtype)
    high = torch.tensor(STAND_HIGH, device=device, dtype=dtype)
    target_joint_position = low + progress * (high - low)
    upright_neighborhood = (projected_gravity[:, 2] + 1.0).abs() <= 0.2
    stand_error = torch.sum((joint_position - target_joint_position).square(), dim=-1)
    step2_target_pos = torch.where(upright_neighborhood, torch.exp(-stand_error), torch.zeros_like(stand_error))

    below = (soft_joint_limits[..., 0] - joint_position).clamp_min(0.0)
    above = (joint_position - soft_joint_limits[..., 1]).clamp_min(0.0)
    second_difference = action - 2.0 * previous_action + previous_previous_action
    smoothness_mask = (previous_action != 0.0) & (previous_previous_action != 0.0)
    lateral = torch.linalg.vector_norm(foot_contact_forces[..., :2], dim=-1)
    vertical = foot_contact_forces[..., 2].abs()

    return {
        "orientation": orientation,
        "orientation_pos_gaussian": orientation_pos,
        "base_height_pos": base_height_pos,
        "foot_contact_pos": foot_contact_pos,
        "step2_target_pos": step2_target_pos,
        "angular_velocity_xyz": torch.zeros_like(orientation),  # filled by the environment
        "torques": torch.sum(applied_torque.square(), dim=-1),
        "joint_acceleration": torch.sum(joint_acceleration.square(), dim=-1),
        "joint_velocity": torch.sum(joint_velocity.square(), dim=-1),
        "joint_position_limits": torch.sum(below + above, dim=-1),
        "collision": base_contacts.to(dtype).sum(dim=-1),
        "action": torch.sum(action.square(), dim=-1),
        "action_rate": torch.sum((action - previous_action).square(), dim=-1),
        "action_smoothness_2": torch.sum(second_difference.square() * smoothness_mask, dim=-1),
        "max_velocity": torch.sum((joint_velocity.abs() - 0.8).clamp_min(0.0), dim=-1),
        "feet_stumble": torch.any(lateral > 4.0 * vertical, dim=-1).to(dtype),
    }


def weighted_reward(terms: dict[str, torch.Tensor], scales: dict, step_dt: float, *, only_positive: bool) -> torch.Tensor:
    missing = set(scales) - set(terms)
    if missing:
        raise KeyError(f"missing recovery reward terms: {sorted(missing)}")
    reward = torch.stack([terms[name] * float(scale) * step_dt for name, scale in scales.items()]).sum(dim=0)
    return reward.clamp_min(0.0) if only_positive else reward
