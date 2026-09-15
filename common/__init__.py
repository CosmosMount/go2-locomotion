"""Shared contracts used by every Go2 task in this repository."""

from .action import ActionMapper, normalized_action_from_target, target_from_normalized_action
from .checkpoint import checkpoint_metadata, load_checkpoint, save_checkpoint
from .observation import (
    GO2_JOINT_NAMES,
    RAW_OBSERVATION_ABI,
    RAW_OBSERVATION_SIZE,
    RAW_OBSERVATION_SPEC,
    RawObservationSpec,
    build_raw_observation,
    build_raw_observation_tensor,
    ppo_observation,
    prepare_task_observation,
    projected_gravity,
    reorder_joints,
)
from .types import RobotState

__all__ = [
    "ActionMapper", "GO2_JOINT_NAMES", "RAW_OBSERVATION_ABI",
    "RAW_OBSERVATION_SIZE", "RAW_OBSERVATION_SPEC", "RawObservationSpec",
    "RobotState", "build_raw_observation", "build_raw_observation_tensor",
    "checkpoint_metadata", "load_checkpoint", "normalized_action_from_target",
    "ppo_observation", "prepare_task_observation", "projected_gravity",
    "reorder_joints", "save_checkpoint", "target_from_normalized_action",
]
