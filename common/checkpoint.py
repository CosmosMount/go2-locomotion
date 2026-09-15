"""Checkpoint envelope shared by training and adaptation."""

from __future__ import annotations

from pathlib import Path

from .observation import GO2_JOINT_NAMES, RAW_OBSERVATION_ABI, RAW_OBSERVATION_SIZE


CHECKPOINT_FORMAT = "go2_multitask_checkpoint"


def checkpoint_metadata(
    *, task: str, policy_view: str, policy_observation_size: int,
    default_joint_position, action_scale, observation_scales: dict | None = None,
) -> dict:
    return {
        "format": CHECKPOINT_FORMAT, "task": str(task),
        "raw_observation_abi": RAW_OBSERVATION_ABI,
        "raw_observation_size": RAW_OBSERVATION_SIZE,
        "policy_view": str(policy_view),
        "policy_observation_size": int(policy_observation_size),
        "joint_order": list(GO2_JOINT_NAMES), "quaternion_order": "WXYZ",
        "angular_velocity_frame": "body", "estimated_velocity_frame": "body",
        "default_joint_position": [float(value) for value in default_joint_position],
        "action_scale": [float(value) for value in action_scale],
        "observation_scales": dict(observation_scales or {}),
    }


def validate_metadata(metadata: dict, *, expected_task: str | None = None) -> None:
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    if metadata.get("raw_observation_abi") != RAW_OBSERVATION_ABI:
        raise ValueError("checkpoint uses an incompatible raw observation ABI")
    if tuple(metadata.get("joint_order", ())) != GO2_JOINT_NAMES:
        raise ValueError("checkpoint uses an incompatible Go2 joint order")
    if metadata.get("quaternion_order") != "WXYZ":
        raise ValueError("checkpoint quaternion order must be WXYZ")
    if expected_task is not None and metadata.get("task") != expected_task:
        raise ValueError(f"expected {expected_task!r} checkpoint, got {metadata.get('task')!r}")


def save_checkpoint(path: str | Path, *, state: dict, metadata: dict) -> None:
    import torch

    validate_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "state": state}, path)


def load_checkpoint(path: str | Path, *, expected_task: str | None = None, map_location="cpu") -> dict:
    import torch

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict) or "state" not in payload:
        raise ValueError("checkpoint must contain metadata and state")
    validate_metadata(payload["metadata"], expected_task=expected_task)
    return payload
