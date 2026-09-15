"""Small backend-neutral data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RobotState:
    """One proprioceptive Go2 state in canonical joint order."""

    joint_position: Any
    joint_velocity: Any
    angular_velocity: Any
    estimated_body_velocity: Any
    base_quaternion_wxyz: Any
    previous_joint_target: Any
