"""Lazy Isaac Lab bootstrap helpers; importing common never starts Isaac Sim."""

from __future__ import annotations

from pathlib import Path

from .observation import (
    RAW_OBSERVATION_ABI,
    RAW_OBSERVATION_SPEC,
    build_raw_observation_tensor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GO2_USD_PATH = PROJECT_ROOT / "assets/robots/go2/usd/go2.usd"


class IsaacVectorizedEnvironmentBase:
    """Backend-light base for vectorized Isaac tasks using the raw46 ABI.

    This module deliberately has no eager Isaac imports, preserving the
    required AppLauncher-before-Isaac-import ordering in every task.
    """

    raw_observation_abi = RAW_OBSERVATION_ABI

    @staticmethod
    def build_canonical_raw_observation(
        joint_position,
        joint_velocity,
        angular_velocity,
        estimated_body_velocity,
        base_quaternion_wxyz,
        previous_joint_target,
        *,
        previous_quaternion=None,
    ):
        raw = build_raw_observation_tensor(
            joint_position,
            joint_velocity,
            angular_velocity,
            estimated_body_velocity,
            base_quaternion_wxyz,
            previous_joint_target,
            previous_quaternion=previous_quaternion,
        )
        next_quaternion = raw[
            ..., RAW_OBSERVATION_SPEC.base_quaternion_wxyz
        ].clone()
        return raw, next_quaternion


def require_go2_usd() -> Path:
    if not GO2_USD_PATH.is_file():
        raise FileNotFoundError(f"missing Go2 USD: {GO2_USD_PATH}")
    return GO2_USD_PATH


def launch_app(*, headless: bool, device: str):
    """Start Isaac Sim before callers import simulation-dependent modules."""

    from isaaclab.app import AppLauncher

    return AppLauncher({"headless": bool(headless), "device": str(device)}).app
