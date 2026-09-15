"""The single CPG/reflex controller shared by search and evaluation."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from common.observation import RAW_OBSERVATION_SIZE, RAW_OBSERVATION_SPEC


PARAMETER_NAMES = (
    "frequency",
    "amplitude",
    "front_lift",
    "rear_lift",
    "height_offset",
    "pitch_gain",
    "roll_gain",
    "forward_swing_offset",
)

# Canonical leg order is FL, FR, RL, RR.
_SIDE = np.array([1.0, -1.0, 1.0, -1.0])
_FRONT = np.array([1.0, 1.0, -1.0, -1.0])


def _roll_pitch_yaw_wxyz(quaternion: np.ndarray) -> tuple[float, float, float]:
    # Preserve the source controller's Euler convention and numerical path.
    # CPG contacts are hybrid dynamics, so sub-ulp differences can select a
    # different contact mode after hundreds of integration steps.
    roll, pitch, yaw = Rotation.from_quat(
        np.roll(np.asarray(quaternion, dtype=np.float64), -1)
    ).as_euler("xyz")
    return float(roll), float(pitch), float(yaw)


def cpg_control(
    raw_observation: np.ndarray,
    phase: np.ndarray,
    imu_odometry: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    """Map canonical raw46 state plus controller state to a 20D CPG command."""
    # Keep MuJoCo's float64 quaternion precision.  The selected CPG parameters
    # sit close to contact-mode boundaries, where an eager float32 conversion
    # can change the deterministic staircase outcome after hundreds of steps.
    raw = np.asarray(raw_observation)
    phase = np.asarray(phase, dtype=np.float64)
    odometry = np.asarray(imu_odometry, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if raw.shape != (RAW_OBSERVATION_SIZE,) or not np.isfinite(raw).all():
        raise ValueError(f"raw_observation must be a finite {RAW_OBSERVATION_SIZE}-vector")
    if phase.shape != (4,) or odometry.shape != (3,) or parameters.shape != (8,):
        raise ValueError("expected phase[4], imu_odometry[3], parameters[8]")
    if not all(np.isfinite(value).all() for value in (phase, odometry, parameters)):
        raise ValueError("controller inputs must be finite")

    frequency, amplitude, front_lift, rear_lift, height_offset, pitch_gain, \
        roll_gain, forward_swing_offset = parameters
    quaternion = raw[RAW_OBSERVATION_SPEC.base_quaternion_wxyz]
    if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-4:
        raise ValueError("raw_observation quaternion must be normalized")
    roll, pitch, yaw = _roll_pitch_yaw_wxyz(quaternion)
    gate = np.maximum(0.0, np.sin(phase))

    command = np.zeros(20, dtype=np.float64)
    command[:4] = frequency
    command[4:8] = np.clip(
        amplitude + 2.0 * (yaw + odometry[1]) * _SIDE, -1.0, 1.0
    )
    feet = command[8:].reshape(4, 3)
    feet[:, 0] = forward_swing_offset * gate
    feet[:, 1] = np.clip(2.0 * odometry[1], -0.6, 0.6)
    lift = np.array([front_lift, front_lift, rear_lift, rear_lift])
    feet[:, 2] = np.clip(
        (roll_gain * roll * _SIDE * 0.14 - pitch_gain * pitch * _FRONT * 0.1934)
        / 0.12
        + height_offset
        + gate * lift,
        -1.0,
        1.0,
    )
    return command
