"""Shared MuJoCo model and Go2 index helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .observation import GO2_JOINT_NAMES, build_raw_observation, continuous_quaternion_wxyz
from .types import RobotState


GO2_JOINT_LOWER_LIMIT = np.asarray(
    [-1.0472, -1.5708, -2.7227] * 2 + [-1.0472, -0.5236, -2.7227] * 2,
    dtype=np.float32,
)
GO2_JOINT_UPPER_LIMIT = np.asarray(
    [1.0472, 3.4907, -0.83776] * 2 + [1.0472, 4.5379, -0.83776] * 2,
    dtype=np.float32,
)


def load_model(xml_path: str | Path, *, physics_dt: float | None = None):
    import mujoco

    path = Path(xml_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    if physics_dt is not None:
        model.opt.timestep = float(physics_dt)
    return model, mujoco.MjData(model)


def go2_indices(model, *, joint_names=GO2_JOINT_NAMES):
    qpos = np.asarray([model.joint(name).qposadr[0] for name in joint_names], dtype=np.int64)
    qvel = np.asarray([model.joint(name).dofadr[0] for name in joint_names], dtype=np.int64)
    actuator = []
    for name in joint_names:
        # The retained CPG source names motors after links while the Recovery
        # source names them after joints.  Both resolve to the same shared
        # mechanical actuator ABI.
        actuator_name = name.removesuffix("_joint")
        try:
            actuator.append(model.actuator(name).id)
        except KeyError:
            actuator.append(model.actuator(actuator_name).id)
    actuator = np.asarray(actuator, dtype=np.int64)
    return qpos, qvel, actuator


def quaternion_rotation_matrix_wxyz(quaternion) -> np.ndarray:
    """Body-to-world rotation matrix for a normalized WXYZ quaternion."""

    w, x, y, z = continuous_quaternion_wxyz(quaternion)
    return np.asarray(
        [[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
         [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
         [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]],
        dtype=np.float64,
    )


def foot_position_velocity_body(joint_position, joint_velocity):
    """Analytic Go2 feet in canonical FL, FR, RL, RR order."""

    q = np.asarray(joint_position, np.float64).reshape(4, 3)
    dq = np.asarray(joint_velocity, np.float64).reshape(4, 3)
    abduction, thigh, calf = q.T
    abduction_dq, thigh_dq, calf_dq = dq.T
    total = thigh + calf
    side = np.asarray([1.0, -1.0, 1.0, -1.0])
    hip_x = np.asarray([0.1934, 0.1934, -0.1934, -0.1934])
    lateral = side * 0.0955
    x_plane = -0.213 * np.sin(thigh) - 0.213 * np.sin(total)
    z_plane = -0.213 * np.cos(thigh) - 0.213 * np.cos(total)
    position = np.stack(
        (hip_x + x_plane,
         side * 0.0465 + lateral * np.cos(abduction) - z_plane * np.sin(abduction),
         lateral * np.sin(abduction) + z_plane * np.cos(abduction)), axis=-1,
    )
    dx = (-0.213 * np.cos(thigh) - 0.213 * np.cos(total)) * thigh_dq \
        - 0.213 * np.cos(total) * calf_dq
    dz_plane = (0.213 * np.sin(thigh) + 0.213 * np.sin(total)) * thigh_dq \
        + 0.213 * np.sin(total) * calf_dq
    velocity = np.stack(
        (dx,
         (-lateral * np.sin(abduction) - z_plane * np.cos(abduction)) * abduction_dq
         - dz_plane * np.sin(abduction),
         (lateral * np.cos(abduction) - z_plane * np.sin(abduction)) * abduction_dq
         + dz_plane * np.cos(abduction)), axis=-1,
    )
    return position, velocity


class VelocityEstimator:
    """Contact-free IMU/leg-odometry filter used by target simulations."""

    def __init__(self, dt: float = 0.02):
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        self.world_velocity = np.zeros(3, np.float64)
        self.covariance = np.eye(3, dtype=np.float64) * 0.1

    def update(self, joint_position, joint_velocity, angular_velocity,
               quaternion_wxyz, accelerometer) -> np.ndarray:
        rotation = quaternion_rotation_matrix_wxyz(quaternion_wxyz)
        acceleration_world = rotation @ np.asarray(accelerometer, np.float64)
        self.world_velocity += (acceleration_world + [0.0, 0.0, -9.81]) * self.dt
        self.covariance += np.eye(3) * 0.03059 * self.dt * self.dt
        positions, relative_velocity = foot_position_velocity_body(joint_position, joint_velocity)
        candidates = -(relative_velocity + np.cross(angular_velocity, positions))
        height_delta = positions[:, 2] - np.min(positions[:, 2])
        confidence = np.exp(-0.5 * (height_delta / 0.05) ** 2
                            - 0.5 * (relative_velocity[:, 2] / 0.35) ** 2)
        if float(confidence.sum()) >= 0.2:
            predicted = rotation.T @ self.world_velocity
            residual_norm = np.linalg.norm(candidates - predicted, axis=1)
            huber = np.ones(4)
            outside = residual_norm > 0.25
            huber[outside] = 0.25 / residual_norm[outside]
            prior = np.exp(-(residual_norm - residual_norm.min()) / 0.05)
            weights = np.sqrt(np.sqrt(confidence)) * huber * prior
            if float(weights.sum()) > np.finfo(np.float64).eps:
                observed = np.average(candidates, axis=0, weights=weights)
                spread = np.average((candidates - observed) ** 2, axis=0, weights=weights)
                effective = weights.sum() ** 2 / max(float(weights @ weights), np.finfo(float).eps)
                measurement_body = np.diag(0.002 / max(effective, 1.0) + spread)
                measurement = rotation @ measurement_body @ rotation.T
                innovation = rotation @ observed - self.world_velocity
                innovation_covariance = self.covariance + measurement
                distance = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
                if np.isfinite(distance) and distance <= 11.34:
                    gain = np.linalg.solve(innovation_covariance.T, self.covariance.T).T
                    self.world_velocity += gain @ innovation
                    identity = np.eye(3) - gain
                    self.covariance = identity @ self.covariance @ identity.T \
                        + gain @ measurement @ gain.T
                elif np.isfinite(distance):
                    self.covariance *= 2.0
                    largest = float(np.diag(self.covariance).max())
                    if largest > 0.1:
                        self.covariance *= 0.1 / largest
                self.covariance = 0.5 * (self.covariance + self.covariance.T)
        if not np.isfinite(self.world_velocity).all():
            raise FloatingPointError("velocity estimator became non-finite")
        return (rotation.T @ self.world_velocity).astype(np.float32)


class MujocoObservationBuilder:
    """Stateful raw46 builder shared by MuJoCo target tasks."""

    def __init__(self, default_joint_position, dt: float = 0.02):
        self.default_joint_position = np.asarray(default_joint_position, np.float32)
        if self.default_joint_position.shape != (12,):
            raise ValueError("default_joint_position must have shape (12,)")
        self.estimator = VelocityEstimator(dt)
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.previous_quaternion = None
        self.previous_joint_target = self.default_joint_position.copy()

    def set_previous_joint_target(self, target) -> None:
        target = np.asarray(target, np.float32)
        if target.shape != (12,) or not np.isfinite(target).all():
            raise ValueError("previous joint target must contain twelve finite values")
        self.previous_joint_target = target.copy()

    def build(self, joint_position, joint_velocity, angular_velocity,
              quaternion_wxyz, accelerometer) -> np.ndarray:
        velocity = self.estimator.update(joint_position, joint_velocity,
                                         angular_velocity, quaternion_wxyz,
                                         accelerometer)
        return self._build(joint_position, joint_velocity, angular_velocity,
                           velocity, quaternion_wxyz)

    def initial(self, joint_position, joint_velocity, angular_velocity,
                quaternion_wxyz) -> np.ndarray:
        return self._build(joint_position, joint_velocity, angular_velocity,
                           np.zeros(3, np.float32), quaternion_wxyz)

    def _build(self, joint_position, joint_velocity, angular_velocity,
               estimated_body_velocity, quaternion_wxyz) -> np.ndarray:
        state = RobotState(joint_position, joint_velocity, angular_velocity,
                           estimated_body_velocity, quaternion_wxyz,
                           self.previous_joint_target)
        observation = build_raw_observation(state, self.previous_quaternion)
        self.previous_quaternion = observation[30:34].copy()
        return observation
