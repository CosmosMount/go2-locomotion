"""Minimal 50 Hz Go2 target-domain MuJoCo staircase simulation."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.mujoco import (
    MujocoObservationBuilder,
    go2_indices,
    quaternion_rotation_matrix_wxyz,
)
from common.observation import (
    GO2_JOINT_NAMES,
    RAW_OBSERVATION_ABI,
    RAW_OBSERVATION_SPEC,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
MODEL_PROFILE = PROJECT_ROOT / "assets" / "robots" / "go2" / "mjcf" / "go2.xml"
LEG_ORDER = ("FL", "FR", "RL", "RR")
SIDE = np.array([1.0, -1.0, 1.0, -1.0])


def build_stair_scene(path: Path, terrain: dict) -> Path:
    """Write a deterministic scene around the local dynamics-only Go2 profile."""
    height = float(terrain["step_height"])
    count = int(terrain["step_count"])
    depth = float(terrain["step_depth"])
    width = float(terrain["width"])
    landing = float(terrain["landing_length"])
    start = float(terrain["stair_start"])
    friction = float(terrain["friction"])
    dimensions = (height, depth, width, landing, start, friction)
    if not all(math.isfinite(value) for value in dimensions):
        raise ValueError("terrain dimensions must be finite")
    if height <= 0 or count != 4 or depth <= 0 or width <= 0.8:
        raise ValueError("expected positive four-step staircase dimensions")
    if landing <= 0.6 or start < 0.5 or friction <= 0:
        raise ValueError("invalid staircase landing, start, or friction")

    root = ET.Element("mujoco", {"model": "go2 4cm staircase"})
    ET.SubElement(root, "include", {"file": str(MODEL_PROFILE)})
    ET.SubElement(root, "statistic", {"center": "1.4 0 0.1", "extent": "2.2"})
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.05",
            "friction": f"{friction} .02 .01",
            "rgba": ".18 .22 .27 1",
        },
    )
    for index in range(count + 1):
        length = depth if index < count else landing
        stair_height = min(index + 1, count) * height
        ET.SubElement(
            world,
            "geom",
            {
                "name": f"stair_{index}",
                "type": "box",
                "pos": f"{start + index * depth + length / 2} 0 {stair_height / 2}",
                "size": f"{length / 2} {width / 2} {stair_height / 2}",
                "friction": f"{friction} .02 .01",
                "rgba": ".55 .65 .72 1",
                "condim": "6",
            },
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


class Go2CpgEnvironment:
    """Sensor-facing simulation that exposes only the canonical raw46 ABI."""

    def __init__(self, work_dir: Path, config: dict):
        self.simulation = config["simulation"]
        self.terrain = config["terrain"]
        scene_path = build_stair_scene(Path(work_dir) / "scene.xml", self.terrain)
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        physics_dt = float(self.simulation["physics_dt"])
        control_dt = float(self.simulation["control_dt"])
        if not math.isclose(self.model.opt.timestep, physics_dt, abs_tol=1e-12):
            raise ValueError("MJCF timestep does not match simulation.physics_dt")
        self.substeps = int(round(control_dt / physics_dt))
        if self.substeps < 1 or not math.isclose(
            self.substeps * physics_dt, control_dt, abs_tol=1e-12
        ):
            raise ValueError("control_dt must be an integer multiple of physics_dt")

        joint_ids = np.array(
            [
                self.model.joint(name).id
                for name in GO2_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.qpos_indices, self.qvel_indices, self.actuator_indices = go2_indices(
            self.model
        )
        self.joint_ranges = self.model.jnt_range[joint_ids]
        self.foot_geom_ids = np.array(
            [self.model.geom(leg).id for leg in LEG_ORDER], dtype=np.int32
        )
        self.base_body_id = self.model.body("base_link").id
        self.default_joint_position = np.tile(
            np.asarray(self.simulation["default_joint_position"], dtype=np.float64), 4
        )
        if self.default_joint_position.shape != (12,):
            raise ValueError("simulation.default_joint_position must contain three values")
        self.previous_joint_target = self.default_joint_position.copy()
        self.observation_builder = MujocoObservationBuilder(
            self.default_joint_position, dt=control_dt
        )

    @property
    def imu_odometry(self) -> np.ndarray:
        return self._imu_odometry.copy()

    def _physics_step(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        quaternion = np.asarray(self.data.sensor("cpg_imu_quat").data, dtype=np.float64)
        rotation = quaternion_rotation_matrix_wxyz(quaternion)
        acceleration_world = (
            rotation @ self.data.sensor("cpg_imu_acc").data
            + np.array([0.0, 0.0, -9.81])
        )
        dt = float(self.simulation["physics_dt"])
        self._imu_velocity += acceleration_world * dt
        self._imu_odometry += self._imu_velocity * dt
        mujoco.mj_step(self.model, self.data)

    def observe(self) -> np.ndarray:
        self.observation_builder.set_previous_joint_target(
            self.previous_joint_target
        )
        return self.observation_builder.build(
            self.data.qpos[self.qpos_indices],
            self.data.qvel[self.qvel_indices],
            self.data.sensor("cpg_imu_gyro").data,
            self.data.sensor("cpg_imu_quat").data,
            self.data.sensor("cpg_imu_acc").data,
        )

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_indices] = self.default_joint_position
        self.data.qpos[:2] += rng.uniform(-0.02, 0.02, 2)
        self._imu_velocity = np.zeros(3, dtype=np.float64)
        self._imu_odometry = np.zeros(3, dtype=np.float64)
        # Equivalent to source order FR, FL, RR, RL = [0, pi, pi, 0].
        self.phase = np.array([np.pi, 0.0, 0.0, np.pi], dtype=np.float64)
        self.previous_command = np.zeros(20, dtype=np.float64)
        self.previous_joint_target = self.default_joint_position.copy()
        self.observation_builder.reset()
        self.steps = 0
        self.hold = 0
        self.total_reward = 0.0
        self.max_x = float(self.data.qpos[0])
        self.success = False

        kp = float(self.simulation["kp"])
        kd = float(self.simulation["kd"])
        torque_limit = float(self.simulation["torque_limit"])
        for _ in range(int(self.simulation["settle_steps"])):
            torque = kp * (
                self.default_joint_position - self.data.qpos[self.qpos_indices]
            ) - kd * self.data.qvel[self.qvel_indices]
            self.data.ctrl[self.actuator_indices] = np.clip(
                torque, -torque_limit, torque_limit
            )
            self._physics_step()
        mujoco.mj_forward(self.model, self.data)
        return self.observation_builder.initial(
            self.data.qpos[self.qpos_indices],
            self.data.qvel[self.qvel_indices],
            self.data.sensor("cpg_imu_gyro").data,
            self.data.sensor("cpg_imu_quat").data,
        )

    def _joint_target(self, command: np.ndarray) -> np.ndarray:
        sine = np.sin(self.phase)
        x = -0.16 * (0.5 + 0.35 * command[4:8]) * np.cos(self.phase)
        z = -0.28 + np.where(sine > 0.0, 0.12 * sine, 0.01 * sine)
        feet = np.column_stack((x, SIDE * 0.0955, z))
        feet += command[8:].reshape(4, 3) * np.array([0.07, 0.035, 0.12])
        x, y, z = feet.T
        knee = -np.arccos(
            np.clip(
                (x * x + y * y + z * z - 0.0955**2 - 2 * 0.213**2)
                / (2 * 0.213**2),
                -1.0,
                1.0,
            )
        )
        root = np.sqrt(np.maximum(y * y + z * z - 0.0955**2, 1e-8))
        target = np.column_stack(
            (
                np.arctan2(z, y) + np.arctan2(root, SIDE * 0.0955),
                np.arctan2(-x, root) - knee / 2.0,
                knee,
            )
        ).ravel()
        return np.clip(
            target,
            self.joint_ranges[:, 0] + 0.02,
            self.joint_ranges[:, 1] - 0.02,
        )

    def _has_fallen(self) -> bool:
        upright = self.data.xmat[self.base_body_id].reshape(3, 3)[2, 2]
        if upright < math.cos(float(self.simulation["fall_tilt_radians"])):
            return True
        for contact in self.data.contact:
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if (
                body1 == self.base_body_id and body2 == 0
            ) or (
                body2 == self.base_body_id and body1 == 0
            ):
                return True
        return False

    def step(self, command: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (20,) or not np.isfinite(command).all():
            raise ValueError("CPG command must be a finite 20-vector")
        command = np.clip(command, -1.0, 1.0)
        kp = float(self.simulation["kp"])
        kd = float(self.simulation["kd"])
        torque_limit = float(self.simulation["torque_limit"])
        physics_dt = float(self.simulation["physics_dt"])
        fallen = False

        for _ in range(self.substeps):
            self.phase = (
                self.phase
                + physics_dt * 2.0 * np.pi * (1.5 + 0.5 * command[:4])
            ) % (2.0 * np.pi)
            self.previous_joint_target = self._joint_target(command)
            torque = kp * (
                self.previous_joint_target - self.data.qpos[self.qpos_indices]
            ) - kd * self.data.qvel[self.qvel_indices]
            self.data.ctrl[self.actuator_indices] = np.clip(
                torque, -torque_limit, torque_limit
            )
            self._physics_step()
            mujoco.mj_forward(self.model, self.data)
            if self._has_fallen():
                fallen = True
                break

        self.steps += 1
        observation = self.observe()
        quaternion = observation[RAW_OBSERVATION_SPEC.base_quaternion_wxyz]
        rotation = quaternion_rotation_matrix_wxyz(quaternion)
        speed = float(np.clip(self._imu_velocity[0], -0.5, 0.6))
        upright = float(rotation[2, 2])
        reward = (
            2.0 * speed
            + 0.3 * np.exp(-((speed - 0.35) / 0.3) ** 2)
            + 0.1 * upright
            - 0.015 * np.sum((command - self.previous_command) ** 2)
            - 0.03 * np.sum(self.data.sensor("cpg_imu_gyro").data ** 2)
        )
        w, x, y, z = quaternion
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        reward -= 0.5 * yaw * yaw + 0.5 * self._imu_odometry[1] ** 2
        if fallen:
            reward -= 5.0
        self.total_reward += float(reward)
        self.previous_command = command.copy()

        self.max_x = max(self.max_x, float(self.data.qpos[0]))
        feet_position = self.data.geom_xpos[self.foot_geom_ids]
        landing = (
            np.all(feet_position[:, 0] > float(self.terrain["stair_start"]) + 1.25)
            and np.all(np.abs(feet_position[:, 1]) < 0.9)
            and np.count_nonzero(
                np.abs(
                    feet_position[:, 2]
                    - 0.022
                    - int(self.terrain["step_count"])
                    * float(self.terrain["step_height"])
                )
                < 0.045
            )
            >= 2
        )
        self.hold = self.hold + 1 if landing and not fallen else 0
        self.success = self.success or self.hold >= 15
        timeout = self.steps >= int(self.simulation["episode_steps"])
        info = {
            "success": bool(self.success),
            "fall": bool(fallen),
            "max_x": self.max_x,
            "x": float(self.data.qpos[0]),
            "y": float(self.data.qpos[1]),
            "imu_odometry": self._imu_odometry.tolist(),
            "score": float(self.total_reward),
            "steps": self.steps,
            "observation_abi": RAW_OBSERVATION_ABI,
        }
        return observation, float(reward), fallen, timeout, info
