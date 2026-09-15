"""Synchronous target-domain Go2 MuJoCo environment."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from common.action import ActionMapper
from common.mujoco import (
    GO2_JOINT_LOWER_LIMIT,
    GO2_JOINT_UPPER_LIMIT,
    MujocoObservationBuilder,
    go2_indices,
    load_model,
    quaternion_rotation_matrix_wxyz,
)
from common.observation import GO2_JOINT_NAMES


REWARD_DEFAULT_JOINT_POSITION = np.asarray([
    0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
    0.0, 1.0, -1.5, 0.0, 1.0, -1.5,
], np.float32)


def _quaternion_to_roll_pitch(quaternion):
    w, x, y, z = np.asarray(quaternion, np.float64)
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(float(np.clip(2 * (w*y - z*x), -1.0, 1.0)))
    return roll, pitch


def _task_reward(world_velocity, quaternion, angular_velocity, joint_position,
                 action, previous_action, target_velocity, base_height):
    body_velocity = quaternion_rotation_matrix_wxyz(quaternion).T @ np.asarray(
        world_velocity, np.float64
    )
    linear_error = (target_velocity - body_velocity[0]) ** 2 + body_velocity[1] ** 2
    angular_error = float(angular_velocity[2]) ** 2
    terms = {
        "tracking_lin_vel": 0.02 * math.exp(-linear_error / 0.25),
        "velocity_error": 0.02 * -3.0 * linear_error,
        "tracking_ang_vel": 0.02 * 0.2 * math.exp(-angular_error / 0.25),
        "lin_vel_z": 0.02 * -1.0 * body_velocity[2] ** 2,
        "base_height": 0.02 * -50.0 * (float(base_height) - 0.3) ** 2,
        "action_rate": 0.02 * -0.005 * float(np.square(previous_action - action).sum()),
        "similar_to_default": 0.02 * -0.1 * float(
            np.abs(joint_position - REWARD_DEFAULT_JOINT_POSITION).sum()
        ),
    }
    return float(sum(terms.values())), terms


class MujocoRecoveryEnv:
    """One-environment adapter returning common canonical raw46 observations."""

    def __init__(self, simulation_config: dict, *, seed: int):
        import mujoco

        self.mujoco = mujoco
        self.control_dt = float(simulation_config["control_dt"])
        self.physics_dt = float(simulation_config["physics_dt"])
        ratio = self.control_dt / self.physics_dt
        if not np.isclose(ratio, round(ratio)) or round(ratio) < 1:
            raise ValueError("control_dt must be a positive integer multiple of physics_dt")
        self.physics_steps_per_action = int(round(ratio))
        if not (np.isclose(self.control_dt, 0.02) and np.isclose(self.physics_dt, 0.002)):
            raise ValueError("onrobot.recoveryrl requires 20 ms control and 2 ms physics")

        project_root = Path(__file__).resolve().parents[2]
        assets_root = project_root / "assets"
        scene_path = (project_root / simulation_config["scene"]).resolve()
        if assets_root not in scene_path.parents:
            raise ValueError("MuJoCo scene must be inside the shared assets directory")
        self.model, self.data = load_model(scene_path, physics_dt=self.physics_dt)
        self.target_velocity = float(simulation_config["target_velocity"])
        self.episode_steps = int(simulation_config["episode_steps"])
        self.fall_angle = float(simulation_config["fall_angle_threshold"])
        self.fall_height = float(simulation_config["fall_min_base_height"])
        self.fall_frames = int(simulation_config["fall_consecutive_physics_frames"])
        self.kp = float(simulation_config["policy_kp"])
        self.kd = float(simulation_config["policy_kd"])
        self.effort_limit = float(simulation_config["effort_limit"])
        self.default_joint_position = np.asarray(
            simulation_config["default_joint_position"], np.float32
        )
        self.action_scale = np.asarray(simulation_config["action_scale"], np.float32)
        self.qpos_addresses, self.qvel_addresses, self.actuator_ids = go2_indices(self.model)
        self.base_body_id = self.model.body("base_link").id

        if simulation_config["joint_constraint_profile"] != "fr_recovery":
            raise ValueError("only the approved fr_recovery joint profile is supported")
        joint_ids = [self.model.joint(name).id for name in GO2_JOINT_NAMES]
        self.model.jnt_solref[joint_ids] = [0.01, 1.0]
        self.model.jnt_solimp[joint_ids] = [0.99, 0.999, 0.001, 0.5, 2.0]
        self.mapper = ActionMapper(
            self.default_joint_position,
            self.action_scale,
            control_dt=self.control_dt,
            max_target_rate=float(simulation_config["max_target_rate"]),
            joint_lower_limit=GO2_JOINT_LOWER_LIMIT,
            joint_upper_limit=GO2_JOINT_UPPER_LIMIT,
        )
        self.builder = MujocoObservationBuilder(
            self.default_joint_position, dt=self.control_dt
        )
        self.rng = np.random.default_rng(seed)
        self.total_physics_steps = 0
        self.reset(seed=seed)

    def _sensors(self):
        return (
            self.data.qpos[self.qpos_addresses].copy(),
            self.data.qvel[self.qvel_addresses].copy(),
            self.data.sensor("recovery_imu_gyro").data.copy(),
            self.data.sensor("recovery_imu_quat").data.copy(),
            self.data.sensor("recovery_imu_acc").data.copy(),
        )

    def reset(self, *, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, 0.27]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_addresses] = self.default_joint_position
        self.mujoco.mj_forward(self.model, self.data)
        self.mapper.reset()
        self.builder.reset()
        self.previous_action = np.zeros(12, np.float32)
        self.steps = self.tilt_frames = self.height_frames = 0
        self.episode_return = 0.0
        joint_position, joint_velocity, gyro, quaternion, _ = self._sensors()
        return self.builder.initial(joint_position, joint_velocity, gyro, quaternion)

    def step(self, normalized_action):
        action = np.asarray(normalized_action, np.float32)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("action must be finite with shape (12,)")
        applied_action, target = self.mapper.apply(action)
        self.builder.set_previous_joint_target(target)
        failure = False
        for _ in range(self.physics_steps_per_action):
            torque = self.kp * (target - self.data.qpos[self.qpos_addresses]) \
                     - self.kd * self.data.qvel[self.qvel_addresses]
            self.data.ctrl[self.actuator_ids] = np.clip(
                torque, -self.effort_limit, self.effort_limit
            )
            self.mujoco.mj_step(self.model, self.data)
            self.mujoco.mj_forward(self.model, self.data)
            self.total_physics_steps += 1
            _, _, _, quaternion, _ = self._sensors()
            roll, pitch = _quaternion_to_roll_pitch(quaternion)
            tilted = abs(roll) > self.fall_angle or abs(pitch) > self.fall_angle
            low = float(self.data.xpos[self.base_body_id, 2]) < self.fall_height
            self.tilt_frames = self.tilt_frames + 1 if tilted else 0
            self.height_frames = self.height_frames + 1 if low else 0
            failure |= self.tilt_frames >= self.fall_frames or self.height_frames >= self.fall_frames

        joint_position, joint_velocity, gyro, quaternion, accelerometer = self._sensors()
        transition_observation = self.builder.build(
            joint_position, joint_velocity, gyro, quaternion, accelerometer
        )
        reward, reward_terms = _task_reward(
            self.data.qvel[:3], quaternion, gyro, joint_position,
            applied_action, self.previous_action, self.target_velocity,
            self.data.xpos[self.base_body_id, 2],
        )
        self.previous_action = applied_action.copy()
        self.steps += 1
        self.episode_return += reward
        terminated = bool(failure)
        truncated = bool(self.steps >= self.episode_steps and not terminated)
        info = {
            "transition_observation": transition_observation.copy(),
            "failure": terminated,
            "forward_velocity": float(
                (quaternion_rotation_matrix_wxyz(quaternion).T @ self.data.qvel[:3])[0]
            ),
            "applied_action": applied_action.copy(),
            "joint_target": target.copy(),
            "reward_terms": reward_terms,
            "transition_physics_steps": self.physics_steps_per_action,
            "total_physics_steps": self.total_physics_steps,
        }
        if terminated or truncated:
            info.update(episode_return=self.episode_return, episode_length=self.steps)
            next_observation = self.reset()
        else:
            next_observation = transition_observation
        return next_observation, reward, terminated, truncated, info

    def close(self):
        pass
