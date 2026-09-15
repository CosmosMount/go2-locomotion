"""Minimal Isaac Lab source-domain Go2 environment for MF safety training.

Import this module only after ``isaaclab.app.AppLauncher`` has started.
"""

from __future__ import annotations

import numpy as np
import torch

from common.isaac import IsaacVectorizedEnvironmentBase, require_go2_usd
from common.observation import GO2_JOINT_NAMES
from common.proprio import ProprioceptiveVelocityEstimator


class SourceGo2Environment(IsaacVectorizedEnvironmentBase):
    """Synchronous vector source domain with canonical raw46 transitions."""

    def __init__(self, backend, config: dict):
        self.backend = backend
        self.config = config
        self.num_envs = int(config["num_envs"])
        self.device = torch.device(config["device"])
        self.robot = backend.scene["robot"]
        indices, names = self.robot.find_joints(
            list(GO2_JOINT_NAMES), preserve_order=True
        )
        if tuple(names) != GO2_JOINT_NAMES:
            raise RuntimeError(f"Isaac joint contract mismatch: {names}")
        self.joint_indices = torch.as_tensor(indices, device=self.device)
        self.imu = backend.scene["imu"]
        self.default_target = torch.tensor(
            config["default_joint_position"],
            dtype=torch.float32,
            device=self.device,
        ).expand(self.num_envs, -1)
        self.action_scale = torch.tensor(
            config["action_scale"], dtype=torch.float32, device=self.device
        )
        self.previous_target = self.default_target.clone()
        self.previous_reward_action = torch.zeros(
            self.num_envs, 12, device=self.device
        )
        self.previous_quaternion = None
        self.estimator = ProprioceptiveVelocityEstimator(
            self.num_envs, self.device, dt=float(config["control_dt"])
        )
        self.tilt_frames = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.height_frames = torch.zeros_like(self.tilt_frames)
        self.physics_steps_per_action = round(
            float(config["control_dt"]) / float(config["physics_dt"])
        )
        if self.physics_steps_per_action < 1:
            raise ValueError("control_dt must be at least physics_dt")

    def _sensor(self, value: torch.Tensor, magnitude: float) -> torch.Tensor:
        value = value.clone()
        if self.config["domain_randomization"] and magnitude:
            value += torch.empty_like(value).uniform_(-magnitude, magnitude)
        return value

    def _components(self):
        joint_q = self._sensor(
            self.robot.data.joint_pos[:, self.joint_indices], 0.01
        )
        joint_dq = self._sensor(
            self.robot.data.joint_vel[:, self.joint_indices], 1.5
        )
        gyro = self._sensor(self.imu.data.ang_vel_b, 0.2)
        acceleration = self._sensor(self.imu.data.lin_acc_b, 0.1)
        quaternion = self.imu.data.quat_w.clone()
        velocity = self.estimator.update(
            joint_q, joint_dq, gyro, quaternion, acceleration
        )
        return joint_q, joint_dq, gyro, velocity, quaternion

    def _observation(self) -> torch.Tensor:
        components = self._components()
        observation, self.previous_quaternion = self.build_canonical_raw_observation(
            *components,
            self.previous_target,
            previous_quaternion=self.previous_quaternion,
        )
        return observation

    def _failure(self, quaternion: torch.Tensor):
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        w, x, y, z = quaternion.unbind(-1)
        roll = torch.atan2(
            2 * (w * x + y * z), 1 - 2 * (x.square() + y.square())
        )
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
        tilted = (roll.abs() > float(self.config["failure_angle"])) | (
            pitch.abs() > float(self.config["failure_angle"])
        )
        low = self.robot.data.root_pos_w[:, 2] < float(
            self.config["failure_min_base_height"]
        )
        self.tilt_frames = torch.where(
            tilted, self.tilt_frames + 1, torch.zeros_like(self.tilt_frames)
        )
        self.height_frames = torch.where(
            low, self.height_frames + 1, torch.zeros_like(self.height_frames)
        )
        threshold = int(self.config["failure_consecutive_frames"])
        return (self.tilt_frames >= threshold) | (self.height_frames >= threshold)

    def _reward(self, action: torch.Tensor) -> torch.Tensor:
        velocity = self.robot.data.root_lin_vel_b
        angular = self.robot.data.root_ang_vel_b
        joint_q = self.robot.data.joint_pos[:, self.joint_indices]
        target = float(self.config["target_velocity"])
        velocity_error = (target - velocity[:, 0]).square() + velocity[:, 1].square()
        tracking_velocity = torch.exp(-velocity_error / 0.25)
        tracking_yaw = torch.exp(-angular[:, 2].square() / 0.25)
        base_height_error = (self.robot.data.root_pos_w[:, 2] - 0.3).square()
        action_rate = (action - self.previous_reward_action).square().sum(-1)
        reward_default = action.new_tensor(
            (0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
             0.0, 1.0, -1.5, 0.0, 1.0, -1.5)
        )
        similar_to_default = (joint_q - reward_default).abs().sum(-1)
        reward = float(self.config["control_dt"]) * (
            tracking_velocity
            - 3.0 * velocity_error
            + 0.2 * tracking_yaw
            - velocity[:, 2].square()
            - 50.0 * base_height_error
            - 0.005 * action_rate
            - 0.1 * similar_to_default
        )
        self.previous_reward_action.copy_(action)
        return reward

    def _reset_wrapper_state(self, env_ids: torch.Tensor) -> torch.Tensor:
        self.previous_target[env_ids] = self.default_target[env_ids]
        self.previous_reward_action[env_ids] = 0
        self.tilt_frames[env_ids] = 0
        self.height_frames[env_ids] = 0
        self.estimator.reset(env_ids)
        joint_q = self.robot.data.joint_pos[env_ids][:, self.joint_indices]
        joint_dq = self.robot.data.joint_vel[env_ids][:, self.joint_indices]
        gyro = self.imu.data.ang_vel_b[env_ids]
        quaternion = self.imu.data.quat_w[env_ids]
        velocity = torch.zeros(len(env_ids), 3, device=self.device)
        observation, continuous = self.build_canonical_raw_observation(
            joint_q,
            joint_dq,
            gyro,
            velocity,
            quaternion,
            self.previous_target[env_ids],
        )
        if self.previous_quaternion is None:
            self.previous_quaternion = torch.zeros(
                self.num_envs, 4, device=self.device
            )
        self.previous_quaternion[env_ids] = continuous
        return observation

    def reset(self) -> np.ndarray:
        self.backend.reset()
        self.previous_target.copy_(self.default_target)
        self.previous_reward_action.zero_()
        self.previous_quaternion = None
        self.tilt_frames.zero_()
        self.height_frames.zero_()
        self.estimator.reset()
        return self._observation().detach().cpu().numpy()

    def step(self, actions):
        action = torch.as_tensor(
            actions, dtype=torch.float32, device=self.device
        )
        if action.shape != (self.num_envs, 12) or not torch.isfinite(action).all():
            raise ValueError("actions must be finite [num_envs, 12]")
        default = self.default_target
        raw_target = default + self.action_scale * action.clamp(-1, 1)
        max_delta = float(self.config["max_target_rate"]) * float(
            self.config["control_dt"]
        )
        target = torch.maximum(
            torch.minimum(raw_target, self.previous_target + max_delta),
            self.previous_target - max_delta,
        )
        applied = ((target - default) / self.action_scale).clamp(-1, 1)
        self.previous_target.copy_(target)

        reset_index = self.backend._reset_idx
        scene_update = self.backend.scene.update
        failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        def tracked_scene_update(dt):
            scene_update(dt)
            if float(dt) > 0.0:
                failure.logical_or_(self._failure(self.robot.data.root_quat_w))

        self.backend._reset_idx = lambda env_ids: None
        self.backend.scene.update = tracked_scene_update
        try:
            _, _, backend_terminated, truncated, _ = self.backend.step(applied)
        finally:
            self.backend._reset_idx = reset_index
            self.backend.scene.update = scene_update
        backend_terminated = backend_terminated.clone()
        truncated = truncated.clone()
        transition_observation = self._observation()
        terminated = backend_terminated | failure
        reward = self._reward(action.clamp(-1, 1))
        next_observation = transition_observation.clone()
        done = terminated | truncated
        env_ids = done.nonzero(as_tuple=False).flatten()
        if len(env_ids):
            reset_index(env_ids)
            self.backend.scene.write_data_to_sim()
            # IMU acceleration is a finite difference divided by the scene dt;
            # zero here would poison the first post-reset observation.
            self.backend.scene.update(float(self.config["physics_dt"]))
            next_observation[env_ids] = self._reset_wrapper_state(env_ids)
        return (
            next_observation.detach().cpu().numpy(),
            reward.detach().cpu().numpy(),
            failure.float().detach().cpu().numpy(),
            terminated.detach().cpu().numpy(),
            truncated.detach().cpu().numpy(),
            {
                "transition_observation": transition_observation.detach().cpu().numpy(),
                "forward_velocity": self.robot.data.root_lin_vel_b[:, 0]
                .detach()
                .cpu()
                .numpy(),
            },
        )

    def close(self):
        self.backend.close()


def make_source_environment(config: dict) -> SourceGo2Environment:
    """Create the built-in Go2 asset after Isaac's application is running."""

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.sensors import ImuCfg
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )

    cfg = UnitreeGo2FlatEnvCfg()
    cfg.scene.robot.spawn.usd_path = str(require_go2_usd())
    cfg.scene.num_envs = int(config["num_envs"])
    cfg.scene.env_spacing = 2.5
    cfg.sim.device = str(config["device"])
    cfg.sim.dt = float(config["physics_dt"])
    cfg.decimation = round(float(config["control_dt"]) / float(config["physics_dt"]))
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = float(config["episode_steps"]) * float(config["control_dt"])
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.27)
    cfg.scene.robot.init_state.joint_pos = {
        ".*_hip_joint": 0.0,
        ".*_thigh_joint": 0.9,
        ".*_calf_joint": -1.8,
    }
    cfg.actions.joint_pos.joint_names = list(GO2_JOINT_NAMES)
    cfg.actions.joint_pos.preserve_order = True
    cfg.actions.joint_pos.use_default_offset = False
    cfg.actions.joint_pos.scale = {
        name: float(value)
        for name, value in zip(GO2_JOINT_NAMES, config["action_scale"])
    }
    cfg.actions.joint_pos.offset = {
        name: value
        for name, value in zip(
            GO2_JOINT_NAMES, config["default_joint_position"]
        )
    }
    cfg.scene.imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=float(config["physics_dt"]),
        history_length=0,
        debug_vis=False,
    )
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    target_velocity = float(config["target_velocity"])
    cfg.commands.base_velocity.ranges.lin_vel_x = (target_velocity, target_velocity)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.terminations.base_contact = None
    cfg.events.reset_base.params["velocity_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    if not config["domain_randomization"]:
        cfg.events.physics_material = None
        cfg.events.add_base_mass = None
        cfg.events.base_com = None
        cfg.events.base_external_force_torque = None
        cfg.events.push_robot = None
    else:
        cfg.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        cfg.events.base_com = None
        # Go2's upstream config explicitly disables this inherited event. Recreate
        # it so source-domain randomization remains effective.
        push_interval = tuple(float(value) for value in config["push_interval_s"])
        push_velocity = tuple(float(value) for value in config["push_velocity_xy"])
        cfg.events.push_robot = EventTerm(
            func=velocity_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=push_interval,
            params={
                "velocity_range": {
                    "x": push_velocity,
                    "y": push_velocity,
                }
            },
        )
    backend = ManagerBasedRLEnv(cfg=cfg)
    return SourceGo2Environment(backend, config)
