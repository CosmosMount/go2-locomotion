"""Flat-terrain Go2 locomotion task built on Isaac Lab primitives."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
from isaaclab.terrains import TerrainImporterCfg

from common.isaac import IsaacVectorizedEnvironmentBase, require_go2_usd
from common.observation import ppo_observation
from common.proprio import ProprioceptiveVelocityEstimator

from .config import LocomotionConfig


class Go2LocomotionEnv(IsaacVectorizedEnvironmentBase, gym.Env):
    """Vectorized Isaac Lab environment containing only the locomotion task."""

    is_vector_env = True

    def __init__(self, config: LocomotionConfig):
        config.validate(check_assets=True)
        self.config = config
        self._closed = False
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=config.friction,
            dynamic_friction=config.friction,
            restitution=config.restitution,
        )
        sim_cfg = sim_utils.SimulationCfg(
            device=config.device,
            dt=config.physics_dt,
            render_interval=config.decimation,
            physics_material=material,
        )
        sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim = SimulationContext(sim_cfg)
        self.device = self.sim.device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        with use_stage(self.sim.get_initial_stage()):
            self.scene = InteractiveScene(
                InteractiveSceneCfg(
                    num_envs=config.num_envs,
                    env_spacing=config.env_spacing,
                    replicate_physics=True,
                )
            )
            self._setup_scene(material)
            attach_stage_to_usd_context()
        with use_stage(self.sim.get_initial_stage()):
            self.sim.reset()
        self.scene.update(dt=config.physics_dt)

        self._joint_ids, joint_names = self._robot.find_joints(
            list(config.joint_names), preserve_order=True
        )
        self._base_ids, _ = self._contact_sensor.find_bodies("base")
        foot_names = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]
        self._foot_contact_ids, _ = self._contact_sensor.find_bodies(
            foot_names, preserve_order=True
        )
        self._foot_body_ids, _ = self._robot.find_bodies(
            foot_names, preserve_order=True
        )
        self._undesired_contact_ids, _ = self._contact_sensor.find_bodies(
            ".*_(hip|thigh|calf)"
        )
        if (
            tuple(joint_names) != config.joint_names
            or len(self._base_ids) != 1
            or len(self._foot_contact_ids) != 4
            or len(self._foot_body_ids) != 4
            or len(self._undesired_contact_ids) != 12
        ):
            raise RuntimeError(
                "Go2 asset contract mismatch: "
                f"joints={joint_names}, base={self._base_ids}, "
                f"feet={self._foot_contact_ids}, undesired={self._undesired_contact_ids}"
            )

        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (config.observation_size,), np.float32
        )
        self.single_critic_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (config.critic_observation_size,), np.float32
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, config.num_envs
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, config.num_envs
        )

        shape = (config.num_envs, config.action_size)
        self._actions = torch.zeros(shape, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._joint_targets = torch.zeros_like(self._actions)
        self._commands = torch.zeros(config.num_envs, 3, device=self.device)
        self._default_joint_pos = torch.as_tensor(
            config.default_joint_pos, device=self.device
        ).unsqueeze(0)
        self._previous_quaternion = self._robot.data.root_quat_w.clone()
        self._velocity_estimator = ProprioceptiveVelocityEstimator(
            config.num_envs, self.device, dt=config.step_dt
        )
        torch.testing.assert_close(
            self._robot.data.default_joint_pos[0, self._joint_ids],
            self._default_joint_pos[0],
        )
        self._command_level = 0.1 if config.command_curriculum else 1.0
        self._next_push = torch.empty(config.num_envs, device=self.device).uniform_(
            *config.push_interval_s
        )
        self._training_steps = 0
        self._last_command_curriculum_step = -1
        self._sim_step = 0
        self.episode_length_buf = torch.zeros(
            config.num_envs, dtype=torch.long, device=self.device
        )
        self.reset_terminated = torch.zeros(
            config.num_envs, dtype=torch.bool, device=self.device
        )
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.extras: dict = {}
        self._episode_sums = {
            name: torch.zeros(config.num_envs, device=self.device)
            for name in config.reward_weights
        }
        if config.domain_randomization:
            self._randomize_dynamics()
        print(
            f"Go2LocomotionEnv: envs={config.num_envs}, "
            f"physics={1 / config.physics_dt:.0f}Hz, policy={1 / config.step_dt:.0f}Hz, "
            f"device={self.device}",
            flush=True,
        )

    @property
    def num_envs(self) -> int:
        return self.config.num_envs

    @property
    def max_episode_length(self) -> int:
        return math.ceil(self.config.episode_length_s / self.config.step_dt)

    def _setup_scene(self, material) -> None:
        cfg = self.config
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(require_go2_usd()),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=cfg.initial_base_pos,
                joint_pos=dict(zip(cfg.joint_names, cfg.default_joint_pos)),
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=cfg.soft_joint_pos_limit_factor,
            actuators={
                "legs": DCMotorCfg(
                    joint_names_expr=list(cfg.joint_names),
                    effort_limit=cfg.effort_limit,
                    effort_limit_sim=cfg.effort_limit,
                    saturation_effort=cfg.effort_limit,
                    velocity_limit=cfg.velocity_limit,
                    velocity_limit_sim=cfg.velocity_limit,
                    stiffness=cfg.kp,
                    damping=cfg.kd,
                    armature=0.0,
                    friction=0.0,
                    dynamic_friction=0.0,
                    viscous_friction=0.0,
                )
            },
        )
        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*",
                update_period=cfg.physics_dt,
                history_length=cfg.contact_history_length,
                track_air_time=True,
            )
        )
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._imu = Imu(
            ImuCfg(
                prim_path="/World/envs/env_.*/Robot/base",
                update_period=cfg.physics_dt,
                history_length=0,
                debug_vis=False,
            )
        )
        self.scene.sensors["imu"] = self._imu
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            terrain_generator=None,
            collision_group=-1,
            physics_material=material,
            num_envs=cfg.num_envs,
            env_spacing=cfg.env_spacing,
            debug_vis=False,
        )
        self._terrain = terrain_cfg.class_type(terrain_cfg)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[terrain_cfg.prim_path])
        light = sim_utils.DomeLightCfg(intensity=750.0, color=(0.8, 0.8, 0.8))
        light.func("/World/light", light)

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)
        self.reset_terminated.zero_()
        self.reset_time_outs.zero_()
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.config.physics_dt)
        raw = self._reset_raw_observation(env_ids)
        return self._observations(
            noisy_policy=True, raw=raw, env_ids=env_ids
        ), self.extras

    def step(self, actions):
        expected_shape = (self.num_envs, self.config.action_size)
        actions = actions.to(self.device)
        if actions.shape != expected_shape:
            raise ValueError(f"expected actions {expected_shape}, got {actions.shape}")
        if not torch.isfinite(actions).all():
            raise RuntimeError("policy produced non-finite actions")
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))
        self._joint_targets.copy_(
            self._default_joint_pos + self.config.action_scale * self._actions
        )

        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.config.decimation):
            self._robot.set_joint_position_target(
                self._joint_targets, joint_ids=self._joint_ids
            )
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self._sim_step += 1
            if rendering and self._sim_step % self.config.decimation == 0:
                self.sim.render()
            self.scene.update(dt=self.config.physics_dt)

        self.episode_length_buf += 1
        self._training_steps += 1
        self._apply_pushes()
        self.reset_terminated, self.reset_time_outs = self._get_dones()
        reward = self._get_reward()

        command_period = max(
            1, round(self.config.command_resampling_s / self.config.step_dt)
        )
        command_ids = (
            self.episode_length_buf % command_period == 0
        ).nonzero().squeeze(-1)
        self._resample_commands(command_ids)
        raw = self._raw_observation()
        final_observations = self._observations(noisy_policy=False, raw=raw)
        next_observations = self._observations(noisy_policy=True, raw=raw)
        self.extras["final_observation"] = final_observations["policy"].clone()
        self.extras["final_critic_observation"] = final_observations["critic"].clone()
        self.extras["metrics"]["termination_fraction"] = (
            self.reset_terminated.float().mean()
        )
        self.extras["metrics"]["timeout_fraction"] = self.reset_time_outs.float().mean()
        self.extras["metrics"]["episode_age_s"] = (
            self.episode_length_buf.float().mean() * self.config.step_dt
        )
        reset_ids = (self.reset_terminated | self.reset_time_outs).nonzero().squeeze(-1)
        if len(reset_ids):
            self._reset_idx(reset_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=self.config.physics_dt)
            reset_raw = self._reset_raw_observation(reset_ids)
            reset_observations = self._observations(
                noisy_policy=True, raw=reset_raw, env_ids=reset_ids
            )
            for name in next_observations:
                next_observations[name][reset_ids] = reset_observations[name]
        return (
            next_observations,
            reward,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _raw_observation(self):
        estimated_velocity = self._velocity_estimator.update(
            self._robot.data.joint_pos[:, self._joint_ids],
            self._robot.data.joint_vel[:, self._joint_ids],
            self._robot.data.root_ang_vel_b,
            self._robot.data.root_quat_w,
            self._imu.data.lin_acc_b,
        )
        raw, next_quaternion = self.build_canonical_raw_observation(
            joint_position=self._robot.data.joint_pos[:, self._joint_ids],
            joint_velocity=self._robot.data.joint_vel[:, self._joint_ids],
            angular_velocity=self._robot.data.root_ang_vel_b,
            estimated_body_velocity=estimated_velocity,
            base_quaternion_wxyz=self._robot.data.root_quat_w,
            previous_joint_target=self._joint_targets,
            previous_quaternion=self._previous_quaternion,
        )
        if raw.shape != (self.num_envs, self.config.raw_observation_size):
            raise RuntimeError(f"common observation API returned invalid shape {raw.shape}")
        self._previous_quaternion.copy_(next_quaternion)
        return raw

    def _reset_raw_observation(self, env_ids):
        raw, next_quaternion = self.build_canonical_raw_observation(
            joint_position=self._robot.data.joint_pos[env_ids][:, self._joint_ids],
            joint_velocity=self._robot.data.joint_vel[env_ids][:, self._joint_ids],
            angular_velocity=self._robot.data.root_ang_vel_b[env_ids],
            estimated_body_velocity=torch.zeros(
                (len(env_ids), 3), device=self.device
            ),
            base_quaternion_wxyz=self._robot.data.root_quat_w[env_ids],
            previous_joint_target=self._joint_targets[env_ids],
            previous_quaternion=self._previous_quaternion[env_ids],
        )
        self._previous_quaternion[env_ids] = next_quaternion
        return raw

    def _policy_observation(self, raw, *, noisy: bool):
        cfg = self.config
        observation = ppo_observation(
            raw,
            self._commands,
            default_joint_position=cfg.default_joint_pos,
            action_scale=(cfg.action_scale,) * cfg.action_size,
            angular_velocity_scale=cfg.angular_velocity_scale,
            command_scale=cfg.command_scale,
            joint_velocity_scale=cfg.joint_velocity_scale,
            clip=cfg.observation_clip,
        )
        if noisy and cfg.observation_noise:
            observation = observation.clone()
            observation[:, 0:3] += torch.empty_like(observation[:, 0:3]).uniform_(
                -0.04, 0.04
            )
            observation[:, 3:6] += torch.empty_like(observation[:, 3:6]).uniform_(
                -0.05, 0.05
            )
            observation[:, 9:21] += torch.empty_like(observation[:, 9:21]).uniform_(
                -0.01, 0.01
            )
            observation[:, 21:33] += torch.empty_like(
                observation[:, 21:33]
            ).uniform_(-0.075, 0.075)
            observation.clamp_(-cfg.observation_clip, cfg.observation_clip)
        if observation.shape != (raw.shape[0], cfg.observation_size):
            raise RuntimeError(f"common PPO adapter returned invalid shape {observation.shape}")
        return observation

    def _observations(self, *, noisy_policy: bool, raw=None, env_ids=None):
        if raw is None:
            raw = self._raw_observation()
        applied_torque = self._robot.data.applied_torque[:, self._joint_ids]
        if env_ids is not None:
            applied_torque = applied_torque[env_ids]
        clean_policy = self._policy_observation(raw, noisy=False)
        critic = torch.cat(
            (
                raw[..., RAW_OBSERVATION_SPEC.estimated_body_velocity],
                clean_policy[:, :33],
                self.config.torque_scale
                * applied_torque,
                clean_policy[:, 33:],
            ),
            dim=-1,
        ).clamp(-self.config.observation_clip, self.config.observation_clip)
        if critic.shape != (raw.shape[0], self.config.critic_observation_size):
            raise RuntimeError(f"invalid critic observation shape {critic.shape}")
        policy = self._policy_observation(raw, noisy=True) if noisy_policy else clean_policy
        return {"policy": policy, "critic": critic}

    def _get_reward(self):
        cfg = self.config
        first_contact = self._contact_sensor.compute_first_contact(cfg.step_dt)[
            :, self._foot_contact_ids
        ]
        force_history = self._contact_sensor.data.net_forces_w_history
        undesired_contact = (
            torch.linalg.vector_norm(
                force_history[:, :, self._undesired_contact_ids], dim=-1
            ).amax(dim=1)
            > cfg.contact_force_threshold
        )
        linear_velocity = self._robot.data.root_lin_vel_b
        angular_velocity = self._robot.data.root_ang_vel_b
        gravity = self._robot.data.projected_gravity_b
        joint_velocity = self._robot.data.joint_vel[:, self._joint_ids]
        joint_position = self._robot.data.joint_pos[:, self._joint_ids]
        joint_torque = self._robot.data.applied_torque[:, self._joint_ids]
        last_air = self._contact_sensor.data.last_air_time[:, self._foot_contact_ids]
        last_contact = self._contact_sensor.data.last_contact_time[:, self._foot_contact_ids]
        foot_contact = (
            force_history[:, :, self._foot_contact_ids].norm(dim=-1).amax(dim=1) > 1.0
        )
        terms = {
            "track_linear_velocity": torch.exp(
                -torch.sum((self._commands[:, :2] - linear_velocity[:, :2]).square(), dim=1)
                / cfg.linear_tracking_std**2
            ),
            "track_yaw_velocity": torch.exp(
                -(self._commands[:, 2] - angular_velocity[:, 2]).square()
                / cfg.yaw_tracking_std**2
            ),
            "vertical_velocity": linear_velocity[:, 2].square(),
            "roll_pitch_velocity": torch.sum(angular_velocity[:, :2].square(), dim=1),
            "flat_orientation": torch.sum(gravity[:, :2].square(), dim=1),
            "joint_torque": torch.sum(joint_torque.square(), dim=1),
            "joint_acceleration": torch.sum(
                self._robot.data.joint_acc[:, self._joint_ids].square(), dim=1
            ),
            "action_rate": torch.sum(
                (self._actions - self._previous_actions).square(), dim=1
            ),
            "feet_air_time": torch.sum(
                (last_air - cfg.feet_air_time_target) * first_contact, dim=1
            )
            * (
                torch.linalg.vector_norm(self._commands[:, :2], dim=1)
                > cfg.moving_command_threshold
            ),
            "undesired_contacts": torch.sum(undesired_contact, dim=1),
            "joint_velocity": joint_velocity.square().sum(dim=-1),
            "joint_position": (joint_position - self._default_joint_pos).norm(dim=-1)
            * torch.where(
                (self._commands.norm(dim=-1) > 0)
                | (linear_velocity[:, :2].norm(dim=-1) > 0.3),
                1.0,
                5.0,
            ),
            "joint_limits": (
                (
                    self._robot.data.soft_joint_pos_limits[:, self._joint_ids, 0]
                    - joint_position
                ).clamp_min(0)
                + (
                    joint_position
                    - self._robot.data.soft_joint_pos_limits[:, self._joint_ids, 1]
                ).clamp_min(0)
            ).sum(dim=-1),
            "energy": (joint_velocity * joint_torque).abs().sum(dim=-1),
            "air_time_variance": last_air.clamp(max=0.5).var(dim=-1)
            + last_contact.clamp(max=0.5).var(dim=-1),
            "feet_slide": (
                self._robot.data.body_lin_vel_w[:, self._foot_body_ids, :2].norm(dim=-1)
                * foot_contact
            ).sum(dim=-1),
        }
        weighted = {
            name: value * cfg.reward_weights[name] * cfg.step_dt
            for name, value in terms.items()
        }
        reward = torch.stack(tuple(weighted.values())).sum(dim=0)
        for name, value in weighted.items():
            self._episode_sums[name] += value
        self.extras["metrics"] = {
            f"reward/{name}": value.mean() for name, value in weighted.items()
        }
        self.extras["metrics"]["linear_error"] = torch.linalg.vector_norm(
            self._commands[:, :2] - linear_velocity[:, :2], dim=-1
        ).mean()
        self.extras["metrics"]["yaw_error"] = (
            self._commands[:, 2] - angular_velocity[:, 2]
        ).abs().mean()
        self.extras["metrics"]["moving_fraction"] = (
            self._commands.norm(dim=-1) > cfg.moving_command_threshold
        ).float().mean()
        self.extras["metrics"]["command_level"] = reward.new_tensor(self._command_level)
        return reward

    def _get_dones(self):
        forces = self._contact_sensor.data.net_forces_w_history
        base_contact = (
            torch.linalg.vector_norm(forces[:, :, self._base_ids], dim=-1).amax(dim=1)
            > self.config.contact_force_threshold
        ).any(dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length
        tilted = -self._robot.data.projected_gravity_b[:, 2] < math.cos(
            self.config.bad_orientation_angle
        )
        return base_contact | tilted, time_out

    def _reset_idx(self, env_ids):
        self._update_command_curriculum(env_ids)
        if len(env_ids) and self._episode_sums:
            self.extras["log"] = {
                f"Episode_Reward/{name}": value[env_ids].mean().item()
                / self.config.episode_length_s
                for name, value in self._episode_sums.items()
            }
            for value in self._episode_sums.values():
                value[env_ids] = 0.0
        self.scene.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = self._default_joint_pos
        self._velocity_estimator.reset(env_ids)
        self._resample_commands(env_ids)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        root_state[:, :2] += torch.empty_like(root_state[:, :2]).uniform_(
            *self.config.initial_xy
        )
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.config.initial_yaw
        )
        root_state[:, 3:7] = 0.0
        root_state[:, 3] = torch.cos(0.5 * yaw)
        root_state[:, 6] = torch.sin(0.5 * yaw)
        self._previous_quaternion[env_ids] = root_state[:, 3:7]
        joint_position = self._robot.data.default_joint_pos[env_ids]
        joint_velocity = torch.empty_like(joint_position).uniform_(
            *self.config.reset_joint_velocity
        )
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_position, joint_velocity, None, env_ids)

    def _resample_commands(self, env_ids):
        self._commands[env_ids, 0] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(*self.config.command_vx) * self._command_level
        self._commands[env_ids, 1] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(*self.config.command_vy) * self._command_level
        self._commands[env_ids, 2] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(*self.config.command_yaw)
        standing = (
            torch.rand(len(env_ids), device=self.device)
            < self.config.standing_fraction
        )
        self._commands[env_ids[standing]] = 0.0

    def _update_command_curriculum(self, env_ids):
        active = env_ids[self.episode_length_buf[env_ids] > 0]
        if not len(active) or not self.config.command_curriculum:
            return
        if self._training_steps - self._last_command_curriculum_step < self.max_episode_length:
            return
        score = (
            self._episode_sums["track_linear_velocity"][active].mean()
            / self.config.episode_length_s
        )
        if score > 0.8 * self.config.reward_weights["track_linear_velocity"]:
            self._command_level = min(1.0, self._command_level + 0.1)
        self._last_command_curriculum_step = self._training_steps

    def _apply_pushes(self):
        if not self.config.pushes:
            return
        self._next_push -= self.config.step_dt
        push_ids = (self._next_push <= 0).nonzero().squeeze(-1)
        if len(push_ids):
            velocity = self._robot.data.root_vel_w[push_ids].clone()
            velocity[:, :2].uniform_(*self.config.push_velocity)
            self._robot.write_root_velocity_to_sim(velocity, push_ids)
            self._next_push[push_ids] = torch.empty(
                len(push_ids), device=self.device
            ).uniform_(*self.config.push_interval_s)

    def _randomize_dynamics(self):
        view = self._robot.root_physx_view
        env_ids = torch.arange(self.num_envs, device="cpu")
        materials = view.get_material_properties()
        buckets = torch.empty(64, 3)
        buckets[:, :2].uniform_(0.3, 1.2)
        buckets[:, 2].uniform_(0.0, 0.15)
        materials[:] = buckets[torch.randint(64, materials.shape[:2])]
        view.set_material_properties(materials, env_ids)
        base_ids, _ = self._robot.find_bodies("base")
        masses = view.get_masses()
        original = masses[:, base_ids].clone()
        masses[:, base_ids] = (
            original + torch.empty_like(original).uniform_(-1.0, 3.0)
        ).clamp_min(0.01)
        inertias = view.get_inertias()
        inertias[:, base_ids] *= (masses[:, base_ids] / original).unsqueeze(-1)
        view.set_masses(masses, env_ids)
        view.set_inertias(inertias, env_ids)

    def close(self):
        if not self._closed:
            del self.scene
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            self._closed = True
