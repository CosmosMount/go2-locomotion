"""Isaac Lab direct-workflow environment for plain PPO fall recovery."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from common.isaac import IsaacVectorizedEnvironmentBase, require_go2_usd
from common.observation import (
    GO2_JOINT_NAMES,
    RAW_OBSERVATION_SPEC,
    ppo_observation,
    projected_gravity,
)
from common.proprio import ProprioceptiveVelocityEstimator
from common.terrain import RECOVERY_TERRAIN_MIX, validate_terrain_mix

from .rewards import recovery_reward_terms, weighted_reward


@height_field_to_mesh
def rough_sloped_terrain(difficulty: float, cfg) -> np.ndarray:
    """Generate the FR-Net rough-slope category without legacy Isaac Gym."""

    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    if np.random.random() < 0.5:
        slope = -slope
    height_max = slope * cfg.size[0] / (2.0 * cfg.vertical_scale)
    x = (width / 2.0 - np.abs(width / 2.0 - np.arange(width))) / (width / 2.0)
    y = (length / 2.0 - np.abs(length / 2.0 - np.arange(length))) / (length / 2.0)
    heights = height_max * x[:, None] * y[None, :]
    platform = max(1, int(cfg.platform_width / cfg.horizontal_scale / 2.0))
    corner = heights[width // 2 - platform, length // 2 - platform]
    heights = np.clip(heights, min(0.0, corner), max(0.0, corner))
    noise_low = int(cfg.noise_range[0] / cfg.vertical_scale)
    noise_high = int(cfg.noise_range[1] / cfg.vertical_scale)
    noise = np.random.randint(noise_low, noise_high + 1, size=heights.shape)
    heights += noise
    heights[width // 2 - platform:width // 2 + platform,
            length // 2 - platform:length // 2 + platform] = corner
    return np.rint(heights).astype(np.int16)


@configclass
class HfRoughSlopedTerrainCfg(HfTerrainBaseCfg):
    """Pyramid slope with discretized surface roughness."""

    function = rough_sloped_terrain
    slope_range: tuple[float, float] = (0.0, 0.4)
    noise_range: tuple[float, float] = (-0.05, 0.05)
    platform_width: float = 2.0


@configclass
class RecoveryEventCfg:
    physics_material: EventTerm = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    base_mass: EventTerm = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )
    limb_mass: EventTerm = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_(hip|thigh|calf)"),
            "mass_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )
    base_com: EventTerm = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )


def _terrain_cfg(settings: dict) -> TerrainImporterCfg:
    validate_terrain_mix(RECOVERY_TERRAIN_MIX)
    proportions = {item.name: item.proportion for item in RECOVERY_TERRAIN_MIX}
    generator = TerrainGeneratorCfg(
        seed=None,
        curriculum=bool(settings["curriculum"]),
        size=tuple(settings["patch_size"]),
        border_width=float(settings["border_width"]),
        num_rows=int(settings["rows"]),
        num_cols=int(settings["columns"]),
        horizontal_scale=float(settings["horizontal_scale"]),
        vertical_scale=float(settings["vertical_scale"]),
        slope_threshold=float(settings["slope_threshold"]),
        use_cache=False,
        sub_terrains={
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=proportions["slope"], slope_range=tuple(settings["slope_range"]),
                platform_width=float(settings["platform_width"]), border_width=0.25,
            ),
            "rough_slope": HfRoughSlopedTerrainCfg(
                proportion=proportions["rough_slope"], slope_range=tuple(settings["slope_range"]),
                noise_range=tuple(settings["roughness_range"]),
                platform_width=float(settings["platform_width"]), border_width=0.25,
            ),
            "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=proportions["stairs_down"],
                step_height_range=tuple(settings["stair_height_range"]),
                step_width=float(settings["stair_width"]),
                platform_width=float(settings["platform_width"]), border_width=1.0,
            ),
            "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=proportions["stairs_up"],
                step_height_range=tuple(settings["stair_height_range"]),
                step_width=float(settings["stair_width"]),
                platform_width=float(settings["platform_width"]), border_width=1.0,
            ),
            "gap": terrain_gen.MeshGapTerrainCfg(
                proportion=proportions["gap"], gap_width_range=tuple(settings["gap_width_range"]),
                platform_width=float(settings["platform_width"]),
            ),
        },
    )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=int(settings["max_initial_level"]),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        debug_vis=False,
    )


def _robot_cfg(settings: dict) -> ArticulationCfg:
    defaults = dict(zip(GO2_JOINT_NAMES, settings["default_joint_position"], strict=True))
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(require_go2_usd()), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False,
                linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(settings["initial_height"])),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=defaults,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=float(settings.get("soft_joint_position_limit", 0.9)),
        actuators={
            "legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=float(settings["effort_limit"]),
                saturation_effort=float(settings["effort_limit"]),
                velocity_limit=float(settings["velocity_limit"]),
                stiffness=float(settings["stiffness"]),
                damping=float(settings["damping"]),
                friction=0.0,
            )
        },
    )


@configclass
class RecoveryEnvCfg(DirectRLEnvCfg):
    episode_length_s = 24.0
    decimation = 4
    action_space = 12
    observation_space = {"policy": 45, "critic": 260}
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    robot: ArticulationCfg = _robot_cfg(
        {
            "initial_height": 0.6,
            "default_joint_position": [0.0, 1.5, -2.4] * 4,
            "stiffness": 30.0, "damping": 0.8, "effort_limit": 23.5, "velocity_limit": 30.0,
        }
    )
    terrain: TerrainImporterCfg = _terrain_cfg(
        {
            "curriculum": True, "patch_size": [8.0, 8.0], "border_width": 5.0,
            "rows": 10, "columns": 10, "max_initial_level": 5,
            "horizontal_scale": 0.1, "vertical_scale": 0.005, "slope_threshold": 0.75,
            "slope_range": [0.0, 0.4], "roughness_range": [-0.05, 0.05],
            "stair_height_range": [0.05, 0.35], "stair_width": 0.31,
            "gap_width_range": [0.0, 1.0], "platform_width": 2.0,
        }
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3,
        update_period=0.005, track_air_time=True,
    )
    imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=0.005,
        history_length=0,
        debug_vis=False,
    )
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        mesh_prim_paths=["/World/ground"], debug_vis=False,
    )
    events: RecoveryEventCfg = RecoveryEventCfg()


def build_env_cfg(task_cfg: dict) -> RecoveryEnvCfg:
    """Translate the single YAML into Isaac Lab configuration objects."""

    env, terrain, randomization = task_cfg["environment"], task_cfg["terrain"], task_cfg["domain_randomization"]
    cfg = RecoveryEnvCfg()
    cfg.seed = int(task_cfg["seed"])
    cfg.episode_length_s = float(env["episode_length_s"])
    cfg.decimation = int(env["control_decimation"])
    cfg.action_space = int(env["action_dim"])
    cfg.observation_space = {"policy": int(env["actor_observation_dim"]), "critic": int(env["critic_observation_dim"])}
    cfg.sim = SimulationCfg(
        dt=float(env["physics_dt"]), render_interval=int(env["control_decimation"]), device=str(task_cfg["device"]),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    cfg.scene = InteractiveSceneCfg(num_envs=int(env["num_envs"]), env_spacing=4.0, replicate_physics=True)
    robot_settings = dict(task_cfg["robot"])
    robot_settings["soft_joint_position_limit"] = task_cfg["rewards"]["soft_joint_position_limit"]
    cfg.robot = _robot_cfg(robot_settings)
    cfg.terrain = _terrain_cfg(terrain)
    scanner = terrain["height_scanner"]
    cfg.contact_sensor.update_period = float(env["physics_dt"])
    cfg.imu.update_period = float(env["physics_dt"])
    cfg.height_scanner.pattern_cfg = patterns.GridPatternCfg(
        resolution=float(scanner["resolution"]), size=tuple(scanner["size"])
    )
    cfg.events.physics_material.params.update(
        static_friction_range=tuple(randomization["friction_range"]),
        dynamic_friction_range=tuple(randomization["friction_range"]),
    )
    cfg.events.base_mass.params["mass_distribution_params"] = tuple(randomization["added_base_mass_range"])
    cfg.events.limb_mass.params["mass_distribution_params"] = tuple(randomization["limb_mass_scale_range"])
    com_range = tuple(randomization["base_com_range"])
    cfg.events.base_com.params["com_range"] = {axis: com_range for axis in "xyz"}
    return cfg


class RecoveryEnv(IsaacVectorizedEnvironmentBase, DirectRLEnv):
    """Vectorized Go2 fall-recovery task with asymmetric observations."""

    cfg: RecoveryEnvCfg

    def __init__(self, cfg: RecoveryEnvCfg, *, task_cfg: dict, render_mode: str | None = None):
        self.task_cfg = task_cfg
        super().__init__(cfg, render_mode=render_mode)

        self._joint_ids, names = self._robot.find_joints(list(GO2_JOINT_NAMES), preserve_order=True)
        if tuple(names) != GO2_JOINT_NAMES:
            raise RuntimeError(f"Go2 USD joint ABI mismatch: {names}")
        self._base_ids, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        limb_names = [name.removesuffix("_joint") for name in GO2_JOINT_NAMES]
        self._limb_ids, found_limb_names = self._robot.find_bodies(limb_names, preserve_order=True)
        if len(self._base_ids) != 1 or len(self._feet_ids) != 4 or len(self._limb_ids) != 12:
            raise RuntimeError(
                f"Go2 USD body ABI mismatch: base={self._base_ids}, feet={self._feet_ids}, limbs={found_limb_names}"
            )

        shape = (self.num_envs, self.cfg.action_space)
        self._actions = torch.zeros(shape, device=self.device)
        self._last_actions = torch.zeros_like(self._actions)
        self._last_last_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._command_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._rotate_steps = torch.zeros_like(self._command_steps)
        self._default_joint_position = torch.tensor(task_cfg["robot"]["default_joint_position"], device=self.device)
        self._action_scale = torch.tensor(task_cfg["robot"]["action_scale"], device=self.device)
        self._joint_targets = self._default_joint_position.repeat(self.num_envs, 1)
        self._previous_quaternion = torch.zeros((self.num_envs, 4), device=self.device)
        self._velocity_estimator = ProprioceptiveVelocityEstimator(
            self.num_envs, self.device, dt=self.step_dt
        )
        self._recently_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._perturbation_force = torch.zeros((self.num_envs, 3), device=self.device)
        self._perturbation_torque = torch.zeros_like(self._perturbation_force)
        self._perturbation_steps = torch.zeros_like(self._command_steps)
        self._next_perturbation = torch.zeros_like(self._command_steps)
        self._base_body_tensor = torch.as_tensor(self._base_ids, dtype=torch.long, device=self.device)
        self._randomize_actuator_gains()
        self._cache_privileged_parameters()

        self.training_iteration = 0
        self.transition_critic_observation = torch.zeros((self.num_envs, 260), device=self.device)
        self._has_reset = False
        reward_names = list(task_cfg["rewards"]["scales"])
        self._episode_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in reward_names}

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _randomize_actuator_gains(self) -> None:
        randomization = self.task_cfg["domain_randomization"]
        kp_range = randomization["stiffness_scale_range"]
        kd_range = randomization["damping_scale_range"]
        self._kp_scale = torch.empty((self.num_envs, 1), device=self.device).uniform_(*kp_range)
        self._kd_scale = torch.empty((self.num_envs, 1), device=self.device).uniform_(*kd_range)
        actuator = self._robot.actuators["legs"]
        actuator.stiffness[:] = float(self.task_cfg["robot"]["stiffness"]) * self._kp_scale
        actuator.damping[:] = float(self.task_cfg["robot"]["damping"]) * self._kd_scale

    def _cache_privileged_parameters(self) -> None:
        masses = self._robot.root_physx_view.get_masses().to(self.device)
        default_masses = self._robot.data.default_mass
        self._base_mass = masses[:, self._base_ids]
        self._limb_mass = masses[:, self._limb_ids]
        self._base_mass_delta = self._base_mass - default_masses[:, self._base_ids]
        materials = self._robot.root_physx_view.get_material_properties().to(self.device)
        self._friction = materials[:, :1, 0].reshape(self.num_envs, 1)
        self._base_com = self._robot.data.body_com_pos_b[:, self._base_ids, :].reshape(self.num_envs, 3).clone()

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ranges = self.task_cfg["commands"]
        for column, key in enumerate(("linear_x", "linear_y", "angular_z")):
            self._commands[env_ids, column].uniform_(*ranges[key])
        self._command_steps[env_ids] = max(1, round(float(ranges["resampling_time_s"]) / self.step_dt))

    def _update_perturbations(self) -> None:
        settings = self.task_cfg["domain_randomization"]["perturbation"]
        active = self._perturbation_steps > 0
        self._perturbation_steps[active] -= 1
        expired = active & (self._perturbation_steps == 0)
        self._perturbation_force[expired] = 0.0
        self._perturbation_torque[expired] = 0.0
        waiting = self._perturbation_steps == 0
        self._next_perturbation[waiting] -= 1
        start = waiting & (self._next_perturbation <= 0)
        if start.any():
            env_ids = start.nonzero(as_tuple=False).squeeze(-1)
            self._perturbation_force[env_ids, :2].uniform_(*settings["force_xy_range"])
            self._perturbation_force[env_ids, 2] = 0.0
            self._perturbation_torque[env_ids].uniform_(*settings["torque_xyz_range"])
            self._perturbation_steps[env_ids] = max(1, round(float(settings["duration_s"]) / self.step_dt))
            interval = max(1, round(float(settings["interval_s"]) / self.step_dt))
            self._next_perturbation[env_ids] = torch.randint(
                max(1, interval // 2), interval + interval // 2 + 1,
                (env_ids.numel(),), device=self.device,
            )
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._perturbation_force.unsqueeze(1),
            torques=self._perturbation_torque.unsqueeze(1),
            body_ids=self._base_body_tensor,
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self._last_last_actions.copy_(self._last_actions)
        self._last_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-self.task_cfg["environment"]["clip_actions"],
                                          self.task_cfg["environment"]["clip_actions"]))
        self._joint_targets = self._default_joint_position + self._action_scale * self._actions
        self._command_steps -= 1
        self._resample_commands((self._command_steps <= 0).nonzero(as_tuple=False).squeeze(-1))
        self._update_perturbations()

    def _apply_action(self):
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def _raw_observation(self) -> torch.Tensor:
        estimated_velocity = self._velocity_estimator.update(
            self._robot.data.joint_pos[:, self._joint_ids],
            self._robot.data.joint_vel[:, self._joint_ids],
            self._robot.data.root_ang_vel_b,
            self._robot.data.root_quat_w,
            self._imu.data.lin_acc_b,
        )
        reset_ids = self._recently_reset.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel():
            # Reset observations represent a new episode and must not integrate
            # the final pre-reset IMU sample into the velocity estimate.
            self._velocity_estimator.reset(reset_ids)
            estimated_velocity[reset_ids] = 0.0
            self._recently_reset[reset_ids] = False
        raw, next_quaternion = self.build_canonical_raw_observation(
            self._robot.data.joint_pos[:, self._joint_ids],
            self._robot.data.joint_vel[:, self._joint_ids],
            self._robot.data.root_ang_vel_b,
            estimated_velocity,
            self._robot.data.root_quat_w,
            self._joint_targets,
            previous_quaternion=self._previous_quaternion,
        )
        self._previous_quaternion.copy_(next_quaternion)
        return raw

    def _critic_observation(self, raw: torch.Tensor | None = None) -> torch.Tensor:
        obs_cfg = self.task_cfg["observation"]
        if raw is None:
            # Timeout bootstrapping is captured before DirectRLEnv resets. Read
            # the needed simulator fields directly so the stateful velocity
            # estimator still advances exactly once in _get_observations().
            joint_position = self._robot.data.joint_pos[:, self._joint_ids]
            joint_velocity = self._robot.data.joint_vel[:, self._joint_ids]
            quaternion = self._robot.data.root_quat_w
        else:
            joint_position = raw[:, RAW_OBSERVATION_SPEC.joint_position]
            joint_velocity = raw[:, RAW_OBSERVATION_SPEC.joint_velocity]
            quaternion = raw[:, RAW_OBSERVATION_SPEC.base_quaternion_wxyz]
        gravity = projected_gravity(quaternion)
        q = joint_position - self._default_joint_position
        dq = float(obs_cfg["joint_velocity_scale"]) * joint_velocity
        command_scale = torch.tensor(obs_cfg["command_scale"], device=self.device)
        heights = (
            self._robot.data.root_pos_w[:, 2:3] - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        ).clamp(-1.0, 1.0) * float(obs_cfg["height_scale"])
        core = torch.cat(
            (
                2.0 * self._robot.data.root_lin_vel_b,
                float(obs_cfg["angular_velocity_scale"]) * self._robot.data.root_ang_vel_b,
                self._commands * command_scale,
                q,
                dq,
                self._actions,
                gravity,
                self._perturbation_force[:, :2],
                self._perturbation_torque,
                self._friction,
                self._base_mass / 15.0,
                self._limb_mass / 5.0,
                self._base_com,
                self._kp_scale,
                self._kd_scale,
                self._base_mass_delta / 3.0,
            ),
            dim=-1,
        )
        if core.shape[-1] != 73 or heights.shape[-1] != 187:
            raise RuntimeError(f"critic ABI mismatch: core={core.shape[-1]}, heights={heights.shape[-1]}")
        return torch.cat((core, heights), dim=-1).clamp(
            -self.task_cfg["environment"]["clip_observations"],
            self.task_cfg["environment"]["clip_observations"],
        )

    def _actor_observation(self, raw: torch.Tensor) -> torch.Tensor:
        cfg = self.task_cfg["observation"]
        observation = ppo_observation(
            raw,
            self._commands,
            default_joint_position=self._default_joint_position,
            action_scale=self._action_scale,
            angular_velocity_scale=float(cfg["angular_velocity_scale"]),
            joint_velocity_scale=float(cfg["joint_velocity_scale"]),
            command_scale=cfg["command_scale"],
            clip=float(self.task_cfg["environment"]["clip_observations"]),
        )
        noise_cfg = cfg["noise"]
        if noise_cfg["enabled"]:
            noise = torch.zeros_like(observation)
            noise[:, 0:3].normal_(std=float(noise_cfg["angular_velocity"]))
            noise[:, 3:6].normal_(std=float(noise_cfg["projected_gravity"]))
            noise[:, 9:21].normal_(std=float(noise_cfg["joint_position"]))
            noise[:, 21:33].normal_(std=float(noise_cfg["joint_velocity"]))
            observation = observation + float(noise_cfg["level"]) * noise
        return observation

    def _get_observations(self) -> dict[str, torch.Tensor]:
        raw = self._raw_observation()
        return {"policy": self._actor_observation(raw), "critic": self._critic_observation(raw)}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        gravity_z = projected_gravity(self._robot.data.root_quat_w)[:, 2]
        rotated = gravity_z > float(self.task_cfg["termination"]["rotated_projected_gravity_z"])
        self._rotate_steps = torch.where(rotated, self._rotate_steps + 1, torch.zeros_like(self._rotate_steps))
        terminated = self._rotate_steps >= int(self.task_cfg["termination"]["rotated_sustain_steps"])
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.transition_critic_observation = self._critic_observation().detach()
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        contact_forces = self._contact_sensor.data.net_forces_w
        foot_forces = contact_forces[:, self._feet_ids]
        base_forces = contact_forces[:, self._base_ids]
        heights = self._robot.data.root_pos_w[:, 2:3] - self._height_scanner.data.ray_hits_w[..., 2]
        terms = recovery_reward_terms(
            projected_gravity=projected_gravity(self._robot.data.root_quat_w),
            base_height=heights.mean(dim=-1),
            foot_contacts=torch.linalg.vector_norm(foot_forces, dim=-1) > 1.0,
            joint_position=self._robot.data.joint_pos[:, self._joint_ids],
            joint_velocity=self._robot.data.joint_vel[:, self._joint_ids],
            joint_acceleration=self._robot.data.joint_acc[:, self._joint_ids],
            applied_torque=self._robot.data.applied_torque[:, self._joint_ids],
            action=self._actions,
            previous_action=self._last_actions,
            previous_previous_action=self._last_last_actions,
            soft_joint_limits=self._robot.data.soft_joint_pos_limits[:, self._joint_ids],
            base_contacts=torch.linalg.vector_norm(base_forces, dim=-1) > 0.1,
            foot_contact_forces=foot_forces,
            iteration=self.training_iteration,
            stand_curriculum_iterations=int(self.task_cfg["rewards"]["stand_target_curriculum_iterations"]),
        )
        terms["angular_velocity_xyz"] = torch.sum(self._robot.data.root_ang_vel_b.square(), dim=-1)
        scales = self.task_cfg["rewards"]["scales"]
        for name, term in terms.items():
            self._episode_sums[name] += term * float(scales[name]) * self.step_dt
        return weighted_reward(
            terms, scales, self.step_dt,
            only_positive=bool(self.task_cfg["rewards"]["only_positive"]),
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if self._has_reset and env_ids.numel() > 0 and self.task_cfg["terrain"]["curriculum"]:
            move_down = self.reset_terminated[env_ids]
            move_up = self.reset_time_outs[env_ids] & ~move_down
            self._terrain.update_env_origins(env_ids, move_up, move_down)

        if self._has_reset and env_ids.numel() > 0:
            self.extras["log"] = {}
            for name, values in self._episode_sums.items():
                self.extras["log"][f"Episode_Reward/{name}"] = values[env_ids].mean().item()
                values[env_ids] = 0.0
            self.extras["log"]["Episode_Termination/rotated"] = self.reset_terminated[env_ids].sum().item()
            self.extras["log"]["Episode_Termination/time_out"] = self.reset_time_outs[env_ids].sum().item()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        count = env_ids.numel()
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._last_last_actions[env_ids] = 0.0
        self._rotate_steps[env_ids] = 0
        self._velocity_estimator.reset(env_ids)
        self._recently_reset[env_ids] = True
        self._joint_targets[env_ids] = self._default_joint_position
        self._perturbation_force[env_ids] = 0.0
        self._perturbation_torque[env_ids] = 0.0
        self._perturbation_steps[env_ids] = 0
        interval = max(
            1,
            round(self.task_cfg["domain_randomization"]["perturbation"]["interval_s"] / self.step_dt),
        )
        self._next_perturbation[env_ids] = torch.randint(1, interval + 1, (count,), device=self.device)
        self._resample_commands(env_ids)

        joint_pos = self._default_joint_position * torch.empty((count, 12), device=self.device).uniform_(0.5, 1.5)
        limits = self._robot.data.soft_joint_pos_limits[env_ids][:, self._joint_ids]
        joint_pos = torch.maximum(torch.minimum(joint_pos, limits[..., 1]), limits[..., 0])
        joint_vel = torch.zeros_like(joint_pos)
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        root_state[:, :2] += torch.empty((count, 2), device=self.device).uniform_(-3.0, 3.0)
        quaternion = torch.randn((count, 4), device=self.device)
        quaternion /= torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        quaternion = torch.where(quaternion[:, :1] < 0.0, -quaternion, quaternion)
        root_state[:, 3:7] = quaternion
        root_state[:, 7:13].uniform_(-0.5, 0.5)
        self._previous_quaternion[env_ids] = quaternion
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, self._joint_ids, env_ids)
        self._robot.set_joint_position_target(self._joint_targets[env_ids], self._joint_ids, env_ids)

        if not self._has_reset and count == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=self.max_episode_length)
        self._has_reset = True
