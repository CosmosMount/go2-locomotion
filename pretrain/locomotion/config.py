"""Typed view of the package's single locomotion configuration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping


CONFIG_PATH = Path(__file__).with_name("config.yaml")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def _tuple(value: Any, size: int, name: str, cast=float) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    return tuple(cast(item) for item in value)


@dataclass(frozen=True)
class LocomotionConfig:
    """Values consumed by the environment and the target ``rl`` PPO API."""

    name: str
    seed: int
    provenance: Mapping[str, Any]

    usd_path: Path
    joint_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    initial_base_pos: tuple[float, float, float]
    action_scale: float
    kp: float
    kd: float
    effort_limit: float
    velocity_limit: float
    soft_joint_pos_limit_factor: float

    raw_observation_abi: str
    raw_observation_size: int
    policy_view: str
    observation_size: int
    critic_view: str
    critic_observation_size: int
    quaternion_order: str
    estimated_velocity_frame: str
    angular_velocity_frame: str
    angular_velocity_scale: float
    command_scale: tuple[float, float, float]
    joint_velocity_scale: float
    torque_scale: float
    observation_noise: bool
    observation_clip: float

    device: str
    num_envs: int
    env_spacing: float
    physics_dt: float
    decimation: int
    episode_length_s: float
    terrain: str
    friction: float
    restitution: float
    contact_history_length: int

    command_vx: tuple[float, float]
    command_vy: tuple[float, float]
    command_yaw: tuple[float, float]
    standing_fraction: float
    command_resampling_s: float
    command_curriculum: bool

    domain_randomization: bool
    pushes: bool
    push_interval_s: tuple[float, float]
    push_velocity: tuple[float, float]
    reset_joint_velocity: tuple[float, float]
    initial_xy: tuple[float, float]
    initial_yaw: tuple[float, float]
    bad_orientation_angle: float
    contact_force_threshold: float

    linear_tracking_std: float
    yaw_tracking_std: float
    feet_air_time_target: float
    moving_command_threshold: float
    reward_weights: Mapping[str, float]

    max_iterations: int
    horizon: int
    gamma: float
    gae_lambda: float
    learning_rate: float
    min_learning_rate: float
    max_learning_rate: float
    adaptive_learning_rate: bool
    update_epochs: int
    num_minibatches: int
    clip_ratio: float
    value_clip: float
    value_loss_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float
    std_dev: float
    min_std_dev: float
    max_std_dev: float
    target_kl: float
    kl_stop_multiplier: float
    action_clipping_and_rescaling: bool
    hidden_dims: tuple[int, ...]
    activation: str
    checkpoint_interval: int
    output_dir: Path
    compile_mode: None = None

    @property
    def action_size(self) -> int:
        return len(self.joint_names)

    @property
    def step_dt(self) -> float:
        return self.physics_dt * self.decimation

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LocomotionConfig":
        task = _section(data, "task")
        robot = _section(data, "robot")
        obs = _section(data, "observation")
        sim = _section(data, "simulation")
        commands = _section(data, "commands")
        randomization = _section(data, "randomization")
        termination = _section(data, "termination")
        reward = _section(data, "reward")
        ppo = _section(data, "ppo")
        usd_path = Path(str(robot["usd_path"]))
        output_dir = Path(str(ppo["output_dir"]))
        config = cls(
            name=str(task["name"]),
            seed=int(task["seed"]),
            provenance=dict(_section(data, "provenance")),
            usd_path=usd_path if usd_path.is_absolute() else REPOSITORY_ROOT / usd_path,
            joint_names=_tuple(robot["joint_names"], 12, "robot.joint_names", str),
            default_joint_pos=_tuple(robot["default_joint_position"], 12, "robot.default_joint_position"),
            initial_base_pos=_tuple(robot["initial_base_position"], 3, "robot.initial_base_position"),
            action_scale=float(robot["action_scale"]),
            kp=float(robot["stiffness"]),
            kd=float(robot["damping"]),
            effort_limit=float(robot["effort_limit"]),
            velocity_limit=float(robot["velocity_limit"]),
            soft_joint_pos_limit_factor=float(robot["soft_joint_position_limit_factor"]),
            raw_observation_abi=str(obs["raw_abi"]),
            raw_observation_size=int(obs["raw_size"]),
            policy_view=str(obs["policy_view"]),
            observation_size=int(obs["actor_size"]),
            critic_view=str(obs["critic_view"]),
            critic_observation_size=int(obs["critic_size"]),
            quaternion_order=str(obs["quaternion_order"]),
            estimated_velocity_frame=str(obs["estimated_velocity_frame"]),
            angular_velocity_frame=str(obs["angular_velocity_frame"]),
            angular_velocity_scale=float(obs["angular_velocity_scale"]),
            command_scale=_tuple(obs["command_scale"], 3, "observation.command_scale"),
            joint_velocity_scale=float(obs["joint_velocity_scale"]),
            torque_scale=float(obs["torque_scale"]),
            observation_noise=bool(obs["noise"]),
            observation_clip=float(obs["clip"]),
            device=str(sim["device"]),
            num_envs=int(sim["num_envs"]),
            env_spacing=float(sim["env_spacing"]),
            physics_dt=float(sim["physics_dt"]),
            decimation=int(sim["decimation"]),
            episode_length_s=float(sim["episode_length_s"]),
            terrain=str(sim["terrain"]),
            friction=float(sim["friction"]),
            restitution=float(sim["restitution"]),
            contact_history_length=int(sim["contact_history_length"]),
            command_vx=_tuple(commands["linear_x"], 2, "commands.linear_x"),
            command_vy=_tuple(commands["linear_y"], 2, "commands.linear_y"),
            command_yaw=_tuple(commands["yaw"], 2, "commands.yaw"),
            standing_fraction=float(commands["standing_fraction"]),
            command_resampling_s=float(commands["resampling_s"]),
            command_curriculum=bool(commands["curriculum"]),
            domain_randomization=bool(randomization["enabled"]),
            pushes=bool(randomization["pushes"]),
            push_interval_s=_tuple(randomization["push_interval_s"], 2, "randomization.push_interval_s"),
            push_velocity=_tuple(randomization["push_velocity_xy"], 2, "randomization.push_velocity_xy"),
            reset_joint_velocity=_tuple(randomization["reset_joint_velocity"], 2, "randomization.reset_joint_velocity"),
            initial_xy=_tuple(randomization["initial_xy"], 2, "randomization.initial_xy"),
            initial_yaw=_tuple(randomization["initial_yaw"], 2, "randomization.initial_yaw"),
            bad_orientation_angle=float(termination["bad_orientation_angle"]),
            contact_force_threshold=float(termination["contact_force_threshold"]),
            linear_tracking_std=float(reward["linear_tracking_std"]),
            yaw_tracking_std=float(reward["yaw_tracking_std"]),
            feet_air_time_target=float(reward["feet_air_time_target"]),
            moving_command_threshold=float(reward["moving_command_threshold"]),
            reward_weights={str(key): float(value) for key, value in _section(reward, "weights").items()},
            max_iterations=int(ppo["max_iterations"]),
            horizon=int(ppo["horizon"]),
            gamma=float(ppo["gamma"]),
            gae_lambda=float(ppo["gae_lambda"]),
            learning_rate=float(ppo["learning_rate"]),
            min_learning_rate=float(ppo["min_learning_rate"]),
            max_learning_rate=float(ppo["max_learning_rate"]),
            adaptive_learning_rate=bool(ppo["adaptive_learning_rate"]),
            update_epochs=int(ppo["update_epochs"]),
            num_minibatches=int(ppo["num_minibatches"]),
            clip_ratio=float(ppo["clip_ratio"]),
            value_clip=float(ppo["value_clip"]),
            value_loss_coefficient=float(ppo["value_loss_coefficient"]),
            entropy_coefficient=float(ppo["entropy_coefficient"]),
            max_grad_norm=float(ppo["max_grad_norm"]),
            std_dev=float(ppo["std_dev"]),
            min_std_dev=float(ppo["min_std_dev"]),
            max_std_dev=float(ppo["max_std_dev"]),
            target_kl=float(ppo["target_kl"]),
            kl_stop_multiplier=float(ppo["kl_stop_multiplier"]),
            action_clipping_and_rescaling=bool(ppo["action_clipping_and_rescaling"]),
            hidden_dims=_tuple(ppo["hidden_dims"], 3, "ppo.hidden_dims", int),
            activation=str(ppo["activation"]),
            checkpoint_interval=int(ppo["checkpoint_interval"]),
            output_dir=output_dir if output_dir.is_absolute() else REPOSITORY_ROOT / output_dir,
        )
        config.validate(check_assets=False)
        return config

    def with_overrides(
        self, *, seed: int | None = None, device: str | None = None, output_dir: Path | None = None
    ) -> "LocomotionConfig":
        return replace(
            self,
            seed=self.seed if seed is None else seed,
            device=self.device if device is None else device,
            output_dir=self.output_dir if output_dir is None else output_dir.expanduser().resolve(),
        )

    def validate(self, *, check_assets: bool = True) -> None:
        if self.name != "pretrain-locomotion" or self.terrain != "flat":
            raise ValueError("this package only supports the flat pretrain-locomotion task")
        expected_joints = tuple(
            f"{leg}_{joint}_joint"
            for leg in ("FL", "FR", "RL", "RR")
            for joint in ("hip", "thigh", "calf")
        )
        if self.joint_names != expected_joints:
            raise ValueError("joint order must be FL, FR, RL, RR with hip/thigh/calf order")
        if self.raw_observation_abi != "go2_raw_46d" or self.raw_observation_size != 46:
            raise ValueError("locomotion requires the canonical 46D raw observation ABI")
        if self.policy_view != "go2_ppo_45d" or self.observation_size != 45:
            raise ValueError("locomotion actor requires go2_ppo_45d")
        if self.critic_observation_size != 60 or self.critic_view != "ppo_go2_privileged_60d":
            raise ValueError("locomotion critic requires ppo_go2_privileged_60d")
        if (
            self.quaternion_order,
            self.estimated_velocity_frame,
            self.angular_velocity_frame,
        ) != (
            "WXYZ",
            "body",
            "body",
        ):
            raise ValueError("raw observation frames/order do not match go2_raw_46d")
        if self.hidden_dims != (512, 256, 128) or self.activation != "elu":
            raise ValueError("the single supported MLP is 512/256/128 with ELU")
        if not math.isclose(self.step_dt, 0.02, abs_tol=1e-12):
            raise ValueError("control period must be 20 ms")
        if self.max_iterations != 5000 or self.num_envs != 4096:
            raise ValueError("the checked-in config must use 4096 envs and 5000 iterations")
        if self.action_size != 12 or len(self.default_joint_pos) != self.action_size:
            raise ValueError("Go2 must expose twelve joints and default positions")
        if self.seed < 0 or self.physics_dt <= 0 or self.decimation <= 0 or self.episode_length_s <= 0:
            raise ValueError("invalid seed or simulation timing")
        if not 0 <= self.standing_fraction <= 1 or self.command_resampling_s <= 0:
            raise ValueError("invalid command sampling configuration")
        for name, limits in (
            ("command_vx", self.command_vx), ("command_vy", self.command_vy),
            ("command_yaw", self.command_yaw), ("push_interval_s", self.push_interval_s),
            ("push_velocity", self.push_velocity), ("reset_joint_velocity", self.reset_joint_velocity),
            ("initial_xy", self.initial_xy), ("initial_yaw", self.initial_yaw),
        ):
            if limits[0] > limits[1]:
                raise ValueError(f"{name} lower limit exceeds its upper limit")
        if not 0 < self.min_std_dev <= self.std_dev <= self.max_std_dev < math.inf:
            raise ValueError("invalid policy standard-deviation bounds")
        if not 0 < self.min_learning_rate <= self.learning_rate <= self.max_learning_rate:
            raise ValueError("invalid PPO learning-rate bounds")
        if min(self.horizon, self.update_epochs, self.num_minibatches, self.checkpoint_interval) <= 0:
            raise ValueError("PPO loop counts must be positive")
        if not self.action_clipping_and_rescaling or self.compile_mode is not None:
            raise ValueError("target rl PPO requires clipped environment actions and compile_mode=None")
        expected_rewards = {
            "track_linear_velocity", "track_yaw_velocity", "vertical_velocity",
            "roll_pitch_velocity", "flat_orientation", "joint_torque",
            "joint_acceleration", "action_rate", "feet_air_time",
            "undesired_contacts", "joint_velocity", "joint_position",
            "joint_limits", "energy", "air_time_variance", "feet_slide",
        }
        if set(self.reward_weights) != expected_rewards:
            raise ValueError("reward.weights does not match the implemented task")
        if check_assets:
            from common.isaac import require_go2_usd

            shared_usd = require_go2_usd()
            if self.usd_path.resolve() != shared_usd.resolve():
                raise ValueError(
                    f"locomotion must use the shared Go2 USD {shared_usd}, got {self.usd_path}"
                )

    def to_mapping(self) -> dict[str, Any]:
        def serializable(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [serializable(item) for item in value]
            if isinstance(value, Mapping):
                return {str(key): serializable(item) for key, item in value.items()}
            return value

        return serializable(asdict(self))


def load_locomotion_config(path: Path = CONFIG_PATH) -> LocomotionConfig:
    """Load the one YAML through the repository-wide configuration API."""

    try:
        from common.config import load_config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "common.config.load_config(path) is required by pretrain.locomotion"
        ) from exc
    data = load_config(path)
    if not isinstance(data, Mapping):
        raise TypeError("common.config.load_config must return a mapping")
    config = LocomotionConfig.from_mapping(data)
    config.validate(check_assets=True)
    return config
