"""Load and validate the single approved recovery configuration."""

from __future__ import annotations

from pathlib import Path

from common.config import apply_runtime_overrides, load_config
from common.observation import PPO_OBSERVATION_ABI, RAW_OBSERVATION_ABI
from common.terrain import RECOVERY_TERRAIN_MIX, validate_terrain_mix


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_recovery_config(cfg: dict) -> None:
    """Protect the fixed task ABI and the approved training recipe."""

    env = cfg["environment"]
    ppo = cfg["ppo"]
    terrain = cfg["terrain"]
    observation = cfg["observation"]
    termination = cfg["termination"]
    robot = cfg["robot"]

    _require(env["num_envs"] == 4096, "recovery training must use 4096 environments")
    _require(env["actor_observation_dim"] == 45, "actor observation must be 45D")
    _require(env["critic_observation_dim"] == 260, "critic observation must be 260D")
    _require(env["action_dim"] == 12, "Go2 recovery action must be 12D")
    _require(ppo["max_iterations"] == 4000, "recovery PPO must use 4000 iterations")
    _require(observation["raw_abi"] == RAW_OBSERVATION_ABI, "raw observation ABI mismatch")
    _require(observation["policy_view"] == PPO_OBSERVATION_ABI, "PPO observation ABI mismatch")
    _require(len(robot["default_joint_position"]) == 12, "default joint position must be 12D")
    _require(len(robot["action_scale"]) == 12, "action scale must be 12D")
    _require(
        termination["rotated_projected_gravity_z"] == -0.8
        and termination["rotated_sustain_steps"] == 400,
        "rotation termination must be projected_gravity_z > -0.8 for 400 steps",
    )

    validate_terrain_mix(RECOVERY_TERRAIN_MIX)
    expected_mix = {item.name: item.proportion for item in RECOVERY_TERRAIN_MIX}
    _require(terrain["mix"] == expected_mix, "terrain mix must contain the five approved 20% terrains")
    scanner = terrain["height_scanner"]
    points = (round(scanner["size"][0] / scanner["resolution"]) + 1) * (
        round(scanner["size"][1] / scanner["resolution"]) + 1
    )
    _require(points == 187, "height scanner must provide 187 critic samples")


def load_recovery_config(**overrides) -> dict:
    cfg = apply_runtime_overrides(load_config(CONFIG_PATH), **overrides)
    validate_recovery_config(cfg)
    return cfg
