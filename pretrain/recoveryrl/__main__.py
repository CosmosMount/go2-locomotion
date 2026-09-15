"""Train the task, safety, and MF recovery parts of the source bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from common.checkpoint import checkpoint_metadata, save_checkpoint
from common.config import load_config
from common.observation import RAW_OBSERVATION_SIZE
from common.output import prepare_output_dir
from .core import AgentConfig, RecoveryAgent, ReplayBuffer
from .core import POLICY_VIEW


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _load_config(path: Path) -> dict:
    config = load_config(path)
    required = {
        "seed": 1701,
        "transitions": 200000,
        "num_envs": 16,
        "batch_size": 256,
        "replay_capacity": 400000,
        "learning_starts": 2500,
        "gamma_safe": 0.9607894391523232,
        "tau_safe": 0.0002,
        "safety_positive_fraction": 0.25,
        "risk_threshold_epsilon": 0.1,
        "alpha_init": 1.0,
        "recovery_updates": 10000,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"fixed protocol requires {key}={expected!r}")
    if config.get("task_observation_view") != "yaw_invariant":
        raise ValueError("task_observation_view must be yaw_invariant")
    if config.get("safety_observation_view") != "raw":
        raise ValueError("safety_observation_view must be raw")
    if config["transitions"] % config["num_envs"]:
        raise ValueError("transitions must divide evenly by num_envs")
    environment = config["environment"]
    if environment.get("backend") != "isaac_lab":
        raise ValueError("source safety training requires Isaac Lab")
    ratio = environment["control_dt"] / environment["physics_dt"]
    if ratio != round(ratio):
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    default = np.asarray(environment.get("default_joint_position"), dtype=float)
    scale = np.asarray(environment.get("action_scale"), dtype=float)
    if default.shape != (12,) or not np.isfinite(default).all():
        raise ValueError("default_joint_position must contain 12 finite values")
    if scale.shape != (12,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("action_scale must contain 12 positive finite values")
    push_interval = np.asarray(environment.get("push_interval_s"), dtype=float)
    if (
        push_interval.shape != (2,)
        or not np.isfinite(push_interval).all()
        or push_interval[0] <= 0
        or push_interval[1] < push_interval[0]
    ):
        raise ValueError("push_interval_s must contain two ordered positive values")
    push_velocity = np.asarray(environment.get("push_velocity_xy"), dtype=float)
    if (
        push_velocity.shape != (2,)
        or not np.isfinite(push_velocity).all()
        or push_velocity[1] < push_velocity[0]
    ):
        raise ValueError("push_velocity_xy must contain two ordered finite values")
    return config


def _agent_config(config: dict, seed: int, batch_size: int) -> AgentConfig:
    return AgentConfig(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["hidden_dim"],
        learning_rate=config["learning_rate"],
        batch_size=batch_size,
        gamma=config["gamma"],
        gamma_safe=config["gamma_safe"],
        tau=config["tau"],
        tau_safe=config["tau_safe"],
        safety_positive_fraction=config["safety_positive_fraction"],
        risk_threshold_epsilon=config["risk_threshold_epsilon"],
        alpha_init=config["alpha_init"],
        automatic_entropy_tuning=config["automatic_entropy_tuning"],
        target_entropy=config["target_entropy"],
        safety_continuation=config["safety_continuation"],
        task_observation_view=config["task_observation_view"],
        seed=seed,
    )


def _module_digest(agent: RecoveryAgent, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        for key, value in sorted(agent.modules[name].state_dict().items()):
            digest.update(f"{name}.{key}".encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    digest.update(
        agent.entropy_coefficient.log_alpha.detach().cpu().numpy().tobytes()
    )
    return digest.hexdigest()


def _metadata(config: dict, config_path: Path, effective: dict) -> dict:
    environment = config["environment"]
    metadata = checkpoint_metadata(
        task="safe",
        policy_view=POLICY_VIEW,
        policy_observation_size=RAW_OBSERVATION_SIZE,
        default_joint_position=environment["default_joint_position"],
        action_scale=environment["action_scale"],
        observation_scales={},
    )
    metadata.update(
        checkpoint_purpose="source_only_safety_pretrain_for_online_adaptation",
        source_domain="isaac_lab_flat_go2",
        policy_action_dim=12,
        safety_q_output="sigmoid_twin_probability",
        safety_q_reduction="maximum",
        safety_target_continuation="next_task_action",
        timeout_bootstrap=True,
        task_replay_action="proposed_action",
        safety_replay_action="executed_action",
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        fixed_protocol={
            key: value for key, value in config.items() if key not in ("environment", "provenance")
        },
        safety_policy_view="raw",
        environment=environment,
        provenance=config["provenance"],
        effective_training=effective,
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=None
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = CONFIG_PATH.resolve()
    config = _load_config(config_path)
    seed = config["seed"] if args.seed is None else args.seed
    environment_config = dict(config["environment"])
    environment_config["device"] = args.device or environment_config["device"]
    if args.headless is not None:
        environment_config["headless"] = args.headless

    effective = {
        "seed": seed,
        "transitions": config["transitions"],
        "num_envs": config["num_envs"],
        "batch_size": config["batch_size"],
        "replay_capacity": config["replay_capacity"],
        "learning_starts": config["learning_starts"],
        "recovery_updates": config["recovery_updates"],
    }
    environment_config["num_envs"] = effective["num_envs"]
    if effective["transitions"] % effective["num_envs"]:
        raise ValueError("effective transitions must divide evenly by num_envs")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Isaac Lab source training requires a working NVIDIA driver"
        )
    output = prepare_output_dir(
        args.output_dir, temp_prefix="go2-pretrain-recoveryrl-"
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    # Isaac/pxr-dependent modules must be imported only after AppLauncher.
    from common.isaac import launch_app

    application = launch_app(
        headless=bool(environment_config["headless"]),
        device=environment_config["device"],
    )
    environment = None
    try:
        from .environment import make_source_environment

        environment = make_source_environment(environment_config)
        agent = RecoveryAgent(
            _agent_config(config, seed, effective["batch_size"]),
            environment_config["device"],
        )
        replay = ReplayBuffer(
            effective["replay_capacity"], seed=seed + 1
        )
        observation = environment.reset()
        transitions = 0
        failures = 0
        metrics = {}
        started = time.monotonic()
        report_every = max(effective["num_envs"] * 25, effective["transitions"] // 20)
        next_report = report_every

        while transitions < effective["transitions"]:
            agent.observe(observation)
            proposed_action = agent.act_task(observation)
            (
                next_observation,
                reward,
                cost,
                terminated,
                truncated,
                info,
            ) = environment.step(proposed_action)
            replay.add_batch(
                observation,
                proposed_action,
                proposed_action,
                reward,
                cost,
                info["transition_observation"],
                terminated,
                truncated,
            )
            observation = next_observation
            transitions += effective["num_envs"]
            failures += int(np.asarray(cost).sum())
            if len(replay) >= effective["learning_starts"]:
                for _ in range(
                    effective["num_envs"] * config["updates_per_transition"]
                ):
                    task_metrics = agent.update_task(replay)
                    safety_metrics = agent.update_safety(replay)
                    agent.stage1_updates += 1
                    metrics = {**task_metrics, **safety_metrics}
            if transitions >= next_report or transitions == effective["transitions"]:
                print(
                    json.dumps(
                        {
                            "stage": 1,
                            "transitions": transitions,
                            "failures": failures,
                            "positive_replay": replay.positive_count,
                            "updates": agent.stage1_updates,
                            "alpha": metrics.get("alpha"),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
                next_report += report_every

        if replay.positive_count == 0:
            raise RuntimeError("source replay has no safety violations")
        frozen_names = (
            "task_actor",
            "reward_q",
            "reward_target",
            "safety_q",
            "safety_target",
            "task_normalizer",
            "safety_normalizer",
        )
        agent.freeze_stage1()
        stage1_digest = _module_digest(agent, frozen_names)
        for update in range(effective["recovery_updates"]):
            metrics = agent.update_recovery(replay)
            if update + 1 == effective["recovery_updates"] or (
                update + 1
            ) % max(1, effective["recovery_updates"] // 10) == 0:
                print(
                    json.dumps(
                        {
                            "stage": 2,
                            "recovery_updates": update + 1,
                            **metrics,
                        }
                    ),
                    flush=True,
                )
        if _module_digest(agent, frozen_names) != stage1_digest:
            raise RuntimeError("a frozen stage-1 module changed during stage 2")
        agent.prepare_for_adaptation()
        metadata = _metadata(config, config_path, effective)
        training_state = {
            "status": "complete",
            "source_transitions": transitions,
            "source_failures": failures,
            "source_positive_replay": replay.positive_count,
            "stage1_frozen_sha256": stage1_digest,
            "formal": True,
        }
        checkpoint = output / "safe.pt"
        save_checkpoint(
            checkpoint,
            state=agent.checkpoint_state(training_state),
            metadata=metadata,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "checkpoint": str(checkpoint),
                    "formal": True,
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        if environment is not None:
            environment.close()
        application.close()


if __name__ == "__main__":
    raise SystemExit(main())
