"""Run target-domain MuJoCo model-free Recovery RL adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import traceback

import numpy as np
import torch

from common.config import load_config
from common.output import prepare_output_dir
from .agent import POLICY_VIEW, RecoveryAgent, ReplayBuffer
from .environment import MujocoRecoveryEnv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_digest(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _configuration() -> dict:
    path = Path(__file__).with_name("config.yaml")
    config = load_config(path)
    adaptation = config["adaptation"]
    if adaptation["seed"] != 3701:
        raise ValueError("the approved target-domain seed is 3701")
    if adaptation["reward_critic_warmup_updates"] != 1000:
        raise ValueError("the approved reward-critic warmup is exactly 1000 updates")
    if adaptation["adaptation_interactions"] != 200000:
        raise ValueError("the approved adaptation budget is exactly 200000 interactions")
    if config["checkpoint_contract"]["policy_view"] != POLICY_VIEW:
        raise ValueError("recoveryrl requires the common yaw-invariant task view")
    return config


def _add_task_transition(replay, observation, action, reward,
                         transition_observation, terminated, truncated):
    replay.add(
        observation,
        action["proposed_action"],
        action["executed_action"],
        reward,
        float(terminated),
        transition_observation,
        terminated,
        truncated,
    )


def _save(agent, path, replay, source_sha256, counters, simulation):
    agent.save(
        path,
        replay=replay,
        source_sha256=source_sha256,
        counters=counters,
        default_joint_position=simulation["default_joint_position"],
        action_scale=simulation["action_scale"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="pretrain.recoveryrl checkpoint using the common metadata/state envelope",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", help="Torch device; defaults to config.yaml")
    parser.add_argument("--seed", type=int, help="override the default seed 3701")
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True,
        help="accepted for root-runner consistency; this MuJoCo task has no viewer",
    )
    args = parser.parse_args(argv)

    config = _configuration()
    simulation = config["simulation"]
    adaptation = config["adaptation"]
    checkpoint_contract = config["checkpoint_contract"]
    seed = adaptation["seed"] if args.seed is None else args.seed
    adaptation["seed"] = seed
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"source-safe checkpoint does not exist: {checkpoint}")
    output = prepare_output_dir(args.output_dir, temp_prefix="go2-recoveryrl-")
    print(f"output={output}", flush=True)
    device = args.device or adaptation["device"]
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    source_sha256 = _sha256(checkpoint)
    state = {
        "status": "starting",
        "phase": "load_source",
        "seed": seed,
        "warmup_collection_interactions": 0,
        "warmup_updates": 0,
        "adaptation_interactions": 0,
        "task_updates": 0,
        "task_replay_size": 0,
        "recovery_interventions": 0,
        "task_failures": 0,
        "replay_failure_transitions": 0,
        "source_checkpoint": str(checkpoint),
        "source_sha256": source_sha256,
    }
    _write_json(output / "state.json", state)
    agent = env = None
    try:
        agent, source_metadata = RecoveryAgent.from_source_checkpoint(
            checkpoint,
            source_task=checkpoint_contract["source_task"],
            policy_view=checkpoint_contract["policy_view"],
            default_joint_position=simulation["default_joint_position"],
            action_scale=simulation["action_scale"],
            seed=seed,
            batch_size=adaptation["batch_size"],
            epsilon=adaptation["risk_threshold"],
            device=device,
        )
        replay = ReplayBuffer(adaptation["replay_capacity"])
        frozen_source = agent.frozen_source_digest()
        source_actor_before_warmup = _module_digest(agent.task_actor)
        protocol = {
            "configuration": config,
            "source_checkpoint": str(checkpoint),
            "source_sha256": source_sha256,
            "source_metadata": source_metadata,
            "device": str(agent.device),
            "transferred_from_source": [
                "task_actor", "safety_q", "safety_target", "recovery_actor",
                "task_normalizer", "safety_normalizer", "entropy_coefficient",
            ],
            "target_initialized": ["reward_q", "reward_target", "task_optimizers"],
            "online_updates": ["task_actor", "reward_q", "reward_target"],
            "frozen": [
                "safety_q", "safety_target", "recovery_actor",
                "task_normalizer", "safety_normalizer", "entropy_coefficient",
            ],
            "task_view": "common.observation.prepare_task_observation(yaw_invariant)",
            "safety_view": "canonical raw46",
            "intervention_replay": "excluded",
            "task_failure_transition": "retained_terminal",
        }
        _write_json(output / "protocol.json", protocol)
        env = MujocoRecoveryEnv(simulation, seed=seed)
        observation = env.reset(seed=seed)

        state.update(status="running", phase="warmup_collection")
        with (output / "training.jsonl").open("w", buffering=1) as trace:
            for interaction in range(1, adaptation["warmup_collection_interactions"] + 1):
                action = agent.act(observation, use_recovery=False)
                next_observation, reward, terminated, truncated, info = env.step(
                    action["executed_action"]
                )
                _add_task_transition(
                    replay, observation, action, reward,
                    info["transition_observation"], terminated, truncated,
                )
                observation = next_observation
                state["warmup_collection_interactions"] = interaction
                state["task_replay_size"] = len(replay)
                state["task_failures"] += int(terminated)
                state["replay_failure_transitions"] += int(terminated)
                trace.write(json.dumps({
                    "phase": "warmup_collection",
                    "interaction": interaction,
                    "reward": reward,
                    "failure": terminated,
                    "truncated": truncated,
                    "risk": action["risk"],
                    "recovery": False,
                    "replay_size": len(replay),
                }, allow_nan=False) + "\n")
                if interaction % adaptation["state_interval"] == 0:
                    _write_json(output / "state.json", state)

            state["phase"] = "reward_critic_warmup"
            for update in range(1, adaptation["reward_critic_warmup_updates"] + 1):
                metrics = agent.update_reward_critic(replay)
                state["warmup_updates"] = update
                if update % adaptation["state_interval"] == 0 or update == 1:
                    state["metrics"] = metrics
                    _write_json(output / "state.json", state)
            if _module_digest(agent.task_actor) != source_actor_before_warmup:
                raise RuntimeError("reward-critic warmup changed the task actor")
            if agent.frozen_source_digest() != frozen_source:
                raise RuntimeError("reward-critic warmup changed a frozen source module")

            state["phase"] = "adaptation"
            metrics = {}
            for interaction in range(1, adaptation["adaptation_interactions"] + 1):
                action = agent.act(observation, use_recovery=True)
                replay_size_before = len(replay)
                updates_before = agent.steps
                next_observation, reward, terminated, truncated, info = env.step(
                    action["executed_action"]
                )
                state["task_failures"] += int(terminated)
                if action["recovered"]:
                    if len(replay) != replay_size_before or agent.steps != updates_before:
                        raise RuntimeError("recovery intervention wrote replay or updated the learner")
                    state["recovery_interventions"] += 1
                else:
                    _add_task_transition(
                        replay, observation, action, reward,
                        info["transition_observation"], terminated, truncated,
                    )
                    if terminated and not bool(replay.data["terminated"][
                        (replay.position - 1) % replay.capacity, 0
                    ]):
                        raise RuntimeError("task failure transition was not terminal")
                    state["replay_failure_transitions"] += int(terminated)
                    # Preserve the completed failure sample, but update from it only
                    # through later replay sampling after the environment reset.
                    if not terminated and len(replay) >= adaptation["batch_size"]:
                        metrics = agent.update_task(replay)
                observation = next_observation
                state["adaptation_interactions"] = interaction
                state["task_updates"] = agent.steps
                state["task_replay_size"] = len(replay)
                trace.write(json.dumps({
                    "phase": "adaptation",
                    "interaction": interaction,
                    "reward": reward,
                    "failure": terminated,
                    "truncated": truncated,
                    "risk": action["risk"],
                    "recovery": action["recovered"],
                    "replay_size": len(replay),
                    "task_updates": agent.steps,
                }, allow_nan=False) + "\n")

                report = interaction % adaptation["state_interval"] == 0
                checkpoint_due = interaction % adaptation["checkpoint_interval"] == 0
                if report or checkpoint_due or interaction == adaptation["adaptation_interactions"]:
                    if agent.frozen_source_digest() != frozen_source:
                        raise RuntimeError("a frozen safety/recovery/normalizer module changed")
                    state["metrics"] = metrics
                    _write_json(output / "state.json", state)
                if checkpoint_due:
                    _save(
                        agent, output / f"checkpoint_{interaction}.pt", replay,
                        source_sha256, state, simulation,
                    )

        if _sha256(checkpoint) != source_sha256:
            raise RuntimeError("source-safe checkpoint changed during adaptation")
        if agent.frozen_source_digest() != frozen_source:
            raise RuntimeError("frozen source state changed during adaptation")
        _save(agent, output / "final.pt", replay, source_sha256, state, simulation)
        state.update(
            status="complete",
            phase="complete",
            frozen_source_sha256=frozen_source,
            final_checkpoint=str(output / "final.pt"),
        )
        _write_json(output / "state.json", state)
        print(json.dumps(state, allow_nan=False), flush=True)
        return 0
    except BaseException as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        _write_json(output / "state.json", state)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
