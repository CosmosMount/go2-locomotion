"""Evaluate Recovery on/off under paired target-domain perturbations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch

from common.checkpoint import load_checkpoint
from common.output import prepare_output_dir
from onrobot.recoveryrl.__main__ import _configuration
from onrobot.recoveryrl.agent import AgentConfig, RecoveryAgent
from onrobot.recoveryrl.environment import MujocoRecoveryEnv


@dataclass(frozen=True)
class Scenario:
    name: str
    qvel_index: int | None
    impulse: float


SCENARIOS = (
    Scenario("nominal", None, 0.0),
    Scenario("backward_push", 0, -1.0),
    Scenario("lateral_push_left", 1, 1.0),
    Scenario("lateral_push_right", 1, -1.0),
    Scenario("roll_left", 3, 2.5),
    Scenario("roll_right", 3, -2.5),
    Scenario("pitch_forward", 4, 2.5),
    Scenario("pitch_backward", 4, -2.5),
)


def _checkpoint_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("checkpoint label cannot be empty")
    return label, Path(path)


def _load_agent(path: Path, config: dict, device: str) -> RecoveryAgent:
    payload = load_checkpoint(path, map_location="cpu")
    metadata = payload["metadata"]
    simulation = config["simulation"]
    adaptation = config["adaptation"]
    if metadata["task"] == "safe":
        contract = config["checkpoint_contract"]
        return RecoveryAgent.from_source_checkpoint(
            path,
            source_task=contract["source_task"],
            policy_view=contract["policy_view"],
            default_joint_position=simulation["default_joint_position"],
            action_scale=simulation["action_scale"],
            seed=adaptation["seed"],
            batch_size=adaptation["batch_size"],
            epsilon=adaptation["risk_threshold"],
            device=device,
        )[0]
    if metadata["task"] != "recovery-rl":
        raise ValueError(f"unsupported checkpoint task: {metadata['task']!r}")
    bundle = payload["state"]
    agent_config = AgentConfig.from_checkpoint(
        bundle["config"],
        seed=adaptation["seed"],
        batch_size=adaptation["batch_size"],
        epsilon=adaptation["risk_threshold"],
    )
    agent = RecoveryAgent(agent_config, device)
    for name, module in agent.modules.items():
        module.load_state_dict(bundle["modules"][name], strict=True)
        module.eval()
    agent.freeze_source_modules()
    return agent


def _rollout(
    agent: RecoveryAgent,
    simulation: dict,
    scenario: Scenario,
    *,
    seed: int,
    use_recovery: bool,
    deterministic: bool,
    perturbation_step: int,
    max_steps: int,
    impulse_scale: float,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    environment = MujocoRecoveryEnv(simulation, seed=seed)
    observation = environment.reset(seed=seed)
    reward_sum = velocity_squared_error = risk_sum = 0.0
    recoveries = failure = 0
    first_recovery_after_perturbation = None
    failure_after_perturbation = None
    target_velocity = float(simulation["target_velocity"])
    try:
        for step in range(max_steps):
            if step == perturbation_step and scenario.qvel_index is not None:
                environment.data.qvel[scenario.qvel_index] += (
                    scenario.impulse * impulse_scale
                )
                environment.mujoco.mj_forward(environment.model, environment.data)
            action = agent.act(
                observation,
                deterministic=deterministic,
                use_recovery=use_recovery,
            )
            observation, reward, terminated, truncated, info = environment.step(
                action["executed_action"]
            )
            reward_sum += reward
            velocity_squared_error += (
                info["forward_velocity"] - target_velocity
            ) ** 2
            risk_sum += action["risk"]
            recoveries += int(action["recovered"])
            if (
                action["recovered"]
                and step >= perturbation_step
                and first_recovery_after_perturbation is None
            ):
                first_recovery_after_perturbation = step - perturbation_step
            if terminated:
                failure = 1
                if step >= perturbation_step:
                    failure_after_perturbation = step - perturbation_step
                break
            if truncated:
                break
    finally:
        environment.close()
    steps = step + 1
    return {
        "seed": seed,
        "scenario": scenario.name,
        "impulse_scale": impulse_scale,
        "recovery": "on" if use_recovery else "off",
        "steps": steps,
        "failure": failure,
        "mean_reward": reward_sum / steps,
        "velocity_rmse": (velocity_squared_error / steps) ** 0.5,
        "mean_risk": risk_sum / steps,
        "recovery_rate": recoveries / steps,
        "first_recovery_delay": first_recovery_after_perturbation,
        "failure_delay": failure_after_perturbation,
    }


def _summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, float], list[dict]] = {}
    for row in rows:
        keys = (
            (row["checkpoint"], row["recovery"], row["scenario"], row["impulse_scale"]),
            (row["checkpoint"], row["recovery"], "ALL", row["impulse_scale"]),
        )
        for key in keys:
            groups.setdefault(key, []).append(row)
    summary = []
    for (checkpoint, recovery, scenario, impulse_scale), group in sorted(groups.items()):
        summary.append({
            "checkpoint": checkpoint,
            "recovery": recovery,
            "scenario": scenario,
            "impulse_scale": impulse_scale,
            "episodes": len(group),
            "failure_rate": float(np.mean([row["failure"] for row in group])),
            "mean_reward": float(np.mean([row["mean_reward"] for row in group])),
            "velocity_rmse": float(np.mean([row["velocity_rmse"] for row in group])),
            "mean_risk": float(np.mean([row["mean_risk"] for row in group])),
            "recovery_rate": float(np.mean([row["recovery_rate"] for row in group])),
        })
    return summary


def _paired_effects(rows: list[dict]) -> list[dict]:
    pairs: dict[tuple[str, str, float, int], dict[str, dict]] = {}
    for row in rows:
        key = (
            row["checkpoint"], row["scenario"],
            row["impulse_scale"], row["seed"],
        )
        pairs.setdefault(key, {})[row["recovery"]] = row
    groups: dict[tuple[str, str, float], list[tuple[dict, dict]]] = {}
    for (checkpoint, scenario, scale, _), pair in pairs.items():
        if set(pair) != {"on", "off"}:
            raise RuntimeError("missing paired Recovery on/off rollout")
        groups.setdefault((checkpoint, scenario, scale), []).append(
            (pair["off"], pair["on"])
        )
    result = []
    for (checkpoint, scenario, scale), group in sorted(groups.items()):
        off_failures = np.asarray([off["failure"] for off, _ in group], dtype=float)
        on_failures = np.asarray([on["failure"] for _, on in group], dtype=float)
        result.append({
            "checkpoint": checkpoint,
            "scenario": scenario,
            "impulse_scale": scale,
            "pairs": len(group),
            "failure_rate_off": float(off_failures.mean()),
            "failure_rate_on": float(on_failures.mean()),
            "absolute_failure_reduction": float((off_failures - on_failures).mean()),
            "off_failed_on_survived": int(np.sum((off_failures == 1) & (on_failures == 0))),
            "off_survived_on_failed": int(np.sum((off_failures == 0) & (on_failures == 1))),
            "mean_reward_delta_on_minus_off": float(np.mean([
                on["mean_reward"] - off["mean_reward"] for off, on in group
            ])),
        })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", type=_checkpoint_argument, required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=51000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--perturbation-step", type=int, default=100)
    parser.add_argument(
        "--impulse-scale", action="append", type=float,
        help="repeatable multiplier for every non-nominal impulse; defaults to 1.0",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.max_steps < 1:
        parser.error("--seeds and --max-steps must be positive")
    if not 0 <= args.perturbation_step < args.max_steps:
        parser.error("--perturbation-step must be within the rollout")
    impulse_scales = args.impulse_scale or [1.0]
    if any(scale <= 0 for scale in impulse_scales):
        parser.error("--impulse-scale values must be positive")

    output = prepare_output_dir(args.output_dir, temp_prefix="go2-recovery-ablation-")
    config = _configuration()
    rows = []
    for label, checkpoint in args.checkpoint:
        checkpoint = checkpoint.resolve()
        agent = _load_agent(checkpoint, config, args.device)
        for impulse_scale in impulse_scales:
            for scenario in SCENARIOS:
                for offset in range(args.seeds):
                    seed = args.seed_start + offset
                    for enabled in (False, True):
                        row = _rollout(
                            agent,
                            config["simulation"],
                            scenario,
                            seed=seed,
                            use_recovery=enabled,
                            deterministic=args.deterministic,
                            perturbation_step=args.perturbation_step,
                            max_steps=args.max_steps,
                            impulse_scale=impulse_scale,
                        )
                        row.update(checkpoint=label, checkpoint_path=str(checkpoint))
                        rows.append(row)

    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    paired_effects = _paired_effects(rows)
    (output / "paired_effects.json").write_text(
        json.dumps(paired_effects, indent=2, allow_nan=False) + "\n"
    )
    protocol = {
        "checkpoints": {label: str(path.resolve()) for label, path in args.checkpoint},
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "seeds": list(range(args.seed_start, args.seed_start + args.seeds)),
        "max_steps": args.max_steps,
        "perturbation_step": args.perturbation_step,
        "deterministic": args.deterministic,
        "impulse_scales": impulse_scales,
        "paired_recovery_on_off": True,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "output": str(output),
        "summary": summary,
        "paired_effects": paired_effects,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
