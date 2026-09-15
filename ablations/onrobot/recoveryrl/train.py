"""Run one arm of the paired Recovery RL training ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.output import PROJECT_ROOT
from onrobot.recoveryrl import agent as agent_module
from onrobot.recoveryrl import __main__ as adaptation_cli


def _force_task_only() -> None:
    original_act = agent_module.RecoveryAgent.act

    def task_only_act(self, observation, *, deterministic=False, use_recovery=True):
        return original_act(
            self,
            observation,
            deterministic=deterministic,
            use_recovery=False,
        )

    agent_module.RecoveryAgent.act = task_only_act


def _override_budget(args) -> None:
    if all(value is None for value in (
        args.warmup_collection_interactions,
        args.reward_critic_warmup_updates,
        args.adaptation_interactions,
    )):
        return
    original_configuration = adaptation_cli._configuration

    def ablation_configuration():
        config = original_configuration()
        adaptation = config["adaptation"]
        for name in (
            "warmup_collection_interactions",
            "reward_critic_warmup_updates",
            "adaptation_interactions",
        ):
            value = getattr(args, name)
            if value is not None:
                adaptation[name] = value
        return config

    adaptation_cli._configuration = ablation_configuration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("full", "task-only"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument("--warmup-collection-interactions", type=int)
    parser.add_argument("--reward-critic-warmup-updates", type=int)
    parser.add_argument("--adaptation-interactions", type=int)
    parser.add_argument(
        "--target-velocity", type=float,
        help="adaptation reward target in m/s; must exceed the 1.0 m/s reference",
    )
    args = parser.parse_args(argv)
    budget_values = (
        args.warmup_collection_interactions,
        args.reward_critic_warmup_updates,
        args.adaptation_interactions,
    )
    if any(value is not None and value < 1 for value in budget_values):
        parser.error("ablation budget overrides must be positive")

    if args.arm == "task-only":
        _force_task_only()
    _override_budget(args)

    forwarded = [
        "--checkpoint", str(args.checkpoint),
        "--output-dir", str(args.output_dir),
        "--device", args.device,
        "--seed", str(args.seed),
        "--headless",
    ]
    if args.target_velocity is not None:
        forwarded.extend(["--target-velocity", str(args.target_velocity)])
    result = adaptation_cli.main(forwarded)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    protocol_path = output_dir.resolve() / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["ablation"] = {
        "arm": args.arm,
        "recovery_enabled_during_adaptation": args.arm == "full",
        "paired_seed": args.seed,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "budget_override": {
            "warmup_collection_interactions": args.warmup_collection_interactions,
            "reward_critic_warmup_updates": args.reward_critic_warmup_updates,
            "adaptation_interactions": args.adaptation_interactions,
            "target_velocity": args.target_velocity,
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
