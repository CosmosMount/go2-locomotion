"""Run exactly one Go2 training or target-domain task."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


COMMAND_MODULES = {
    "pretrain-locomotion": "pretrain.locomotion",
    "pretrain-recovery": "pretrain.recovery",
    "pretrain-recoveryrl": "pretrain.recoveryrl",
    "cem-cpg": "onrobot.cem_cpg",
    "recovery-rl": "onrobot.recoveryrl",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in COMMAND_MODULES:
        command = commands.add_parser(name)
        command.add_argument("--seed", type=int)
        command.add_argument("--device")
        command.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
        command.add_argument("--output-dir", type=Path)
        if name == "cem-cpg":
            command.add_argument("--evaluate-only", action="store_true")
            command.add_argument("--checkpoint", type=Path)
        elif name == "recovery-rl":
            command.add_argument("--checkpoint", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    module = importlib.import_module(COMMAND_MODULES[args.command])
    forwarded = []
    for flag in ("seed", "device", "output_dir", "checkpoint"):
        value = getattr(args, flag, None)
        if value is not None:
            forwarded.extend(("--" + flag.replace("_", "-"), str(value)))
    if args.headless is not None:
        forwarded.append("--headless" if args.headless else "--no-headless")
    if getattr(args, "evaluate_only", False):
        forwarded.append("--evaluate-only")
    result = module.main(forwarded)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
