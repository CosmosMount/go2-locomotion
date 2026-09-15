"""Command-line entry point for Go2 recovery PPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.isaac import launch_app

from .config import load_recovery_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Resume a pretrain-recovery checkpoint.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_recovery_config(
        seed=args.seed,
        device=args.device,
        headless=args.headless,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
    )
    simulation_app = launch_app(headless=cfg["headless"], device=cfg["device"])
    try:
        # Isaac Lab modules must only be imported after AppLauncher starts Kit.
        from .environment import RecoveryEnv, build_env_cfg
        from .trainer import train

        env = RecoveryEnv(build_env_cfg(cfg), task_cfg=cfg)
        try:
            train(env, cfg)
        finally:
            env.close()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
