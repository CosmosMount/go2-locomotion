"""Run the single supported Go2 locomotion pretraining job."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable the Isaac Sim window",
    )
    parser.add_argument("--device", help="override the simulation device")
    parser.add_argument("--seed", type=int, help="override the random seed")
    parser.add_argument(
        "--output-dir", type=Path, help="override the generated checkpoint directory"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .config import load_locomotion_config

    config = load_locomotion_config().with_overrides(
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )
    config.validate(check_assets=True)

    try:
        from rl import ActionSpaceType  # noqa: F401 - explicit submodule preflight
    except ImportError as exc:
        raise RuntimeError(
            "initialize the target rl submodule at commit "
            "9a85f348dcf31aca487bb844c319a012790edf20 before training"
        ) from exc
    from .trainer import LocomotionTrainer

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Isaac Sim requires a working NVIDIA driver even when physics uses --device cpu"
        )

    from common.isaac import launch_app

    app = launch_app(headless=args.headless, device=config.device)
    try:
        LocomotionTrainer(config).run()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
