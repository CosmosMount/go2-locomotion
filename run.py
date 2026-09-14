"""Unified command-line entry point for Go2 training and deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser('export', help='export a trusted local checkpoint to TorchScript')
    export_parser.add_argument('checkpoint', type=Path)
    export_parser.add_argument('--output', type=Path)

    train_parser = commands.add_parser("train", help="train a PPO policy")
    train_parser.add_argument(
        "--headless", action="store_true", help="disable the Isaac Sim window"
    )
    train_parser.add_argument("--device", help="simulation device, e.g. cuda:0 or cpu")
    train_parser.add_argument("--seed", type=int, help="random seed")
    train_parser.add_argument("--num-envs", type=int, help="parallel environments")
    train_parser.add_argument(
        "--max-iterations", type=int, help="number of PPO iterations"
    )
    train_parser.add_argument(
        "--checkpoint-interval", type=int, help="iterations between checkpoints"
    )
    train_parser.add_argument("--log-root", type=Path, help="training output root")
    train_parser.add_argument("--terrain", choices=("flat", "rough"), default="flat")

    deploy_parser = commands.add_parser(
        "deploy", help="run an exported policy through SDK2 DDS"
    )
    deploy_parser.add_argument("policy", type=Path, help="TorchScript actor path")
    deploy_parser.add_argument("--domain", type=int, help="DDS domain (0 is refused)")
    deploy_parser.add_argument("--iface", help="DDS network interface")
    eval_parser = commands.add_parser("evaluate", help="evaluate a deterministic TorchScript actor in Isaac Lab")
    eval_parser.add_argument("policy", type=Path)
    eval_parser.add_argument("--headless", action="store_true")
    eval_parser.add_argument("--device")
    eval_parser.add_argument("--num-envs", type=int, default=16)
    eval_parser.add_argument("--seed", type=int)
    eval_parser.add_argument("--steps", type=int, default=1000)
    eval_parser.add_argument("--vx", type=float, default=0.5)
    eval_parser.add_argument("--yaw", type=float, default=0.0)
    eval_parser.add_argument("--vy", type=float, default=0.0)
    eval_parser.add_argument("--terrain", choices=("flat", "rough"), default="flat")
    eval_parser.set_defaults(max_iterations=None, checkpoint_interval=None, log_root=None)
    return parser


class Runner:
    """Build one shared config and dispatch the selected application mode."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = self._build_config(args)

    @staticmethod
    def _build_config(args: argparse.Namespace) -> Config:
        config = Config()
        if args.command in ("train", "evaluate"):
            config.terrain = args.terrain
            if args.device is not None:
                config.device = args.device
            if args.seed is not None:
                config.seed = args.seed
            if args.num_envs is not None:
                config.num_envs = args.num_envs
            if args.max_iterations is not None:
                config.max_iterations = args.max_iterations
            if args.checkpoint_interval is not None:
                config.checkpoint_interval = args.checkpoint_interval
            if args.log_root is not None:
                config.log_root = args.log_root.expanduser()
        elif args.command == "deploy":
            if args.domain is not None:
                config.domain = args.domain
            if args.iface is not None:
                config.interface = args.iface
        else:
            raise ValueError(f"unsupported command: {args.command}")
        config.validate()
        return config

    def run(self) -> None:
        if self.args.command in ("train", "evaluate"):
            self._run_training()
        else:
            self._run_deployment()

    def _run_training(self) -> None:
        # AppLauncher must exist before importing modules that use Isaac Lab APIs.
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            {
                "headless": self.args.headless,
                "device": self.config.device,
            }
        ).app
        try:
            from train import Train

            if self.args.command == "evaluate":
                from evaluate import evaluate

                evaluate(self.config, self.args.policy, self.args.steps, self.args.vx, self.args.yaw, self.args.vy)
            else:
                Train(self.config).run()
        finally:
            app.close()

    def _run_deployment(self) -> None:
        from deploy import Deploy

        Deploy(self.config, self.args.policy).run()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == 'export':
        from policy_io import export_checkpoint
        export_checkpoint(args.checkpoint, args.output)
        return 0
    Runner(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
