"""Target-domain MuJoCo model-free Recovery RL adaptation."""

from common.observation import GO2_JOINT_NAMES, RAW_OBSERVATION_ABI
from common.types import RobotState

from .agent import RecoveryAgent, ReplayBuffer


def main(argv: list[str] | None = None) -> int:
    """Dispatch without importing the CLI module during ``python -m`` startup."""
    from .__main__ import main as cli_main

    return cli_main(argv)


__all__ = [
    "RAW_OBSERVATION_ABI",
    "GO2_JOINT_NAMES",
    "RecoveryAgent",
    "ReplayBuffer",
    "RobotState",
    "main",
]
