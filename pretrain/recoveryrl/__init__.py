"""Source-domain Task SAC, Safety-Q, and MF recovery pretraining."""

from .core import (
    AgentConfig,
    POLICY_VIEW,
    RecoveryAgent,
    ReplayBuffer,
    safety_bellman_target,
)
from common.observation import (
    GO2_JOINT_NAMES,
    RAW_OBSERVATION_ABI,
    build_raw_observation_tensor,
    prepare_task_observation,
)
from .recovery import freeze_dependencies, update_recovery
from .safety import update_safety
from .task import update_task


def main(argv: list[str] | None = None) -> int:
    """Lazy package-level hook used by the repository runner."""

    from .__main__ import main as run

    return run(argv)

__all__ = [
    "AgentConfig",
    "GO2_JOINT_NAMES",
    "POLICY_VIEW",
    "RAW_OBSERVATION_ABI",
    "RecoveryAgent",
    "ReplayBuffer",
    "build_raw_observation_tensor",
    "freeze_dependencies",
    "main",
    "prepare_task_observation",
    "safety_bellman_target",
    "update_recovery",
    "update_safety",
    "update_task",
]
