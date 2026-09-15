"""Target-domain MuJoCo CEM-CPG package for Go2 stair ascent."""

from .controller import PARAMETER_NAMES, cpg_control
from common.observation import RAW_OBSERVATION_ABI


def main(argv: list[str] | None = None) -> int:
    """Lazy package-level hook used by the repository runner."""

    from .__main__ import main as run

    return run(argv)

__all__ = [
    "PARAMETER_NAMES",
    "RAW_OBSERVATION_ABI",
    "cpg_control",
    "main",
]
