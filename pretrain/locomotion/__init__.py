"""Flat-terrain Go2 locomotion pretraining with Isaac Lab and PPO."""

from .config import CONFIG_PATH, LocomotionConfig, load_locomotion_config


def main(argv: list[str] | None = None) -> int:
    """Lazy package-level hook used by the repository runner."""

    from .__main__ import main as run

    return run(argv)


__all__ = ["CONFIG_PATH", "LocomotionConfig", "load_locomotion_config", "main"]
