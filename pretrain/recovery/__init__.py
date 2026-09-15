"""Isaac Lab plain-PPO fall recovery training for Unitree Go2."""


def main(argv=None):
    """Run recovery training without importing Isaac Sim at package import time."""

    from .__main__ import main as _main

    return _main(argv)


__all__ = ["main"]
