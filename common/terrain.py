"""Small backend-neutral terrain specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerrainSpec:
    name: str
    proportion: float


RECOVERY_TERRAIN_MIX = (
    TerrainSpec("slope", 0.2), TerrainSpec("rough_slope", 0.2),
    TerrainSpec("stairs_down", 0.2), TerrainSpec("stairs_up", 0.2),
    TerrainSpec("gap", 0.2),
)


def validate_terrain_mix(specs) -> None:
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("terrain names must be unique")
    if any(spec.proportion <= 0 for spec in specs):
        raise ValueError("terrain proportions must be positive")
    if abs(sum(spec.proportion for spec in specs) - 1.0) > 1.0e-6:
        raise ValueError("terrain proportions must sum to one")
