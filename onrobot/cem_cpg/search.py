"""Episode-level CEM and frozen-parameter evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .controller import PARAMETER_NAMES, cpg_control
from .environment import Go2CpgEnvironment


LOW = np.array([-1.0, -0.5, 0.0, 0.0, -0.4, 0.0, 0.0, -0.5])
HIGH = np.array([1.0, 1.0, 1.0, 1.0, 0.4, 1.5, 1.0, 0.5])
INITIAL_PARAMETERS = np.array([0.0, 0.0, 0.2, 0.4, 0.0, 0.6, 0.6, 0.0])


def run_episode(
    environment: Go2CpgEnvironment,
    parameters: np.ndarray,
    seed: int,
) -> dict:
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape != (8,) or not np.isfinite(parameters).all():
        raise ValueError("parameters must be a finite 8-vector")
    observation = environment.reset(seed)
    while True:
        command = cpg_control(
            observation,
            environment.phase,
            environment.imu_odometry,
            parameters,
        )
        observation, _, terminated, truncated, info = environment.step(command)
        if terminated or truncated:
            break
    return {"seed": int(seed), "parameters": parameters.tolist(), **info}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def cem_search(environment: Go2CpgEnvironment, config: dict, output: Path) -> dict:
    cem = config["cem"]
    generations = int(cem["generations"])
    population = int(cem["population"])
    elite_count = int(cem["elite_count"])
    random_candidates = int(cem["random_candidates"])
    if generations < 1 or population < 4:
        raise ValueError("CEM requires at least one generation and four candidates")
    if not 1 <= elite_count <= population - random_candidates:
        raise ValueError("invalid CEM elite_count")

    rng = np.random.default_rng(int(cem["seed"]))
    mean = (INITIAL_PARAMETERS - LOW) / (HIGH - LOW)
    standard_deviation = np.full(8, float(cem["initial_std"]))
    minimum_std = float(cem["minimum_std"])
    mean_update = float(cem["mean_update"])
    std_update = float(cem["std_update"])
    all_rows: list[dict] = []
    trials_path = output / "trials.jsonl"

    for generation in range(generations):
        candidates = np.clip(
            rng.normal(mean, standard_deviation, (population, 8)), 0.0, 1.0
        )
        candidates[0] = mean
        if random_candidates:
            candidates[-random_candidates:] = rng.uniform(
                0.0, 1.0, (random_candidates, 8)
            )
        generation_rows = []
        for normalized in candidates:
            parameters = LOW + normalized * (HIGH - LOW)
            seed = int(cem["seed"]) + len(all_rows)
            row = run_episode(environment, parameters, seed)
            row["generation"] = generation
            generation_rows.append(row)
            all_rows.append(row)
            with trials_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        elite = sorted(
            generation_rows, key=lambda item: item["score"], reverse=True
        )[:elite_count]
        normalized_elite = (
            np.asarray([row["parameters"] for row in elite]) - LOW
        ) / (HIGH - LOW)
        mean = (1.0 - mean_update) * mean + mean_update * normalized_elite.mean(0)
        standard_deviation = np.maximum(
            minimum_std,
            (1.0 - std_update) * standard_deviation
            + std_update * normalized_elite.std(0),
        )

    best = max(all_rows, key=lambda item: item["score"])
    best["parameter_names"] = list(PARAMETER_NAMES)
    _write_json(output / "best.json", best)
    return best


def evaluate(
    environment: Go2CpgEnvironment,
    parameters: np.ndarray,
    seeds: Iterable[int],
    output: Path,
) -> list[dict]:
    rows = []
    for seed in seeds:
        row = run_episode(environment, parameters, int(seed))
        rows.append(row)
        print(json.dumps(row), flush=True)
    _write_json(output / "evaluation.json", rows)
    successes = sum(row["success"] and not row["fall"] for row in rows)
    print(f"FINAL {successes} {len(rows)}", flush=True)
    return rows
