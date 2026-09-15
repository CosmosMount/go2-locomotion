"""CLI for CEM search and selected-parameter evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from common.config import load_config as load_yaml_config
from common.observation import GO2_JOINT_NAMES, RAW_OBSERVATION_ABI
from common.output import prepare_output_dir

from .environment import Go2CpgEnvironment
from .search import cem_search, evaluate


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Go2 target-domain MuJoCo CEM-CPG stair ascent"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="evaluate parameters from a best.json produced by this repository",
    )
    parser.add_argument("--checkpoint", type=Path, help="CEM best.json to evaluate")
    parser.add_argument("--seed", type=int, help="override CEM seed")
    parser.add_argument("--device", choices=("cpu",), help="MuJoCo runs on CPU")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.evaluate_only and args.checkpoint is None:
        parser.error("--evaluate-only requires --checkpoint <best.json>")
    if args.checkpoint is not None and not args.evaluate_only:
        parser.error("--checkpoint is only valid with --evaluate-only")
    return args


def load_config(path: Path) -> dict:
    config = load_yaml_config(path)
    required = {"source", "observation", "simulation", "terrain", "cem", "evaluation"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"config missing sections: {sorted(missing)}")
    if config["observation"]["abi"] != RAW_OBSERVATION_ABI:
        raise ValueError("unsupported observation ABI")
    if tuple(config["observation"]["joint_order"]) != GO2_JOINT_NAMES:
        raise ValueError("config joint order does not match the canonical Go2 ABI")
    return config


def load_parameters(path: Path) -> np.ndarray:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = np.asarray(payload.get("parameters"), dtype=float)
    if parameters.shape != (8,) or not np.isfinite(parameters).all():
        raise ValueError("CEM checkpoint must contain a finite parameters[8]")
    return parameters


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = deepcopy(load_config(DEFAULT_CONFIG))
    if args.seed is not None:
        config["cem"]["seed"] = args.seed
    output = prepare_output_dir(args.output_dir, temp_prefix="go2-cem-cpg-")
    print(f"output={output}", flush=True)

    environment = Go2CpgEnvironment(output, config)
    if args.evaluate_only:
        parameters = load_parameters(args.checkpoint)
    else:
        best = cem_search(environment, config, output)
        parameters = np.asarray(best["parameters"], dtype=float)

    first_seed = int(config["evaluation"]["seed_start"])
    episode_count = int(config["evaluation"]["episodes"])
    evaluate(
        environment,
        parameters,
        range(first_seed, first_seed + episode_count),
        output,
    )
    return 0


if __name__ == "__main__":
    main()
