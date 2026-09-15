"""Summarize paired Recovery RL adaptation logs in fixed windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _run_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use LABEL=RUN_DIR")
    label, path = value.split("=", 1)
    return label, Path(path)


def _window(rows: list[dict], start: int, stop: int) -> dict:
    values = rows[start:stop]
    return {
        "start": start + 1,
        "stop": stop,
        "mean_reward": float(np.mean([row["reward"] for row in values])),
        "mean_risk": float(np.mean([row["risk"] for row in values])),
        "recovery_rate": float(np.mean([row["recovery"] for row in values])),
        "failures": int(sum(row["failure"] for row in values)),
        "truncations": int(sum(row["truncated"] for row in values)),
    }


def _adaptation_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["phase"] == "adaptation":
                rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_run_argument, required=True)
    parser.add_argument("--window", type=int, default=20000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.window < 1:
        parser.error("--window must be positive")

    result = {}
    for label, directory in args.run:
        trace = directory / "training.jsonl"
        rows = _adaptation_rows(trace)
        result[label] = [
            _window(rows, start, min(start + args.window, len(rows)))
            for start in range(0, len(rows), args.window)
        ]
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
