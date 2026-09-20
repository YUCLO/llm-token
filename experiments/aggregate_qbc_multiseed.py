#!/usr/bin/env python3
"""Aggregate paired QBC diagnostic effects across independent training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("flip_all", "flip_interior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SEED:SPLIT=JSON",
        help="Paired comparison JSON, for example 23:test-clean=path.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_run(value: str) -> tuple[int, str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator:
        raise ValueError(f"Expected SEED:SPLIT=JSON, got {value!r}")
    raw_seed, separator, split = label.partition(":")
    if not separator or not split:
        raise ValueError(f"Expected SEED:SPLIT=JSON, got {value!r}")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return int(raw_seed), split, path


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "seeds": int(array.size),
        "mean_candidate_minus_baseline": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "improved_seeds": int((array < 0).sum()),
    }


def main() -> None:
    args = parse_args()
    per_split: dict[str, list[dict]] = {}
    seen: set[tuple[int, str]] = set()
    for value in args.run:
        seed, split, path = parse_run(value)
        key = (seed, split)
        if key in seen:
            raise ValueError(f"Duplicate seed/split: {key}")
        seen.add(key)
        payload = json.loads(path.read_text())
        cluster = payload["cluster_bootstrap"]
        row = {
            "seed": seed,
            "path": str(path.resolve()),
            "clusters": int(cluster["flip_all"]["clusters"]),
        }
        for metric in METRICS:
            row[metric] = float(cluster[metric]["candidate_minus_baseline"])
            row[f"{metric}_speaker_ci95"] = [
                float(cluster[metric]["ci95_low"]),
                float(cluster[metric]["ci95_high"]),
            ]
        per_split.setdefault(split, []).append(row)

    result = {"splits": {}}
    for split, rows in sorted(per_split.items()):
        rows.sort(key=lambda item: item["seed"])
        result["splits"][split] = {
            "per_seed": rows,
            "across_training_seeds": {
                metric: summarize([float(row[metric]) for row in rows]) for metric in METRICS
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
