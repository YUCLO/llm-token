#!/usr/bin/env python3
"""Paired utterance-bootstrap comparison of common-context LM runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stats_utils import bootstrap_mean_ci


KEYS = ["utterance_id", "frame_in_slice", "full_frame"]
METRICS = [
    "clean_nll_full_vocab",
    "slice_nll_full_vocab",
    "delta_nll_common_context",
    "flip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--cluster-key",
        choices=["utterance_id", "speaker_id"],
        default="utterance_id",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = pd.read_csv(args.baseline)
    candidate = pd.read_csv(args.candidate)
    paired = baseline.merge(candidate, on=KEYS, suffixes=("_base", "_candidate"), validate="one_to_one")
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise RuntimeError("Runs are not perfectly frame-paired")
    cluster_column = args.cluster_key
    if args.cluster_key != "utterance_id":
        cluster_column = f"{args.cluster_key}_base"
        candidate_cluster_column = f"{args.cluster_key}_candidate"
        if not (paired[cluster_column] == paired[candidate_cluster_column]).all():
            raise RuntimeError(f"Mismatched {args.cluster_key} labels between paired runs")

    rng = np.random.default_rng(args.seed)
    result = {
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "frames": int(len(paired)),
        "utterances": int(paired["utterance_id"].nunique()),
        "speakers": int(paired["speaker_id_base"].nunique()),
        "cluster_key": args.cluster_key,
        "metrics": {},
    }
    for metric in METRICS:
        base_col = f"{metric}_base"
        candidate_col = f"{metric}_candidate"
        grouped = paired.groupby(cluster_column)[[base_col, candidate_col]].mean()
        differences = (grouped[candidate_col] - grouped[base_col]).to_numpy()
        estimate, low, high = bootstrap_mean_ci(
            differences,
            samples=args.bootstrap_samples,
            generator=rng,
        )
        result["metrics"][metric] = {
            args.baseline_name: float(paired[base_col].mean()),
            args.candidate_name: float(paired[candidate_col].mean()),
            "candidate_minus_baseline": estimate,
            "ci95_low": low,
            "ci95_high": high,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
