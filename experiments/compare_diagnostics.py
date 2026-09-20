#!/usr/bin/env python3
"""Paired comparison of two QBC diagnostic runs on identical frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stats_utils import bootstrap_mean_ci, distribution_summary


KEYS = ["utterance_id", "frame_in_slice", "full_frame"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--cluster-key",
        choices=["utterance_id", "speaker_id"],
        default="utterance_id",
    )
    return parser.parse_args()


def cluster_bootstrap_difference(
    paired: pd.DataFrame,
    column_baseline: str,
    column_candidate: str,
    cluster_column: str,
    cluster_name: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    by_cluster = paired.groupby(cluster_column)[[column_baseline, column_candidate]].mean()
    differences = (by_cluster[column_candidate] - by_cluster[column_baseline]).to_numpy()
    estimate, low, high = bootstrap_mean_ci(
        differences,
        samples=samples,
        generator=np.random.default_rng(seed),
    )
    return {
        "candidate_minus_baseline": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "cluster_key": cluster_name,
        "clusters": int(differences.size),
    }


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.baseline)
    candidate = pd.read_csv(args.candidate)
    paired = base.merge(candidate, on=KEYS, suffixes=("_base", "_candidate"), validate="one_to_one")
    if len(paired) != len(base) or len(paired) != len(candidate):
        raise RuntimeError(
            f"Runs are not perfectly paired: base={len(base)}, candidate={len(candidate)}, paired={len(paired)}"
        )
    cluster_column = args.cluster_key
    if args.cluster_key != "utterance_id":
        cluster_column = f"{args.cluster_key}_base"
        candidate_cluster_column = f"{args.cluster_key}_candidate"
        if not (paired[cluster_column] == paired[candidate_cluster_column]).all():
            raise RuntimeError(f"Mismatched {args.cluster_key} labels between paired runs")

    base_flip = paired["flip_base"].astype(bool)
    candidate_flip = paired["flip_candidate"].astype(bool)
    base_only = int((base_flip & ~candidate_flip).sum())
    candidate_only = int((~base_flip & candidate_flip).sum())
    discordant = base_only + candidate_only
    mcnemar_p = float(binomtest(candidate_only, discordant, 0.5).pvalue) if discordant else 1.0

    metrics = [
        "flip",
        "unit_angular_displacement",
        "angular_radius",
        "gamma_angle",
        "certified_angle",
    ]
    if (
        "view_teacher_signed_margin_e_base" in paired.columns
        and "view_teacher_signed_margin_e_candidate" in paired.columns
    ):
        metrics.append("view_teacher_signed_margin_e")
    distributions = {
        metric: {
            args.baseline_name: distribution_summary(paired[f"{metric}_base"]),
            args.candidate_name: distribution_summary(paired[f"{metric}_candidate"]),
        }
        for metric in metrics
    }
    bootstrap = {
        "flip_all": cluster_bootstrap_difference(
            paired,
            "flip_base",
            "flip_candidate",
            cluster_column,
            args.cluster_key,
            args.bootstrap_samples,
            args.seed,
        )
    }
    interior = paired[paired["interior_base"].astype(bool) & paired["interior_candidate"].astype(bool)]
    bootstrap["flip_interior"] = cluster_bootstrap_difference(
        interior,
        "flip_base",
        "flip_candidate",
        cluster_column,
        args.cluster_key,
        args.bootstrap_samples,
        args.seed + 1,
    )

    result = {
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "counts": {
            "frames": int(len(paired)),
            "utterances": int(paired["utterance_id"].nunique()),
            "speakers": int(paired["speaker_id_base"].nunique()),
            "baseline_flip_candidate_stable": base_only,
            "baseline_stable_candidate_flip": candidate_only,
            "discordant_frames": discordant,
        },
        "paired_mcnemar_exact_p": mcnemar_p,
        "cluster_bootstrap": bootstrap,
        "distributions": distributions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
