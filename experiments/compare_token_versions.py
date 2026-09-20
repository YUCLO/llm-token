#!/usr/bin/env python3
"""Check whether two aligned codec-token releases differ by a simple ID map."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-cache", type=Path, required=True)
    parser.add_argument("--reference-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = torch.load(args.current_cache, map_location="cpu")["sequences"]
    table = pq.read_table(args.reference_parquet, columns=["id", "audio_codes"])
    reference = dict(zip(table["id"].to_pylist(), table["audio_codes"].to_pylist()))

    pair_counts: dict[int, Counter] = defaultdict(Counter)
    raw_matches = 0
    total = 0
    per_utterance: list[float] = []
    length_mismatches: list[dict] = []
    missing: list[str] = []
    for utterance_id, current_tensor in cache.items():
        if utterance_id not in reference:
            missing.append(utterance_id)
            continue
        current = current_tensor.numpy()
        old = np.asarray(reference[utterance_id])
        if current.size != old.size:
            length_mismatches.append(
                {
                    "utterance_id": utterance_id,
                    "current_frames": int(current.size),
                    "reference_frames": int(old.size),
                }
            )
            continue
        matches = int((current == old).sum())
        raw_matches += matches
        total += int(current.size)
        per_utterance.append(matches / current.size)
        for current_code, old_code in zip(current, old):
            pair_counts[int(current_code)][int(old_code)] += 1

    majority_matches = sum(max(counter.values()) for counter in pair_counts.values())
    top_targets = [counter.most_common(1)[0][0] for counter in pair_counts.values()]
    singleton_sources = sum(sum(counter.values()) == 1 for counter in pair_counts.values())
    frequent = sorted(
        (
            {
                "frames": sum(counter.values()),
                "current_code": current_code,
                "top_reference_codes": counter.most_common(5),
                "top_fraction": counter.most_common(1)[0][1] / sum(counter.values()),
            }
            for current_code, counter in pair_counts.items()
        ),
        key=lambda row: row["frames"],
        reverse=True,
    )
    summary = {
        "current_cache": str(args.current_cache.resolve()),
        "reference_parquet": str(args.reference_parquet.resolve()),
        "utterances_compared": len(per_utterance),
        "frames_compared": total,
        "missing_utterances": missing,
        "length_mismatches": length_mismatches,
        "raw_id_agreement": raw_matches / total,
        "per_utterance_agreement_median": float(np.median(per_utterance)),
        "per_utterance_agreement_min": float(np.min(per_utterance)),
        "per_utterance_agreement_max": float(np.max(per_utterance)),
        "observed_current_codes": len(pair_counts),
        "singleton_current_codes": singleton_sources,
        "majority_map_oracle_accuracy": majority_matches / total,
        "majority_map_target_collision_fraction": 1.0
        - len(set(top_targets)) / len(top_targets),
        "most_frequent_current_codes": frequent[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "most_frequent_current_codes"}, indent=2))


if __name__ == "__main__":
    main()
