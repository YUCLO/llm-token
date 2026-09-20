#!/usr/bin/env python3
"""Compare matched token-LM runs against train-unigram validation CE."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_small_token_lm import stable_validation_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=TOKENS_DIR=SUMMARY_JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=20480)
    parser.add_argument("--val-permille", type=int, default=20)
    parser.add_argument("--unigram-alpha", type=float, default=1.0)
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path, Path]:
    pieces = value.split("=", 2)
    if len(pieces) != 3:
        raise ValueError(f"Expected NAME=TOKENS_DIR=SUMMARY_JSON, got {value}")
    name, token_dir, summary = pieces
    return name, Path(token_dir), Path(summary)


def unigram_validation_ce(
    token_dir: Path,
    vocab_size: int,
    val_permille: int,
    alpha: float,
) -> dict[str, float | int]:
    boundary = vocab_size
    counts = np.full(vocab_size + 1, float(alpha), dtype=np.float64)
    validation_sequences: list[np.ndarray] = []
    train_tokens = 0
    validation_tokens = 0
    train_utterances = 0
    validation_utterances = 0
    for path in sorted(token_dir.glob("part-*.parquet")):
        table = pq.read_table(path, columns=["id", "audio_codes"])
        for row in table.to_pylist():
            codes = np.asarray(row["audio_codes"], dtype=np.int64)
            sequence = np.concatenate([np.asarray([boundary]), codes])
            if stable_validation_assignment(str(row["id"]), val_permille=val_permille):
                validation_sequences.append(sequence)
                validation_tokens += int(sequence.size)
                validation_utterances += 1
            else:
                counts += np.bincount(sequence, minlength=vocab_size + 1)
                train_tokens += int(sequence.size)
                train_utterances += 1
    probabilities = counts / counts.sum()
    log_probabilities = np.log(probabilities)
    total_nll = 0.0
    for sequence in validation_sequences:
        total_nll -= float(log_probabilities[sequence].sum())
    ce = total_nll / validation_tokens
    return {
        "ce_nats": ce,
        "perplexity": math.exp(ce),
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "train_utterances": train_utterances,
        "validation_utterances": validation_utterances,
    }


def main() -> None:
    args = parse_args()
    parsed = [parse_run(value) for value in args.run]
    result = {"runs": {}}
    for name, token_dir, summary_path in parsed:
        if not token_dir.is_dir() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing inputs for {name}")
        summary = json.loads(summary_path.read_text())
        lm_ce = float(summary["training"]["best_validation_loss_nats"])
        unigram = unigram_validation_ce(
            token_dir,
            args.vocab_size,
            args.val_permille,
            args.unigram_alpha,
        )
        gain = float(unigram["ce_nats"]) - lm_ce
        result["runs"][name] = {
            "token_dir": str(token_dir.resolve()),
            "summary": str(summary_path.resolve()),
            "unigram_validation": unigram,
            "lm_validation_ce_nats": lm_ce,
            "lm_validation_ppl": math.exp(lm_ce),
            "modeling_gain_nats": gain,
            "relative_modeling_gain": gain / float(unigram["ce_nats"]),
            "codebook": summary["data"]["codebook"],
        }
    names = [name for name, _, _ in parsed]
    if len(names) >= 2:
        baseline = result["runs"][names[0]]
        for name in names[1:]:
            candidate = result["runs"][name]
            result.setdefault("candidate_minus_baseline", {})[name] = {
                "lm_ce_nats": candidate["lm_validation_ce_nats"] - baseline["lm_validation_ce_nats"],
                "lm_ppl_relative": candidate["lm_validation_ppl"] / baseline["lm_validation_ppl"] - 1.0,
                "unigram_ce_nats": candidate["unigram_validation"]["ce_nats"] - baseline["unigram_validation"]["ce_nats"],
                "modeling_gain_nats": candidate["modeling_gain_nats"] - baseline["modeling_gain_nats"],
                "relative_modeling_gain": candidate["relative_modeling_gain"] - baseline["relative_modeling_gain"],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
