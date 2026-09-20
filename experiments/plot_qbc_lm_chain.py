#!/usr/bin/env python3
"""Plot the joint Gamma -> token-flip -> common-context LM-penalty result."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.frames_csv.open()))
    gamma = np.asarray([float(row["gamma_angle"]) for row in rows])
    flip = np.asarray([int(row["flip"]) for row in rows], dtype=bool)
    delta = np.asarray([float(row["delta_nll_common_context"]) for row in rows])

    edges = [0.0, 1.0, 2.0, 4.0, 8.0, np.inf]
    labels = ["<1", "1-2", "2-4", "4-8", ">=8"]
    flip_rate: list[float] = []
    mean_delta: list[float] = []
    counts: list[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (gamma >= left) & (gamma < right)
        counts.append(int(mask.sum()))
        flip_rate.append(float(flip[mask].mean()))
        mean_delta.append(float(delta[mask].mean()))

    x = np.arange(len(labels))
    fig, axis_flip = plt.subplots(figsize=(8.5, 4.8))
    bars = axis_flip.bar(x, flip_rate, width=0.62, color="#4c78a8", alpha=0.82)
    axis_flip.set_ylabel("Token flip rate", color="#2f5f8f")
    axis_flip.set_ylim(0, 1)
    axis_flip.set_xticks(x, labels)
    axis_flip.set_xlabel(r"Boundary stress $\Gamma=D/R$")
    axis_delta = axis_flip.twinx()
    axis_delta.plot(x, mean_delta, color="#e45756", marker="o", linewidth=2.2)
    axis_delta.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis_delta.set_ylabel(r"Mean common-context $\Delta$NLL (nats)", color="#b33c3b")
    for bar, count in zip(bars, counts):
        axis_flip.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(args.output_dir / "gamma_flip_lm_penalty.png", dpi=200)
    plt.close(fig)

    flip_delta = delta[flip]
    low, high = np.quantile(flip_delta, [0.01, 0.99])
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.hist(np.clip(flip_delta, low, high), bins=60, color="#72b7b2", alpha=0.9)
    axis.axvline(0, color="black", linewidth=1)
    axis.axvline(flip_delta.mean(), color="#e45756", linewidth=2, label=f"mean={flip_delta.mean():.3f}")
    axis.set_xlabel(r"$\Delta$NLL on token-flip frames (nats)")
    axis.set_ylabel("Frames")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "flip_delta_nll_histogram.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
