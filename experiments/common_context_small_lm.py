#!/usr/bin/env python3
"""Common-context substitution NLL using the version-matched compact token LM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common_context_nll import (
    load_diagnostic_frames,
    radius_quartiles,
    safe_spearman,
    summarize_subset,
)
from experiments.io_utils import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-csv",
        type=Path,
        default=ROOT
        / "artifacts/qbc_diagnostic_v2/llm_codec_test_clean_32speakers/frames.csv",
    )
    parser.add_argument(
        "--sequence-cache",
        type=Path,
        default=ROOT
        / "artifacts/qbc_diagnostic_v2/llm_codec_test_clean_common_context/full_token_sequences.pt",
    )
    parser.add_argument(
        "--lm-checkpoint",
        type=Path,
        default=ROOT / "artifacts/small_lm_current_trainclean100_seed17/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "artifacts/qbc_diagnostic_v2/llm_codec_test_clean_common_context_small_lm",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-utterances", type=int, default=None)
    return parser.parse_args()


def make_context(
    full_codes: torch.Tensor,
    target_frame: int,
    boundary_token: int,
    context_length: int,
) -> torch.Tensor:
    if target_frame <= 0:
        prefix = torch.tensor([boundary_token], dtype=torch.long)
    else:
        prefix = torch.cat(
            [torch.tensor([boundary_token], dtype=torch.long), full_codes[:target_frame].long()]
        )
    return prefix[-context_length:]


def gamma_bin_summary(rows: list[dict]) -> list[dict]:
    edges = [0.0, 1.0, 2.0, 4.0, 8.0, math.inf]
    output: list[dict] = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = [
            row for row in rows if left <= float(row["gamma_angle"]) < right
        ]
        delta = np.asarray(
            [row["delta_nll_common_context"] for row in selected], dtype=np.float64
        )
        flips = np.asarray([row["flip"] for row in selected], dtype=np.float64)
        output.append(
            {
                "left": left,
                "right": right,
                "frames": len(selected),
                "flip_rate": float(flips.mean()) if flips.size else None,
                "mean_delta_nll_nats": float(delta.mean()) if delta.size else None,
            }
        )
    return output


@torch.inference_mode()
def score_utterance(
    model: GPT2LMHeadModel,
    full_codes: torch.Tensor,
    rows: list[dict],
    boundary_token: int,
    context_length: int,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    contexts = [
        make_context(full_codes, int(row["full_frame"]), boundary_token, context_length)
        for row in rows
    ]
    scored: list[dict] = []
    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        batch_contexts = contexts[batch_start : batch_start + batch_size]
        lengths = torch.tensor([item.numel() for item in batch_contexts], dtype=torch.long)
        width = int(lengths.max())
        input_ids = torch.full(
            (len(batch_contexts), width), boundary_token, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, context in enumerate(batch_contexts):
            input_ids[index, width - context.numel() :] = context
            attention_mask[index, width - context.numel() :] = 1
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.clamp_(min=0)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            hidden = model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1, :]
            logits = model.lm_head(hidden).float()
        clean = torch.tensor(
            [row["full_token"] for row in batch_rows], device=device, dtype=torch.long
        )
        view = torch.tensor(
            [row["slice_token"] for row in batch_rows], device=device, dtype=torch.long
        )
        index = torch.arange(logits.shape[0], device=device)
        clean_logits = logits[index, clean]
        view_logits = logits[index, view]
        full_norm = torch.logsumexp(logits, dim=-1)
        audio_logits = logits[:, :boundary_token]
        audio_norm = torch.logsumexp(audio_logits, dim=-1)
        clean_rank = 1 + (audio_logits > clean_logits[:, None]).sum(dim=-1)
        view_rank = 1 + (audio_logits > view_logits[:, None]).sum(dim=-1)
        top1 = torch.argmax(audio_logits, dim=-1)
        values = {
            "clean_logit": clean_logits,
            "slice_logit": view_logits,
            "delta_nll_common_context": clean_logits - view_logits,
            "clean_nll_full_vocab": full_norm - clean_logits,
            "slice_nll_full_vocab": full_norm - view_logits,
            "clean_nll_audio_only": audio_norm - clean_logits,
            "slice_nll_audio_only": audio_norm - view_logits,
            "clean_rank_audio": clean_rank,
            "slice_rank_audio": view_rank,
            "audio_top1_code": top1,
        }
        cpu = {key: value.detach().cpu().numpy() for key, value in values.items()}
        for local_index, source in enumerate(batch_rows):
            item = dict(source)
            item["prefix_mode"] = "version_matched_small_lm_clean_prefix"
            item["clean_prefix_tokens_used"] = int(lengths[local_index])
            for key in (
                "clean_logit",
                "slice_logit",
                "delta_nll_common_context",
                "clean_nll_full_vocab",
                "slice_nll_full_vocab",
                "clean_nll_audio_only",
                "slice_nll_audio_only",
            ):
                item[key] = float(cpu[key][local_index])
            for key in ("clean_rank_audio", "slice_rank_audio", "audio_top1_code"):
                item[key] = int(cpu[key][local_index])
            item["lm_prefers_clean"] = int(item["delta_nll_common_context"] > 0.0)
            scored.append(item)
    return scored


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for path in (args.frames_csv, args.sequence_cache, args.lm_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows, utterance_order = load_diagnostic_frames(
        args.frames_csv, args.max_utterances
    )
    by_utterance: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_utterance[str(row["utterance_id"])].append(row)
    for utterance_id in utterance_order:
        by_utterance[utterance_id].sort(key=lambda row: row["frame_in_slice"])

    sequence_payload = torch.load(args.sequence_cache, map_location="cpu")
    sequences = sequence_payload["sequences"]
    checkpoint = torch.load(args.lm_checkpoint, map_location="cpu")
    config = GPT2Config.from_dict(checkpoint["config"])
    model = GPT2LMHeadModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    boundary_token = int(checkpoint["boundary_token"])
    if boundary_token != config.vocab_size - 1:
        raise RuntimeError("Unexpected boundary-token/vocabulary layout")

    mismatches = 0
    scored_rows: list[dict] = []
    for index, utterance_id in enumerate(utterance_order, start=1):
        full_codes = sequences[utterance_id].long()
        for row in by_utterance[utterance_id]:
            mismatches += int(
                int(full_codes[int(row["full_frame"])]) != int(row["full_token"])
            )
        scored = score_utterance(
            model,
            full_codes,
            by_utterance[utterance_id],
            boundary_token,
            config.n_positions,
            args.batch_size,
            device,
        )
        scored_rows.extend(scored)
        flip_delta = [row["delta_nll_common_context"] for row in scored if row["flip"]]
        print(
            f"[lm {index}/{len(utterance_order)}] {utterance_id} "
            f"frames={len(scored)} flips={len(flip_delta)} "
            f"mean_flip_delta={np.mean(flip_delta) if flip_delta else math.nan:.4f}"
        )
    if mismatches:
        raise RuntimeError(f"Diagnostic/cache token mismatch at {mismatches} frames")

    output_csv = args.output_dir / "frames_common_context_nll.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scored_rows)

    flips = [row for row in scored_rows if row["flip"]]
    interior_flips = [row for row in flips if row["interior"]]
    flip_delta = np.asarray(
        [row["delta_nll_common_context"] for row in flips], dtype=np.float64
    )
    all_delta = np.asarray(
        [row["delta_nll_common_context"] for row in scored_rows], dtype=np.float64
    )
    all_radius = np.asarray(
        [row["angular_radius"] for row in scored_rows], dtype=np.float64
    )
    all_displacement = np.asarray(
        [row["unit_angular_displacement"] for row in scored_rows], dtype=np.float64
    )
    all_gamma = np.asarray([row["gamma_angle"] for row in scored_rows], dtype=np.float64)
    flip_radius = np.asarray([row["angular_radius"] for row in flips], dtype=np.float64)
    flip_gamma = np.asarray([row["gamma_angle"] for row in flips], dtype=np.float64)
    positive_flips = int((flip_delta > 0).sum())
    negative_flips = int((flip_delta < 0).sum())
    utterance_flip_means = [
        float(
            np.mean(
                [
                    row["delta_nll_common_context"]
                    for row in flips
                    if row["utterance_id"] == utterance_id
                ]
            )
        )
        for utterance_id in utterance_order
    ]
    summary = {
        "definition": {
            "delta_nll_common_context": (
                "NLL(slice token | full clean prefix) - "
                "NLL(full token | full clean prefix), in nats"
            ),
            "positive_means": "the version-matched fixed LM prefers the full token",
            "claim_scope": "mechanistic observational evidence, not a causal estimate",
            "context_policy": f"rolling clean prefix up to {config.n_positions} tokens",
        },
        "configuration": {
            "frames_csv": str(args.frames_csv.resolve()),
            "sequence_cache": str(args.sequence_cache.resolve()),
            "lm_checkpoint": str(args.lm_checkpoint.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "device": str(device),
            "batch_size": args.batch_size,
        },
        "provenance": {
            "lm_checkpoint_sha256": file_sha256(args.lm_checkpoint),
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "lm_validation_history": checkpoint.get("history"),
            "lm_training_data": checkpoint.get("data"),
        },
        "verification": {"diagnostic_full_token_mismatches": mismatches},
        "counts": {
            "utterances": len(utterance_order),
            "speakers": len({row["speaker_id"] for row in scored_rows}),
            "frames": len(scored_rows),
            "flips": len(flips),
            "interior_flips": len(interior_flips),
        },
        "all_frames": summarize_subset(
            scored_rows, args.bootstrap_replicates, args.seed
        ),
        "flip_frames": summarize_subset(flips, args.bootstrap_replicates, args.seed + 1),
        "interior_flip_frames": summarize_subset(
            interior_flips, args.bootstrap_replicates, args.seed + 2
        ),
        "relations_on_flip_frames": {
            "negative_radius_vs_delta_nll": safe_spearman(-flip_radius, flip_delta),
            "gamma_vs_delta_nll": safe_spearman(flip_gamma, flip_delta),
        },
        "relations_on_all_frames": {
            "negative_radius_vs_delta_nll": safe_spearman(-all_radius, all_delta),
            "displacement_vs_delta_nll": safe_spearman(all_displacement, all_delta),
            "gamma_vs_delta_nll": safe_spearman(all_gamma, all_delta),
        },
        "flip_sign_test": {
            "positive": positive_flips,
            "negative": negative_flips,
            "ties": int((flip_delta == 0).sum()),
            "two_sided_binomial_p": float(
                binomtest(positive_flips, positive_flips + negative_flips, 0.5).pvalue
            ),
            "utterances_with_positive_mean": int(
                sum(value > 0 for value in utterance_flip_means)
            ),
            "utterances_total": len(utterance_flip_means),
        },
        "gamma_bins_all_frames": gamma_bin_summary(scored_rows),
        "radius_quartiles_on_flip_frames": radius_quartiles(flips),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["flip_frames"], indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
