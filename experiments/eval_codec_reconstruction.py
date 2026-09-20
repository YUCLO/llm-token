#!/usr/bin/env python3
"""Evaluate matched codec checkpoints on identical hop-aligned crops."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
AUV_SRC = ROOT / "sources" / "AUV" / "src"
LLM_CODEC_SRC = ROOT / "sources" / "llm-codec"
for path in (ROOT, AUV_SRC, LLM_CODEC_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from auv.model import AUV  # noqa: E402
from experiments.io_utils import decode_audio, iter_audio_rows  # noqa: E402
from experiments.stats_utils import bootstrap_mean_ci, paired_bootstrap_difference  # noqa: E402
from llm_codec.losses import MelLoss, MultiResolutionSTFTLoss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-utterances", type=int, default=32)
    parser.add_argument("--crop-seconds", type=float, default=4.0)
    parser.add_argument("--min-context-frames", type=int, default=25)
    parser.add_argument(
        "--allow-repeated-speakers",
        action="store_true",
        help="Permit multiple utterances per speaker when evaluating larger cohorts.",
    )
    parser.add_argument(
        "--max-utterances-per-speaker",
        type=int,
        default=None,
        help="Optional finite speaker cap; overrides the default one-per-speaker policy.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=CHECKPOINT, got {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not name or not path.is_file():
            raise ValueError(f"Invalid model specification: {value}")
        result.append((name, path))
    if len({name for name, _ in result}) != len(result):
        raise ValueError("Model names must be unique")
    return result


def match_length(wav: torch.Tensor, target: int) -> torch.Tensor:
    if wav.shape[-1] > target:
        return wav[..., :target]
    if wav.shape[-1] < target:
        return F.pad(wav, (0, target - wav.shape[-1]))
    return wav


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    reference = reference.float() - reference.float().mean(dim=-1, keepdim=True)
    estimate = estimate.float() - estimate.float().mean(dim=-1, keepdim=True)
    scale = (estimate * reference).sum(dim=-1, keepdim=True) / reference.square().sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    target = scale * reference
    noise = estimate - target
    return 10.0 * torch.log10(
        target.square().sum(dim=-1).clamp_min(eps)
        / noise.square().sum(dim=-1).clamp_min(eps)
    )


def bootstrap_difference(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    return paired_bootstrap_difference(candidate, baseline, samples=samples, seed=seed)


def bootstrap_speaker_cluster_difference(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    candidate_by_id = {str(row["utterance_id"]): row for row in candidate_rows}
    differences_by_speaker: dict[int, list[float]] = {}
    for baseline_row in baseline_rows:
        utterance_id = str(baseline_row["utterance_id"])
        candidate_row = candidate_by_id[utterance_id]
        speaker_id = int(baseline_row["speaker_id"])
        difference = float(candidate_row[metric]) - float(baseline_row[metric])
        differences_by_speaker.setdefault(speaker_id, []).append(difference)
    differences = np.asarray(
        [np.mean(values) for values in differences_by_speaker.values()], dtype=np.float64
    )
    estimate, low, high = bootstrap_mean_ci(
        differences,
        samples=samples,
        generator=np.random.default_rng(seed),
    )
    return {
        "candidate_minus_baseline": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "cluster_key": "speaker_id",
        "clusters": int(differences.size),
    }


def main() -> None:
    args = parse_args()
    if args.max_utterances_per_speaker is not None and args.max_utterances_per_speaker < 1:
        raise ValueError("--max-utterances-per-speaker must be at least 1")
    models = parse_models(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Checkpoints share AUV's fixed 16 kHz / 320-sample contract.
    sample_rate = 16000
    hop = 320
    crop_samples = int(round(args.crop_seconds * sample_rate / hop)) * hop
    min_context_samples = args.min_context_frames * hop
    selected: list[tuple[str, int, np.ndarray]] = []
    used_speakers: set[int] = set()
    speaker_counts: dict[int, int] = {}
    speaker_cap = args.max_utterances_per_speaker
    if speaker_cap is None and not args.allow_repeated_speakers:
        speaker_cap = 1
    for row in iter_audio_rows(args.parquet):
        if len(selected) >= args.max_utterances:
            break
        speaker = int(row["speaker_id"])
        if speaker_cap is not None and speaker_counts.get(speaker, 0) >= speaker_cap:
            continue
        wav, row_sr = decode_audio(row["audio"])
        if row_sr != sample_rate or wav.size < crop_samples + 2 * min_context_samples:
            continue
        max_start = wav.size - crop_samples
        start = ((max_start // 2) // hop) * hop
        if start < min_context_samples or wav.size - start - crop_samples < min_context_samples:
            continue
        selected.append(
            (str(row["id"]), speaker, np.ascontiguousarray(wav[start : start + crop_samples]))
        )
        used_speakers.add(speaker)
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    if len(selected) != args.max_utterances:
        raise RuntimeError(f"Only selected {len(selected)} utterances")

    rows: list[dict] = []
    for model_index, (name, checkpoint) in enumerate(models):
        model = AUV()
        model.from_pretrained(str(checkpoint))
        model = model.to(device).eval()
        mel_fn = MelLoss(sr=sample_rate, n_fft=1024, hop_length=hop, n_mels=100).to(device)
        mrstft_fn = MultiResolutionSTFTLoss().to(device)
        for utterance_id, speaker_id, wav_np in selected:
            wav = torch.from_numpy(wav_np).to(device=device, dtype=torch.float32)[None, :]
            with torch.inference_mode():
                encoded = model.tokenizer(wav, input_sample_rate=sample_rate)
                reconstruction = model.token2wav(encoded["quantized"])
                reconstruction = match_length(reconstruction, wav.shape[-1])
                reference = wav[:, None, :]
                row = {
                    "model": name,
                    "utterance_id": utterance_id,
                    "speaker_id": speaker_id,
                    "mel_log_l1": float(mel_fn(reference, reconstruction)),
                    "mrstft": float(mrstft_fn(reference, reconstruction)),
                    "waveform_l1": float(F.l1_loss(reference, reconstruction)),
                    "si_sdr_db": float(si_sdr(reference.squeeze(1), reconstruction.squeeze(1)).mean()),
                    "tokens": int(encoded["tokens"].numel()),
                }
            rows.append(row)
        del model, mel_fn, mrstft_fn
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[{model_index + 1}/{len(models)}] evaluated {name}", flush=True)

    csv_path = args.output_dir / "per_utterance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = ["mel_log_l1", "mrstft", "waveform_l1", "si_sdr_db"]
    summary: dict = {
        "configuration": {
            "parquet": str(args.parquet.resolve()),
            "models": {name: str(path.resolve()) for name, path in models},
            "utterances": len(selected),
            "speakers": len(used_speakers),
            "crop_seconds": args.crop_seconds,
            "allow_repeated_speakers": args.allow_repeated_speakers,
            "max_utterances_per_speaker": args.max_utterances_per_speaker,
        },
        "means": {},
        "paired_vs_first_model": {},
        "paired_vs_first_model_speaker_cluster": {},
    }
    by_model = {name: [row for row in rows if row["model"] == name] for name, _ in models}
    for name, _ in models:
        summary["means"][name] = {
            metric: float(np.mean([row[metric] for row in by_model[name]])) for metric in metrics
        }
    baseline_name = models[0][0]
    for candidate_index, (name, _) in enumerate(models[1:], start=1):
        summary["paired_vs_first_model"][name] = {}
        summary["paired_vs_first_model_speaker_cluster"][name] = {}
        for metric_index, metric in enumerate(metrics):
            baseline = np.asarray([row[metric] for row in by_model[baseline_name]])
            candidate = np.asarray([row[metric] for row in by_model[name]])
            summary["paired_vs_first_model"][name][metric] = bootstrap_difference(
                candidate,
                baseline,
                samples=args.bootstrap_samples,
                seed=args.seed + 10 * candidate_index + metric_index,
            )
            summary["paired_vs_first_model_speaker_cluster"][name][metric] = (
                bootstrap_speaker_cluster_difference(
                    by_model[name],
                    by_model[baseline_name],
                    metric,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 100 * candidate_index + metric_index,
                )
            )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
