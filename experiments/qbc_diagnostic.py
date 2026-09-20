#!/usr/bin/env python3
"""Run the first no-training AUV quantization-boundary diagnostic.

The experiment compares an utterance encoded in full with a hop-aligned crop
of the same waveform.  It measures normalized VQ-space displacement, exact
spherical boundary radius, the D/R certificate, and deployed token flips.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

AUV_SRC = ROOT / "sources" / "AUV" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AUV_SRC) not in sys.path:
    sys.path.insert(0, str(AUV_SRC))

from auv.model import AUV  # noqa: E402
from experiments.io_utils import decode_audio, file_sha256, iter_audio_rows  # noqa: E402
from experiments.vq_geometry import (  # noqa: E402
    exact_boundary_geometry,
    nearest_code,
    spherical_displacement,
)


@dataclass
class EncodedView:
    hidden: torch.Tensor
    z_e: torch.Tensor
    unit: torch.Tensor
    official_tokens: torch.Tensor
    geometry_tokens: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=Path,
        default=ROOT / "data/LibriSpeech/modelscope_parquet/all/test.clean/0000.parquet",
    )
    parser.add_argument("--auv-ckpt", type=Path, default=ROOT / "checkpoints/auv.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/qbc_diagnostic_v0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-utterances", type=int, default=8)
    parser.add_argument("--crop-seconds", type=float, default=4.0)
    parser.add_argument("--min-context-frames", type=int, default=25)
    parser.add_argument("--guard-frames", type=int, default=25)
    parser.add_argument("--frame-chunk", type=int, default=512)
    parser.add_argument(
        "--allow-repeated-speakers",
        action="store_true",
        help="By default, accept at most one utterance per speaker.",
    )
    parser.add_argument(
        "--max-utterances-per-speaker",
        type=int,
        default=None,
        help="Optional finite speaker cap; overrides the default one-per-speaker policy.",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def flatten_official_tokens(tokens: torch.Tensor, expected_frames: int) -> torch.Tensor:
    # AUV ResidualVQ returns (num_quantizers, B, T); this checkpoint has one Q.
    flat = tokens.reshape(-1)
    if flat.numel() != expected_frames:
        raise RuntimeError(
            f"Unexpected token shape {tuple(tokens.shape)} for {expected_frames} frames"
        )
    return flat.long()


@torch.inference_mode()
def encode_view(
    tokenizer: torch.nn.Module,
    quantizer: torch.nn.Module,
    codebook: torch.Tensor,
    wav: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> EncodedView:
    waveform = torch.from_numpy(wav).to(device=device, dtype=torch.float32)[None, :]
    captured_z_e: list[torch.Tensor] = []

    def capture_projected_latent(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured_z_e.append(output.detach())

    hook = quantizer.in_proj.register_forward_hook(capture_projected_latent)
    try:
        output = tokenizer(waveform, input_sample_rate=sample_rate)
    finally:
        hook.remove()
    if len(captured_z_e) != 1:
        raise RuntimeError(f"Expected one captured in_proj output, got {len(captured_z_e)}")
    hidden = output["before_quantize"].squeeze(0).float()  # (T, 512)
    z_e = captured_z_e[0].squeeze(0).transpose(0, 1).float()  # (T, 8)
    unit = F.normalize(z_e, dim=-1)
    official = flatten_official_tokens(output["tokens"], unit.shape[0])
    geometry = nearest_code(unit, codebook, assume_normalized=True)
    return EncodedView(hidden, z_e, unit, official, geometry)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if np.unique(labels).size < 2:
        return {"auroc": None, "average_precision": None}
    finite = np.isfinite(scores)
    if finite.sum() == 0:
        return {"auroc": None, "average_precision": None}
    clipped = np.nan_to_num(scores[finite], nan=0.0, posinf=1e6, neginf=-1e6)
    return {
        "auroc": float(roc_auc_score(labels[finite], clipped)),
        "average_precision": float(average_precision_score(labels[finite], clipped)),
    }


def grouped_logistic_auc(
    labels: np.ndarray,
    features: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | int | None]:
    """Group-held-out logistic regression, keeping utterances out of train folds."""

    finite = np.isfinite(features).all(axis=1)
    labels = labels[finite]
    features = features[finite]
    groups = groups[finite]
    unique_groups = np.unique(groups)
    n_splits = min(5, unique_groups.size)
    if n_splits < 2 or np.unique(labels).size < 2:
        return {"auroc": None, "average_precision": None, "folds": n_splits}

    predictions = np.full(labels.shape, np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for train_index, test_index in splitter.split(features, labels, groups):
        if np.unique(labels[train_index]).size < 2:
            return {"auroc": None, "average_precision": None, "folds": n_splits}
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=0),
        )
        model.fit(features[train_index], labels[train_index])
        predictions[test_index] = model.predict_proba(features[test_index])[:, 1]
    return {
        "auroc": float(roc_auc_score(labels, predictions)),
        "average_precision": float(average_precision_score(labels, predictions)),
        "folds": n_splits,
    }


def binned_flip_rate(values: np.ndarray, flips: np.ndarray, edges: list[float]) -> list[dict]:
    rows: list[dict] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (values >= left) & (values < right)
        count = int(mask.sum())
        rows.append(
            {
                "left": left,
                "right": right,
                "count": count,
                "flips": int(flips[mask].sum()),
                "flip_rate": float(flips[mask].mean()) if count else None,
            }
        )
    return rows


def save_plots(rows: list[dict], output_dir: Path) -> None:
    gamma = np.asarray([r["gamma_angle"] for r in rows], dtype=np.float64)
    flips = np.asarray([r["flip"] for r in rows], dtype=bool)
    boundary = np.asarray([r["boundary_distance_frames"] for r in rows], dtype=np.int64)

    gamma_edges = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 4, 8, np.inf]
    stats = binned_flip_rate(gamma, flips, gamma_edges)
    x = np.arange(len(stats))
    y = [np.nan if item["flip_rate"] is None else item["flip_rate"] for item in stats]
    labels = [
        f"{item['left']:g}-{item['right']:g}" if np.isfinite(item["right"]) else f">={item['left']:g}"
        for item in stats
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, y)
    ax.axvline(3.5, color="red", linestyle="--", linewidth=1, label="certificate threshold")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Token flip rate")
    ax.set_xlabel("Angular displacement / certified angular radius")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "flip_rate_vs_gamma.png", dpi=180)
    plt.close(fig)

    max_boundary = int(min(boundary.max(initial=0), 100))
    bx: list[int] = []
    by: list[float] = []
    bn: list[int] = []
    for distance in range(max_boundary + 1):
        mask = boundary == distance
        if mask.any():
            bx.append(distance)
            by.append(float(flips[mask].mean()))
            bn.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(bx, by, marker="o", markersize=3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Distance to nearest crop boundary (frames)")
    ax.set_ylabel("Token flip rate")
    ax2 = ax.twinx()
    ax2.plot(bx, bn, color="gray", alpha=0.35)
    ax2.set_ylabel("Frame count")
    fig.tight_layout()
    fig.savefig(output_dir / "flip_rate_vs_crop_boundary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_utterances_per_speaker is not None and args.max_utterances_per_speaker < 1:
        raise ValueError("--max-utterances-per-speaker must be at least 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.parquet.is_file():
        raise FileNotFoundError(args.parquet)
    if not args.auv_ckpt.is_file():
        raise FileNotFoundError(args.auv_ckpt)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    model = AUV()
    model.from_pretrained(str(args.auv_ckpt))
    model = model.to(device).eval()
    tokenizer = model.tokenizer
    hop = int(tokenizer.hop_length)
    sample_rate = int(tokenizer.sample_rate)
    quantizer = tokenizer.vq.quantizers[0]
    codebook = F.normalize(quantizer.codebook.weight.detach().float(), dim=-1).to(device)

    crop_samples = int(round(args.crop_seconds * sample_rate / hop)) * hop
    min_context_samples = args.min_context_frames * hop
    frame_rows: list[dict] = []
    utterance_rows: list[dict] = []
    used_speakers: set[int] = set()
    speaker_counts: dict[int, int] = {}
    speaker_cap = args.max_utterances_per_speaker
    if speaker_cap is None and not args.allow_repeated_speakers:
        speaker_cap = 1
    used = 0

    for row in iter_audio_rows(args.parquet):
        if used >= args.max_utterances:
            break
        speaker_id = int(row["speaker_id"])
        if speaker_cap is not None and speaker_counts.get(speaker_id, 0) >= speaker_cap:
            continue
        wav, row_sr = decode_audio(row["audio"])
        if row_sr != sample_rate:
            # Crop alignment is defined after resampling. LibriSpeech is 16 kHz,
            # so fail loudly instead of introducing an untracked grid change.
            raise RuntimeError(f"Expected {sample_rate} Hz, got {row_sr} Hz for {row['id']}")
        if wav.size < crop_samples + 2 * min_context_samples:
            continue

        max_start = wav.size - crop_samples
        crop_start = ((max_start // 2) // hop) * hop
        if crop_start < min_context_samples or wav.size - crop_start - crop_samples < min_context_samples:
            continue
        crop = wav[crop_start : crop_start + crop_samples]
        if crop_start % hop != 0 or crop.size % hop != 0:
            raise AssertionError("Crop is not aligned to the AUV hop grid")

        full = encode_view(tokenizer, quantizer, codebook, wav, row_sr, device)
        view = encode_view(tokenizer, quantizer, codebook, crop, row_sr, device)
        full_start = crop_start // hop
        full_stop = full_start + view.unit.shape[0]
        if full_stop > full.unit.shape[0]:
            raise RuntimeError(
                f"Frame alignment overflow for {row['id']}: full={full.unit.shape[0]}, "
                f"slice={view.unit.shape[0]}, start={full_start}"
            )

        full_hidden = full.hidden[full_start:full_stop]
        full_ze = full.z_e[full_start:full_stop]
        full_unit = full.unit[full_start:full_stop]
        full_official = full.official_tokens[full_start:full_stop]
        full_geometry = full.geometry_tokens[full_start:full_stop]

        geometry = exact_boundary_geometry(
            full_unit,
            codebook,
            token_ids=full_geometry,
            frame_chunk=args.frame_chunk,
            assume_normalized=True,
        )
        # This is the quantity directly optimized by QBC: the crop/view's
        # signed margin relative to the full-context teacher code. It remains
        # negative after the view has crossed any teacher-cell boundary.
        view_teacher_geometry = exact_boundary_geometry(
            view.unit,
            codebook,
            token_ids=full_geometry,
            frame_chunk=args.frame_chunk,
            assume_normalized=True,
        )
        d_angle, d_chord = spherical_displacement(
            full_unit, view.unit, assume_normalized=True
        )
        hidden_l2 = torch.linalg.vector_norm(full_hidden - view.hidden, dim=-1)
        ze_l2 = torch.linalg.vector_norm(full_ze - view.z_e, dim=-1)

        full_official_match = full_official == full_geometry
        view_official_match = view.official_tokens == view.geometry_tokens
        flips = full_official != view.official_tokens
        r_angle = geometry.angular_radius
        r_chord = geometry.chord_radius
        gamma_angle = d_angle / r_angle.clamp_min(1e-12)
        gamma_chord = d_chord / r_chord.clamp_min(1e-12)
        certified_angle = d_angle < (r_angle - 1e-6)
        certified_chord = d_chord < (r_chord - 1e-6)
        boundary_distance = torch.minimum(
            torch.arange(view.unit.shape[0], device=device),
            torch.arange(view.unit.shape[0] - 1, -1, -1, device=device),
        )

        tensors = {
            "full_official": full_official,
            "view_official": view.official_tokens,
            "full_geometry": full_geometry,
            "view_geometry": view.geometry_tokens,
            "competitor": geometry.competitor_ids,
            "flip": flips,
            "official_match_full": full_official_match,
            "official_match_view": view_official_match,
            "hidden_l2": hidden_l2,
            "ze_l2": ze_l2,
            "d_angle": d_angle,
            "d_chord": d_chord,
            "signed_margin_e": geometry.signed_euclidean_margin,
            "view_teacher_competitor": view_teacher_geometry.competitor_ids,
            "view_teacher_signed_margin_e": view_teacher_geometry.signed_euclidean_margin,
            "r_angle": r_angle,
            "r_chord": r_chord,
            "gamma_angle": gamma_angle,
            "gamma_chord": gamma_chord,
            "certified_angle": certified_angle,
            "certified_chord": certified_chord,
            "boundary_distance": boundary_distance,
        }
        cpu = {key: value.detach().cpu().numpy() for key, value in tensors.items()}

        for index in range(view.unit.shape[0]):
            frame_rows.append(
                {
                    "utterance_id": row["id"],
                    "speaker_id": row["speaker_id"],
                    "chapter_id": row["chapter_id"],
                    "crop_start_sample": crop_start,
                    "frame_in_slice": index,
                    "full_frame": full_start + index,
                    "boundary_distance_frames": int(cpu["boundary_distance"][index]),
                    "full_token": int(cpu["full_official"][index]),
                    "slice_token": int(cpu["view_official"][index]),
                    "geometry_token_full": int(cpu["full_geometry"][index]),
                    "geometry_token_slice": int(cpu["view_geometry"][index]),
                    "nearest_boundary_code": int(cpu["competitor"][index]),
                    "flip": int(cpu["flip"][index]),
                    "official_geometry_match_full": int(cpu["official_match_full"][index]),
                    "official_geometry_match_slice": int(cpu["official_match_view"][index]),
                    "hidden_l2_512": float(cpu["hidden_l2"][index]),
                    "projected_l2_8": float(cpu["ze_l2"][index]),
                    "unit_angular_displacement": float(cpu["d_angle"][index]),
                    "unit_chord_displacement": float(cpu["d_chord"][index]),
                    "signed_boundary_margin_e": float(cpu["signed_margin_e"][index]),
                    "view_teacher_nearest_boundary_code": int(cpu["view_teacher_competitor"][index]),
                    "view_teacher_signed_margin_e": float(cpu["view_teacher_signed_margin_e"][index]),
                    "angular_radius": float(cpu["r_angle"][index]),
                    "chord_radius": float(cpu["r_chord"][index]),
                    "gamma_angle": float(cpu["gamma_angle"][index]),
                    "gamma_chord": float(cpu["gamma_chord"][index]),
                    "certified_angle": int(cpu["certified_angle"][index]),
                    "certified_chord": int(cpu["certified_chord"][index]),
                    "interior": int(cpu["boundary_distance"][index] >= args.guard_frames),
                }
            )

        utterance_rows.append(
            {
                "utterance_id": row["id"],
                "duration_seconds": wav.size / sample_rate,
                "crop_start_sample": crop_start,
                "crop_frames": int(view.unit.shape[0]),
                "left_context_frames": full_start,
                "right_context_frames": int(full.unit.shape[0] - full_stop),
                "flip_rate": float(flips.float().mean().item()),
            }
        )
        used_speakers.add(speaker_id)
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        used += 1
        print(
            f"[{used}/{args.max_utterances}] {row['id']} frames={view.unit.shape[0]} "
            f"flip={flips.float().mean().item():.4f}"
        )

    if not frame_rows:
        raise RuntimeError("No utterances satisfied crop/context constraints")

    frame_csv = args.output_dir / "frames.csv"
    with frame_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)
    utterance_csv = args.output_dir / "utterances.csv"
    with utterance_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(utterance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(utterance_rows)

    flips = np.asarray([r["flip"] for r in frame_rows], dtype=np.int64)
    interior = np.asarray([r["interior"] for r in frame_rows], dtype=bool)
    d_angle = np.asarray([r["unit_angular_displacement"] for r in frame_rows])
    radius = np.asarray([r["angular_radius"] for r in frame_rows])
    gamma = np.asarray([r["gamma_angle"] for r in frame_rows])
    certified = np.asarray([r["certified_angle"] for r in frame_rows], dtype=bool)
    view_teacher_margin = np.asarray(
        [r["view_teacher_signed_margin_e"] for r in frame_rows], dtype=np.float64
    )
    geometry_match = np.asarray(
        [r["official_geometry_match_full"] and r["official_geometry_match_slice"] for r in frame_rows],
        dtype=bool,
    )
    groups = np.asarray([r["utterance_id"] for r in frame_rows])
    eps = 1e-12
    log_d = np.log(d_angle + eps)[:, None]
    log_r = np.log(radius + eps)[:, None]
    log_gamma = np.log(np.nan_to_num(gamma, nan=1.0, posinf=1e12) + eps)[:, None]
    gamma_edges = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 4, 8, math.inf]
    summary = {
        "configuration": {
            **vars(args),
            "parquet": str(args.parquet.resolve()),
            "auv_ckpt": str(args.auv_ckpt.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "hop_length": hop,
            "sample_rate": sample_rate,
            "codebook_size": int(codebook.shape[0]),
            "codebook_dim": int(codebook.shape[1]),
        },
        "provenance": {
            "auv_checkpoint_sha256": file_sha256(args.auv_ckpt),
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "counts": {
            "utterances": len(utterance_rows),
            "speakers": len(used_speakers),
            "frames": len(frame_rows),
            "interior_frames": int(interior.sum()),
            "flips": int(flips.sum()),
            "certified_frames": int(certified.sum()),
            "certificate_violations": int((certified & (flips == 1)).sum()),
            "official_geometry_mismatches": int((~geometry_match).sum()),
        },
        "rates": {
            "flip_all": float(flips.mean()),
            "flip_interior": float(flips[interior].mean()) if interior.any() else None,
            "certified_fraction": float(certified.mean()),
            "view_inside_teacher_cell": float((view_teacher_margin >= 0).mean()),
            "view_inside_teacher_cell_interior": (
                float((view_teacher_margin[interior] >= 0).mean()) if interior.any() else None
            ),
            "view_teacher_signed_margin_mean": float(view_teacher_margin.mean()),
            "view_teacher_signed_margin_interior_mean": (
                float(view_teacher_margin[interior].mean()) if interior.any() else None
            ),
        },
        "prediction": {
            "angular_displacement": safe_auc(flips, d_angle),
            "negative_angular_radius": safe_auc(flips, -radius),
            "gamma_angle": safe_auc(flips, gamma),
        },
        "prediction_group_cv": {
            "log_displacement": grouped_logistic_auc(flips, log_d, groups),
            "log_radius": grouped_logistic_auc(flips, log_r, groups),
            "log_displacement_plus_log_radius": grouped_logistic_auc(
                flips, np.concatenate([log_d, log_r], axis=1), groups
            ),
            "log_gamma": grouped_logistic_auc(flips, log_gamma, groups),
        },
        "gamma_bins": binned_flip_rate(gamma, flips.astype(bool), gamma_edges),
        "utterances": utterance_rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    save_plots(frame_rows, args.output_dir)
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["rates"], indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
