#!/usr/bin/env python3
"""Matched reconstruction/QBC pilot for the current LLM-Codec checkpoint.

Each example supplies two views of the same waveform: the full utterance is a
stop-gradient teacher and a hop-aligned four-second crop is reconstructed and
optimized.  The QBC arm differs from the control arm only by the signed
all-code Voronoi-margin hinge loss.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
AUV_SRC = ROOT / "sources" / "AUV" / "src"
LLM_CODEC_SRC = ROOT / "sources" / "llm-codec"
for path in (ROOT, AUV_SRC, LLM_CODEC_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from auv.model import AUV  # noqa: E402
from experiments.io_utils import decode_audio  # noqa: E402
from experiments.qbc_loss import qbc_hinge_loss, signed_boundary_margin  # noqa: E402
from llm_codec.losses import MelLoss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data/LibriSpeech/modelscope_parquet/all/train.clean.100",
    )
    parser.add_argument("--auv-ckpt", type=Path, default=ROOT / "checkpoints/llm-codec.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of valid utterance crops per optimizer update (effective batch size).",
    )
    parser.add_argument("--crop-seconds", type=float, default=4.0)
    parser.add_argument("--min-context-frames", type=int, default=25)
    parser.add_argument("--guard-frames", type=int, default=25)
    parser.add_argument("--max-full-seconds", type=float, default=20.0)
    parser.add_argument("--lambda-mel", type=float, default=1.0)
    parser.add_argument("--lambda-vq", type=float, default=1.0)
    parser.add_argument("--lambda-qbc", type=float, default=1.0)
    parser.add_argument("--target-margin", type=float, default=0.03)
    parser.add_argument("--teacher-confidence-scale", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--train-decoder",
        action="store_true",
        help="Adapt token2wav jointly with the tokenizer to preserve reconstruction quality.",
    )
    parser.add_argument(
        "--decoder-lr",
        type=float,
        default=5e-6,
        help="Decoder learning rate when --train-decoder is enabled.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--frame-chunk", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--freeze-codebook",
        action="store_true",
        help="Freeze the codebook even for the original VQ loss. QBC always detaches it.",
    )
    return parser.parse_args()


def iter_rows(input_dir: Path) -> Iterator[dict]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files under {input_dir}")
    while True:
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=1,
                columns=["audio", "id", "speaker_id", "chapter_id"],
            ):
                yield batch.to_pylist()[0]


def flatten_tokens(tokens: torch.Tensor, expected_frames: int) -> torch.Tensor:
    values = tokens.reshape(-1).long()
    if values.numel() != expected_frames:
        raise RuntimeError(f"Unexpected token shape {tuple(tokens.shape)} for {expected_frames} frames")
    return values


def choose_aligned_crop(
    num_samples: int,
    crop_samples: int,
    hop: int,
    context_frames: int,
    rng: random.Random,
) -> int | None:
    min_start_frame = context_frames
    max_start_frame = (num_samples - crop_samples) // hop - context_frames
    if max_start_frame < min_start_frame:
        return None
    return rng.randint(min_start_frame, max_start_frame) * hop


def match_length(wav: torch.Tensor, target: int) -> torch.Tensor:
    if wav.shape[-1] > target:
        return wav[..., :target]
    if wav.shape[-1] < target:
        return F.pad(wav, (0, target - wav.shape[-1]))
    return wav


def save_checkpoint(model: AUV, output_dir: Path, name: str) -> Path:
    path = output_dir / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)
    return path


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = AUV()
    model.from_pretrained(str(args.auv_ckpt))
    model = model.to(device)
    model.tokenizer.train()
    if args.train_decoder:
        model.token2wav.train()
        for parameter in model.token2wav.parameters():
            parameter.requires_grad_(True)
    else:
        model.token2wav.eval()
        for parameter in model.token2wav.parameters():
            parameter.requires_grad_(False)

    quantizer = model.tokenizer.vq.quantizers[0]
    codebook = quantizer.codebook.weight
    if args.freeze_codebook:
        codebook.requires_grad_(False)

    encoder_trainable = [
        parameter for parameter in model.tokenizer.parameters() if parameter.requires_grad
    ]
    decoder_trainable = [
        parameter for parameter in model.token2wav.parameters() if parameter.requires_grad
    ]
    optimizer_groups = [{"params": encoder_trainable, "lr": args.lr}]
    if decoder_trainable:
        optimizer_groups.append({"params": decoder_trainable, "lr": args.decoder_lr})
    all_trainable = encoder_trainable + decoder_trainable
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    mel_loss_fn = MelLoss(
        sr=int(model.tokenizer.sample_rate),
        n_fft=1024,
        hop_length=int(model.tokenizer.hop_length),
        n_mels=100,
    ).to(device)
    sample_rate = int(model.tokenizer.sample_rate)
    hop = int(model.tokenizer.hop_length)
    crop_samples = int(round(args.crop_seconds * sample_rate / hop)) * hop
    max_full_samples = int(args.max_full_seconds * sample_rate)
    amp_enabled = device.type == "cuda" and not args.fp32

    metrics_path = args.output_dir / "train_metrics.jsonl"
    row_iterator = iter_rows(args.input_dir)
    started = time.monotonic()
    step = 0
    skipped = 0
    micro_examples = 0
    running: list[dict[str, float]] = []
    pending: list[dict[str, float | int | str]] = []
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        source = next(row_iterator)
        wav_np, source_sr = decode_audio(source["audio"])
        if source_sr != sample_rate or wav_np.size > max_full_samples:
            skipped += 1
            continue
        start_sample = choose_aligned_crop(
            wav_np.size,
            crop_samples,
            hop,
            args.min_context_frames,
            rng,
        )
        if start_sample is None:
            skipped += 1
            continue

        full_wav = torch.from_numpy(wav_np).to(device=device, dtype=torch.float32)[None, :]
        crop_wav = full_wav[:, start_sample : start_sample + crop_samples]
        start_frame = start_sample // hop

        # Full-context code is a stop-gradient teacher. Capture its real
        # projected/normalized pre-VQ representation for confidence weighting.
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            full_hidden = model.tokenizer.get_hidden_feat(full_wav, input_sample_rate=sample_rate)
            full_z_e = quantizer.in_proj(full_hidden)
            _, full_indices = quantizer.decode_latents(full_z_e)
        full_ids = flatten_tokens(full_indices, full_z_e.shape[-1])

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            view_out = model.tokenizer(crop_wav, input_sample_rate=sample_rate)
            view_hidden = view_out["before_quantize"].transpose(1, 2)
            view_z_e = quantizer.in_proj(view_hidden)
            view_frames = view_z_e.shape[-1]
            teacher_ids = full_ids[start_frame : start_frame + view_frames]
            teacher_z_e = full_z_e[0, :, start_frame : start_frame + view_frames].transpose(0, 1)
            if teacher_ids.numel() != view_frames:
                skipped += 1
                continue

            reconstruction = model.token2wav(view_out["quantized"])
            reconstruction = match_length(reconstruction, crop_samples)
            mel = mel_loss_fn(crop_wav[:, None, :], reconstruction)
            vq = view_out["vq_loss"]

        view_frames_td = view_z_e[0].transpose(0, 1)
        guard = min(args.guard_frames, max(0, view_frames // 2 - 1))
        frame_mask = torch.zeros(view_frames, dtype=torch.bool, device=device)
        frame_mask[guard : view_frames - guard] = True

        with torch.no_grad():
            teacher_geometry = signed_boundary_margin(
                teacher_z_e,
                codebook,
                teacher_ids,
                frame_chunk=args.frame_chunk,
                detach_codebook=True,
            )
            confidence = teacher_geometry.margin.clamp_min(0.0)
            if args.teacher_confidence_scale > 0:
                confidence = confidence / (confidence + args.teacher_confidence_scale)
            else:
                confidence = torch.ones_like(confidence)

        qbc, view_geometry, per_frame_qbc = qbc_hinge_loss(
            view_frames_td,
            codebook,
            teacher_ids,
            target_margin=args.target_margin,
            frame_mask=frame_mask,
            frame_weights=confidence,
            frame_chunk=args.frame_chunk,
        )
        total = args.lambda_mel * mel + args.lambda_vq * vq + args.lambda_qbc * qbc
        (total / args.grad_accum_steps).backward()

        with torch.no_grad():
            view_ids = flatten_tokens(view_out["tokens"], view_frames)
            selected = frame_mask
            agreement = (view_ids[selected] == teacher_ids[selected]).float().mean()
            active = (per_frame_qbc[selected] > 0).float().mean()
            pending.append({
                "utterance_id": str(source["id"]),
                "start_frame": start_frame,
                "frames": view_frames,
                "loss_total": float(total.detach()),
                "loss_mel": float(mel.detach()),
                "loss_vq": float(vq.detach()),
                "loss_qbc": float(qbc.detach()),
                "agreement_interior": float(agreement),
                "view_margin_mean": float(view_geometry.margin[selected].mean()),
                "teacher_margin_mean": float(teacher_geometry.margin[selected].mean()),
                "qbc_active_fraction": float(active),
                "teacher_confidence_mean": float(confidence[selected].mean()),
            })
        micro_examples += 1
        if len(pending) < args.grad_accum_steps:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip)
        codebook_grad = codebook.grad
        codebook_grad_norm = float(codebook_grad.norm()) if codebook_grad is not None else 0.0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        averaged_keys = (
            "loss_total",
            "loss_mel",
            "loss_vq",
            "loss_qbc",
            "agreement_interior",
            "view_margin_mean",
            "teacher_margin_mean",
            "qbc_active_fraction",
            "teacher_confidence_mean",
        )
        row = {
            "step": step,
            "micro_batch_size": len(pending),
            "utterance_id": pending[-1]["utterance_id"],
            "utterance_ids": [item["utterance_id"] for item in pending],
            "start_frame": pending[-1]["start_frame"],
            "frames": sum(int(item["frames"]) for item in pending),
            **{
                key: float(np.mean([float(item[key]) for item in pending]))
                for key in averaged_keys
            },
            "grad_norm": float(grad_norm),
            "codebook_grad_norm_total": codebook_grad_norm,
        }
        pending.clear()
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        running.append(row)

        if step == 1 or step % args.log_every == 0:
            window = running[-args.log_every :]
            print(
                f"[step {step}/{args.max_steps}] "
                f"total={np.mean([x['loss_total'] for x in window]):.4f} "
                f"mel={np.mean([x['loss_mel'] for x in window]):.4f} "
                f"vq={np.mean([x['loss_vq'] for x in window]):.4f} "
                f"qbc={np.mean([x['loss_qbc'] for x in window]):.4f} "
                f"agree={np.mean([x['agreement_interior'] for x in window]):.3f} "
                f"margin={np.mean([x['view_margin_mean'] for x in window]):.4f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
        if step % args.save_every == 0:
            save_checkpoint(model, args.output_dir, f"step-{step:06d}.pt")

    final_path = save_checkpoint(model, args.output_dir, "final.pt")
    elapsed = time.monotonic() - started
    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "checkpoint": str(final_path.resolve()),
        "steps": step,
        "micro_examples": micro_examples,
        "effective_batch_size": args.grad_accum_steps,
        "skipped_rows": skipped,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(step, 1),
        "trainable_parameters": sum(parameter.numel() for parameter in all_trainable),
        "encoder_trainable_parameters": sum(
            parameter.numel() for parameter in encoder_trainable
        ),
        "decoder_trainable_parameters": sum(
            parameter.numel() for parameter in decoder_trainable
        ),
        "mean_last_50": {
            key: float(np.mean([row[key] for row in running[-50:]]))
            for key in (
                "loss_total",
                "loss_mel",
                "loss_vq",
                "loss_qbc",
                "agreement_interior",
                "view_margin_mean",
                "teacher_margin_mean",
                "qbc_active_fraction",
                "teacher_confidence_mean",
                "grad_norm",
                "codebook_grad_norm_total",
            )
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
