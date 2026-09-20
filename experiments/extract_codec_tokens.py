#!/usr/bin/env python3
"""Extract current AUV/LLM-Codec token sequences from local Parquet audio."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
AUV_SRC = ROOT / "sources" / "AUV" / "src"
for path in (ROOT, AUV_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from auv.model import AUV  # noqa: E402
from experiments.io_utils import decode_audio  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--auv-ckpt", type=Path, default=ROOT / "checkpoints/llm-codec.pt"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def flatten_tokens(tokens: torch.Tensor, expected_frames: int) -> list[int]:
    values = tokens.reshape(-1)
    if values.numel() != expected_frames:
        raise RuntimeError(
            f"Unexpected token shape {tuple(tokens.shape)} for {expected_frames} frames"
        )
    return values.detach().cpu().to(torch.int32).tolist()


@torch.inference_mode()
def encode_tokens(
    tokenizer: torch.nn.Module,
    wav: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> list[int]:
    waveform = torch.from_numpy(wav).to(device=device, dtype=torch.float32)[None, :]
    output = tokenizer(waveform, input_sample_rate=sample_rate)
    expected_frames = int(output["before_quantize"].shape[1])
    return flatten_tokens(output["tokens"], expected_frames)


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    input_files = sorted(args.input_dir.glob("*.parquet"))
    assigned = input_files[args.shard_index :: args.shard_count]
    if not assigned:
        raise RuntimeError("No assigned input Parquet files")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = AUV()
    model.from_pretrained(str(args.auv_ckpt))
    model = model.to(device).eval()
    tokenizer = model.tokenizer
    sample_rate = int(tokenizer.sample_rate)
    hop = int(tokenizer.hop_length)

    started = time.monotonic()
    total_utterances = 0
    total_audio_seconds = 0.0
    total_tokens = 0
    completed_files: list[str] = []
    stop = False

    for input_path in assigned:
        output_path = args.output_dir / f"part-{input_path.stem}.parquet"
        if output_path.is_file() and args.max_utterances is None:
            print(f"[skip] {output_path}")
            completed_files.append(str(output_path.resolve()))
            continue

        rows: list[dict] = []
        parquet = pq.ParquetFile(input_path)
        columns = ["audio", "id", "text", "speaker_id", "chapter_id"]
        for batch in parquet.iter_batches(batch_size=1, columns=columns):
            source = batch.to_pylist()[0]
            wav, row_sample_rate = decode_audio(source["audio"])
            if row_sample_rate != sample_rate:
                raise RuntimeError(
                    f"Expected {sample_rate} Hz, got {row_sample_rate} for {source['id']}"
                )
            codes = encode_tokens(tokenizer, wav, row_sample_rate, device)
            expected = int(round(wav.size / hop))
            if abs(len(codes) - expected) > 1:
                raise RuntimeError(
                    f"Unexpected frame count for {source['id']}: {len(codes)} vs {expected}"
                )
            rows.append(
                {
                    "id": str(source["id"]),
                    "speaker_id": int(source["speaker_id"]),
                    "chapter_id": int(source["chapter_id"]),
                    "text": str(source["text"]),
                    "audio_seconds": float(wav.size / sample_rate),
                    "audio_codes": codes,
                }
            )
            total_utterances += 1
            total_audio_seconds += wav.size / sample_rate
            total_tokens += len(codes)
            if total_utterances % args.progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"[extract] worker={args.shard_index}/{args.shard_count} "
                    f"utterances={total_utterances} audio_h={total_audio_seconds / 3600:.3f} "
                    f"tokens={total_tokens} throughput={total_audio_seconds / elapsed:.1f}x"
                )
            if args.max_utterances is not None and total_utterances >= args.max_utterances:
                stop = True
                break

        table = pa.Table.from_pylist(
            rows,
            schema=pa.schema(
                [
                    ("id", pa.string()),
                    ("speaker_id", pa.int64()),
                    ("chapter_id", pa.int64()),
                    ("text", pa.string()),
                    ("audio_seconds", pa.float64()),
                    ("audio_codes", pa.list_(pa.int32())),
                ]
            ),
        )
        temporary_path = output_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(output_path)
        completed_files.append(str(output_path.resolve()))
        print(f"[write] {output_path} rows={len(rows)}")
        if stop:
            break

    elapsed = time.monotonic() - started
    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "checkpoint": str(args.auv_ckpt.resolve()),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sample_rate": sample_rate,
        "hop_length": hop,
        "utterances": total_utterances,
        "audio_hours": total_audio_seconds / 3600.0,
        "tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "audio_realtime_throughput": total_audio_seconds / elapsed,
        "completed_files": completed_files,
    }
    summary_path = args.output_dir / f"extract-worker-{args.shard_index:02d}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
