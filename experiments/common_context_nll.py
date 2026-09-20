#!/usr/bin/env python3
"""Measure the LM penalty of a full-vs-slice token flip under one clean prefix.

For every frame from an existing QBC diagnostic cohort, this script evaluates

    NLL(slice token | full-utterance token prefix)
      - NLL(full token | full-utterance token prefix).

The prefix is identical in the two terms, so the difference isolates the
current target-token substitution from accumulated prefix divergence.  This is
mechanistic/observational evidence; it is not called a causal estimate.
"""

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
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
AUV_SRC = ROOT / "sources" / "AUV" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AUV_SRC) not in sys.path:
    sys.path.insert(0, str(AUV_SRC))

from auv.model import AUV  # noqa: E402
from experiments.io_utils import decode_audio, file_sha256, iter_audio_rows  # noqa: E402
from experiments.qbc_diagnostic import flatten_official_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-csv",
        type=Path,
        default=ROOT
        / "artifacts/qbc_diagnostic_v2/llm_codec_test_clean_32speakers/frames.csv",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=ROOT / "data/LibriSpeech/modelscope_parquet/all/test.clean/0000.parquet",
    )
    parser.add_argument(
        "--auv-ckpt", type=Path, default=ROOT / "checkpoints/llm-codec.pt"
    )
    parser.add_argument(
        "--lm-dir", type=Path, default=ROOT / "checkpoints/llm-codec-hf"
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / "checkpoints/llm-codec-librispeech-adapter",
        help="Official downstream LibriSpeech speech-LM LoRA adapter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "artifacts/qbc_diagnostic_v2/llm_codec_test_clean_common_context",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--n-audio-tokens", type=int, default=20480)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--sequence-cache",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_DIR/full_token_sequences.pt.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Encode and verify the clean full-token prefixes, then stop before loading the LM.",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=None,
        help="Optional smoke-test limit, preserving frames.csv utterance order.",
    )
    return parser.parse_args()


def load_diagnostic_frames(path: Path, max_utterances: int | None) -> tuple[list[dict], list[str]]:
    numeric_int = {
        "speaker_id",
        "chapter_id",
        "crop_start_sample",
        "frame_in_slice",
        "full_frame",
        "boundary_distance_frames",
        "full_token",
        "slice_token",
        "flip",
        "interior",
    }
    numeric_float = {
        "unit_angular_displacement",
        "angular_radius",
        "gamma_angle",
    }
    rows: list[dict] = []
    utterance_order: list[str] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            utterance_id = raw["utterance_id"]
            if utterance_id not in seen:
                if max_utterances is not None and len(utterance_order) >= max_utterances:
                    continue
                seen.add(utterance_id)
                utterance_order.append(utterance_id)
            if utterance_id not in seen:
                continue
            row = dict(raw)
            for key in numeric_int:
                row[key] = int(row[key])
            for key in numeric_float:
                row[key] = float(row[key])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No diagnostic frames found in {path}")
    return rows, utterance_order


@torch.inference_mode()
def encode_full_tokens(
    tokenizer: torch.nn.Module,
    wav: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    waveform = torch.from_numpy(wav).to(device=device, dtype=torch.float32)[None, :]
    output = tokenizer(waveform, input_sample_rate=sample_rate)
    expected_frames = int(output["before_quantize"].shape[1])
    return flatten_official_tokens(output["tokens"], expected_frames).cpu()


def prepare_full_sequences(
    parquet_path: Path,
    target_ids: list[str],
    checkpoint: Path,
    checkpoint_sha256: str,
    cache_path: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, dict], int, int]:
    cache_identity = {
        "parquet": str(parquet_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "utterance_ids": target_ids,
    }
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("identity") == cache_identity:
            print(f"[codec cache] Reusing {cache_path}")
            return (
                payload["sequences"],
                payload["metadata"],
                int(payload["sample_rate"]),
                int(payload["hop"]),
            )
        print(f"[codec cache] Ignoring stale cache {cache_path}")

    model = AUV()
    model.from_pretrained(str(checkpoint))
    model = model.to(device).eval()
    tokenizer = model.tokenizer
    sample_rate = int(tokenizer.sample_rate)
    hop = int(tokenizer.hop_length)
    target_set = set(target_ids)
    sequences: dict[str, torch.Tensor] = {}
    metadata: dict[str, dict] = {}
    for row in iter_audio_rows(parquet_path):
        utterance_id = str(row["id"])
        if utterance_id not in target_set:
            continue
        wav, row_sample_rate = decode_audio(row["audio"])
        if row_sample_rate != sample_rate:
            raise RuntimeError(
                f"Expected {sample_rate} Hz, got {row_sample_rate} Hz for {utterance_id}"
            )
        sequences[utterance_id] = encode_full_tokens(
            tokenizer, wav, row_sample_rate, device
        )
        metadata[utterance_id] = {
            "duration_seconds": float(wav.size / sample_rate),
            "full_frames": int(sequences[utterance_id].numel()),
        }
        print(
            f"[codec {len(sequences)}/{len(target_ids)}] {utterance_id} "
            f"frames={sequences[utterance_id].numel()}"
        )
        if len(sequences) == len(target_ids):
            break
    missing = [item for item in target_ids if item not in sequences]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} target utterances in parquet: {missing[:5]}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "identity": cache_identity,
            "sequences": sequences,
            "metadata": metadata,
            "sample_rate": sample_rate,
            "hop": hop,
        },
        cache_path,
    )
    print(f"[codec cache] Wrote {cache_path}")
    return sequences, metadata, sample_rate, hop


def build_audio_id_table(tokenizer, n_audio_tokens: int) -> tuple[torch.Tensor, dict]:
    token_strings = [f"<CODEC_{index}>" for index in range(n_audio_tokens)]
    vocab = tokenizer.get_vocab()
    missing = [token for token in token_strings if token not in vocab]
    if missing:
        raise RuntimeError(f"Tokenizer is missing CODEC tokens, first missing: {missing[0]}")
    ids = tokenizer.convert_tokens_to_ids(token_strings)
    if len(set(ids)) != n_audio_tokens:
        raise RuntimeError("CODEC token IDs are not unique")
    contiguous = ids == list(range(ids[0], ids[0] + n_audio_tokens))
    return torch.tensor(ids, dtype=torch.long), {
        "count": n_audio_tokens,
        "first_id": int(ids[0]),
        "last_id": int(ids[-1]),
        "unique": True,
        "contiguous": bool(contiguous),
        "tokenizer_length": int(len(tokenizer)),
        "tokenizer_vocab_size_attribute": int(tokenizer.vocab_size),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


@torch.inference_mode()
def score_utterance(
    model: torch.nn.Module,
    audio_id_table_cpu: torch.Tensor,
    codec_tokens_cpu: torch.Tensor,
    selected_rows: list[dict],
    device: torch.device,
) -> list[dict]:
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    audio_id_table = audio_id_table_cpu.to(device)
    full_ids = audio_id_table[codec_tokens_cpu.to(device)]

    # Match the released training implementation: its tokenizer has no BOS, so
    # input token t-1 predicts target t.  The BOS branch is retained for explicit
    # compatibility if a future tokenizer defines one.
    tokenizer_bos = getattr(model, "_qbc_tokenizer_bos_id", None)
    if tokenizer_bos is not None:
        inputs = torch.cat(
            [torch.tensor([tokenizer_bos], device=device), full_ids[:-1]], dim=0
        )
        predictor_positions = torch.tensor(
            [row["full_frame"] for row in selected_rows], device=device, dtype=torch.long
        )
        prefix_mode = "tokenizer_bos_plus_clean_prefix"
    else:
        inputs = full_ids[:-1]
        predictor_positions = torch.tensor(
            [row["full_frame"] - 1 for row in selected_rows],
            device=device,
            dtype=torch.long,
        )
        prefix_mode = "clean_prefix_without_bos"
    if int(predictor_positions.min()) < 0 or int(predictor_positions.max()) >= inputs.numel():
        raise RuntimeError("A selected target does not have a valid clean-prefix predictor")

    outputs = causal_lm.model(
        input_ids=inputs.unsqueeze(0),
        attention_mask=torch.ones_like(inputs).unsqueeze(0),
        use_cache=False,
        return_dict=True,
    )
    selected_hidden = outputs.last_hidden_state[0].index_select(0, predictor_positions)
    logits = causal_lm.lm_head(selected_hidden).float()

    clean_codec = torch.tensor(
        [row["full_token"] for row in selected_rows], device=device, dtype=torch.long
    )
    view_codec = torch.tensor(
        [row["slice_token"] for row in selected_rows], device=device, dtype=torch.long
    )
    clean_ids = audio_id_table.index_select(0, clean_codec)
    view_ids = audio_id_table.index_select(0, view_codec)
    row_index = torch.arange(logits.shape[0], device=device)
    clean_logits = logits[row_index, clean_ids]
    view_logits = logits[row_index, view_ids]
    full_log_norm = torch.logsumexp(logits, dim=-1)
    audio_logits = logits.index_select(1, audio_id_table)
    audio_log_norm = torch.logsumexp(audio_logits, dim=-1)
    audio_top1 = torch.argmax(audio_logits, dim=-1)
    clean_rank_audio = 1 + (audio_logits > clean_logits[:, None]).sum(dim=-1)
    view_rank_audio = 1 + (audio_logits > view_logits[:, None]).sum(dim=-1)

    scored: list[dict] = []
    values = {
        "clean_logit": clean_logits,
        "slice_logit": view_logits,
        "delta_nll_common_context": clean_logits - view_logits,
        "clean_nll_full_vocab": full_log_norm - clean_logits,
        "slice_nll_full_vocab": full_log_norm - view_logits,
        "clean_nll_audio_only": audio_log_norm - clean_logits,
        "slice_nll_audio_only": audio_log_norm - view_logits,
        "clean_rank_audio": clean_rank_audio,
        "slice_rank_audio": view_rank_audio,
        "audio_top1_code": audio_top1,
    }
    cpu = {key: value.detach().cpu().numpy() for key, value in values.items()}
    for index, source in enumerate(selected_rows):
        item = dict(source)
        item["prefix_mode"] = prefix_mode
        for key in (
            "clean_logit",
            "slice_logit",
            "delta_nll_common_context",
            "clean_nll_full_vocab",
            "slice_nll_full_vocab",
            "clean_nll_audio_only",
            "slice_nll_audio_only",
        ):
            item[key] = float(cpu[key][index])
        for key in ("clean_rank_audio", "slice_rank_audio", "audio_top1_code"):
            item[key] = int(cpu[key][index])
        item["lm_prefers_clean"] = int(item["delta_nll_common_context"] > 0.0)
        scored.append(item)
    del outputs, selected_hidden, logits, audio_logits
    return scored


def summarize_subset(rows: list[dict], bootstrap_replicates: int, seed: int) -> dict:
    if not rows:
        return {"frames": 0}
    delta = np.asarray([row["delta_nll_common_context"] for row in rows], dtype=np.float64)
    clean_full = np.asarray([row["clean_nll_full_vocab"] for row in rows], dtype=np.float64)
    view_full = np.asarray([row["slice_nll_full_vocab"] for row in rows], dtype=np.float64)
    clean_audio = np.asarray([row["clean_nll_audio_only"] for row in rows], dtype=np.float64)
    view_audio = np.asarray([row["slice_nll_audio_only"] for row in rows], dtype=np.float64)
    clean_rank = np.asarray([row["clean_rank_audio"] for row in rows], dtype=np.float64)
    view_rank = np.asarray([row["slice_rank_audio"] for row in rows], dtype=np.float64)
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["utterance_id"])].append(float(row["delta_nll_common_context"]))
    group_names = list(groups)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        values = np.concatenate([np.asarray(groups[name]) for name in sampled])
        bootstrap[index] = values.mean()
    return {
        "frames": len(rows),
        "utterances": len(group_names),
        "mean_delta_nll_nats": float(delta.mean()),
        "mean_delta_nll_bits": float(delta.mean() / math.log(2.0)),
        "median_delta_nll_nats": float(np.median(delta)),
        "mean_delta_nll_cluster_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "fraction_lm_prefers_clean": float((delta > 0.0).mean()),
        "fraction_equal": float((delta == 0.0).mean()),
        "mean_clean_nll_full_vocab": float(clean_full.mean()),
        "mean_slice_nll_full_vocab": float(view_full.mean()),
        "clean_ppl_full_vocab": float(math.exp(clean_full.mean())),
        "slice_ppl_full_vocab": float(math.exp(view_full.mean())),
        "mean_clean_nll_audio_only": float(clean_audio.mean()),
        "mean_slice_nll_audio_only": float(view_audio.mean()),
        "mean_clean_rank_audio": float(clean_rank.mean()),
        "mean_slice_rank_audio": float(view_rank.mean()),
        "fraction_clean_top1_audio": float((clean_rank == 1).mean()),
        "fraction_slice_top1_audio": float((view_rank == 1).mean()),
    }


def safe_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
        return {"rho": None, "pvalue": None, "frames": int(finite.sum())}
    result = spearmanr(x[finite], y[finite])
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue), "frames": int(finite.sum())}


def radius_quartiles(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    radius = np.asarray([row["angular_radius"] for row in rows], dtype=np.float64)
    edges = np.quantile(radius, [0.0, 0.25, 0.5, 0.75, 1.0])
    output: list[dict] = []
    for index in range(4):
        if index == 3:
            mask = (radius >= edges[index]) & (radius <= edges[index + 1])
        else:
            mask = (radius >= edges[index]) & (radius < edges[index + 1])
        selected = [row for row, keep in zip(rows, mask) if keep]
        delta = np.asarray(
            [row["delta_nll_common_context"] for row in selected], dtype=np.float64
        )
        output.append(
            {
                "quartile": index + 1,
                "left": float(edges[index]),
                "right": float(edges[index + 1]),
                "frames": len(selected),
                "mean_delta_nll_nats": float(delta.mean()) if delta.size else None,
                "fraction_lm_prefers_clean": float((delta > 0).mean()) if delta.size else None,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for required in (args.frames_csv, args.parquet, args.auv_ckpt):
        if not required.is_file():
            raise FileNotFoundError(required)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_cache = args.sequence_cache or args.output_dir / "full_token_sequences.pt"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    diagnostic_rows, utterance_order = load_diagnostic_frames(
        args.frames_csv, args.max_utterances
    )
    rows_by_utterance: dict[str, list[dict]] = defaultdict(list)
    for row in diagnostic_rows:
        rows_by_utterance[str(row["utterance_id"])].append(row)
    for utterance_id in utterance_order:
        rows_by_utterance[utterance_id].sort(key=lambda row: row["frame_in_slice"])

    auv_checkpoint_sha256 = file_sha256(args.auv_ckpt)
    sequences, utterance_metadata, sample_rate, hop = prepare_full_sequences(
        args.parquet,
        utterance_order,
        args.auv_ckpt,
        auv_checkpoint_sha256,
        sequence_cache,
        device,
    )
    diagnostic_mismatches = 0
    for utterance_id in utterance_order:
        sequence = sequences[utterance_id]
        for row in rows_by_utterance[utterance_id]:
            full_frame = int(row["full_frame"])
            if full_frame >= sequence.numel():
                raise RuntimeError(f"Frame overflow for {utterance_id}: {full_frame}")
            diagnostic_mismatches += int(int(sequence[full_frame]) != int(row["full_token"]))
    if diagnostic_mismatches:
        raise RuntimeError(
            f"Codec re-encoding disagrees with diagnostic CSV at {diagnostic_mismatches} frames"
        )
    print("[verify] diagnostic_full_token_mismatches=0")
    if args.prepare_only:
        print("Prepared and verified full-token clean prefixes; --prepare-only requested.")
        return

    model_weights = args.lm_dir / "model.safetensors"
    if not model_weights.is_file():
        raise FileNotFoundError(model_weights)
    adapter_weights = args.adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(adapter_weights)
    # The published tokenizer_config was produced by Transformers 5 and stores
    # extra_special_tokens as a list.  Transformers 4.57 (needed by this host's
    # Torch 2.3 runtime) expects a dict for that new field.  The canonical
    # additional_special_tokens are still loaded from special_tokens_map.json.
    tokenizer = AutoTokenizer.from_pretrained(
        args.lm_dir,
        local_files_only=True,
        extra_special_tokens={},
    )
    audio_id_table, tokenizer_info = build_audio_id_table(tokenizer, args.n_audio_tokens)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.lm_dir,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        local_files_only=True,
    )
    model.eval()
    # The training code branches on tokenizer.bos_token_id, not config.bos_token_id.
    model._qbc_tokenizer_bos_id = tokenizer.bos_token_id

    scored_rows: list[dict] = []
    for index, utterance_id in enumerate(utterance_order, start=1):
        selected = score_utterance(
            model,
            audio_id_table,
            sequences[utterance_id],
            rows_by_utterance[utterance_id],
            device,
        )
        scored_rows.extend(selected)
        flip_delta = [row["delta_nll_common_context"] for row in selected if row["flip"]]
        print(
            f"[lm {index}/{len(utterance_order)}] {utterance_id} "
            f"frames={len(selected)} flips={len(flip_delta)} "
            f"mean_flip_delta={np.mean(flip_delta) if flip_delta else math.nan:.4f}"
        )

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
    flip_radius = np.asarray([row["angular_radius"] for row in flips], dtype=np.float64)
    flip_gamma = np.asarray([row["gamma_angle"] for row in flips], dtype=np.float64)
    summary = {
        "definition": {
            "delta_nll_common_context": (
                "NLL(slice token | full clean prefix) - "
                "NLL(full token | full clean prefix), in nats"
            ),
            "positive_means": "the released LM assigns higher probability to the full token",
            "claim_scope": "mechanistic observational evidence, not a causal estimate",
        },
        "configuration": {
            "frames_csv": str(args.frames_csv.resolve()),
            "parquet": str(args.parquet.resolve()),
            "auv_checkpoint": str(args.auv_ckpt.resolve()),
            "lm_dir": str(args.lm_dir.resolve()),
            "adapter_dir": str(args.adapter_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "device": str(device),
            "dtype": args.dtype,
            "sample_rate": sample_rate,
            "hop_length": hop,
        },
        "provenance": {
            "auv_checkpoint_sha256": auv_checkpoint_sha256,
            "lm_model_safetensors_sha256": file_sha256(model_weights),
            "speech_lm_adapter_sha256": file_sha256(adapter_weights),
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "tokenizer": tokenizer_info,
        "verification": {
            "diagnostic_full_token_mismatches": diagnostic_mismatches,
        },
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
        "radius_quartiles_on_flip_frames": radius_quartiles(flips),
        "utterances": utterance_metadata,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["flip_frames"], indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
