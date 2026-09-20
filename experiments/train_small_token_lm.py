#!/usr/bin/env python3
"""Train a compact causal LM on tokens from the exact deployed codec version."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocab-size", type=int, default=20480)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--val-permille", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def stable_validation_assignment(utterance_id: str, val_permille: int) -> bool:
    digest = hashlib.sha1(utterance_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 1000
    return bucket < val_permille


def load_flat_splits(
    tokens_dir: Path,
    boundary_token: int,
    val_permille: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    files = sorted(tokens_dir.glob("part-*.parquet"))
    if not files:
        raise RuntimeError(f"No part-*.parquet files in {tokens_dir}")
    split_sequences: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
    split_utterances = {"train": 0, "validation": 0}
    split_audio_seconds = {"train": 0.0, "validation": 0.0}
    code_counts = np.zeros(boundary_token, dtype=np.int64)
    for path in files:
        table = pq.read_table(path, columns=["id", "audio_seconds", "audio_codes"])
        for row in table.to_pylist():
            codes = np.asarray(row["audio_codes"], dtype=np.int32)
            if codes.size == 0:
                continue
            if int(codes.min()) < 0 or int(codes.max()) >= boundary_token:
                raise RuntimeError(f"Out-of-range code in {row['id']}")
            split = (
                "validation"
                if stable_validation_assignment(str(row["id"]), val_permille)
                else "train"
            )
            split_sequences[split].append(codes)
            split_utterances[split] += 1
            split_audio_seconds[split] += float(row["audio_seconds"])
            code_counts += np.bincount(codes, minlength=boundary_token)

    def concatenate(sequences: list[np.ndarray]) -> np.ndarray:
        total = sum(sequence.size + 1 for sequence in sequences)
        flat = np.empty(total, dtype=np.int32)
        offset = 0
        for sequence in sequences:
            flat[offset] = boundary_token
            offset += 1
            flat[offset : offset + sequence.size] = sequence
            offset += sequence.size
        return flat

    train = concatenate(split_sequences["train"])
    validation = concatenate(split_sequences["validation"])
    probabilities = code_counts[code_counts > 0] / code_counts.sum()
    entropy_nats = float(-(probabilities * np.log(probabilities)).sum())
    metadata = {
        "files": [str(path.resolve()) for path in files],
        "utterances": split_utterances,
        "audio_hours": {
            key: value / 3600.0 for key, value in split_audio_seconds.items()
        },
        "tokens_including_boundaries": {
            "train": int(train.size),
            "validation": int(validation.size),
        },
        "codebook": {
            "used_codes": int((code_counts > 0).sum()),
            "entropy_nats": entropy_nats,
            "effective_vocabulary": float(math.exp(entropy_nats)),
        },
    }
    return train, validation, metadata


class FlatBlockDataset(Dataset):
    def __init__(self, tokens: np.ndarray, block_size: int):
        self.tokens = tokens
        self.block_size = block_size
        self.blocks = tokens.size // block_size
        if self.blocks == 0:
            raise RuntimeError("Token split is shorter than one block")

    def __len__(self) -> int:
        return self.blocks

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.block_size
        return torch.from_numpy(self.tokens[start : start + self.block_size]).long()


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    loss_sum = 0.0
    predicted_tokens = 0
    started = time.monotonic()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(input_ids=batch, labels=batch, use_cache=False)
        count = batch.shape[0] * (batch.shape[1] - 1)
        loss_sum += float(output.loss) * count
        predicted_tokens += count
    loss = loss_sum / predicted_tokens
    return {
        "loss_nats": loss,
        "perplexity": math.exp(loss),
        "bits_per_token": loss / math.log(2.0),
        "predicted_tokens": predicted_tokens,
        "tokens_per_second": predicted_tokens / (time.monotonic() - started),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.d_model % args.heads:
        raise ValueError("d-model must be divisible by heads")

    boundary_token = args.vocab_size
    train_tokens, validation_tokens, data_metadata = load_flat_splits(
        args.tokens_dir, boundary_token, args.val_permille
    )
    train_dataset = FlatBlockDataset(train_tokens, args.block_size)
    validation_dataset = FlatBlockDataset(validation_tokens, args.block_size)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = GPT2Config(
        vocab_size=args.vocab_size + 1,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.d_model,
        n_layer=args.layers,
        n_head=args.heads,
        n_inner=4 * args.d_model,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=boundary_token,
        eos_token_id=boundary_token,
        tie_word_embeddings=True,
        use_cache=False,
    )
    model = GPT2LMHeadModel(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=device.type == "cuda",
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

    epoch_steps = len(train_loader)
    planned_steps = epoch_steps * args.epochs
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    history: list[dict] = []
    global_step = 0
    best_validation = math.inf
    training_started = time.monotonic()
    stop = False
    print(
        json.dumps(
            {
                "parameters": parameter_count,
                "train_blocks": len(train_dataset),
                "validation_blocks": len(validation_dataset),
                "epoch_steps": epoch_steps,
                "total_steps": total_steps,
                "data": data_metadata,
            },
            indent=2,
        )
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_steps = 0
        window_started = time.monotonic()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(input_ids=batch, labels=batch, use_cache=False)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += float(output.loss.detach())
            running_steps += 1
            if global_step % args.log_every == 0:
                elapsed = time.monotonic() - window_started
                tokens = running_steps * args.batch_size * (args.block_size - 1)
                print(
                    f"[train] epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                    f"loss={running_loss / running_steps:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} tok_s={tokens / elapsed:.0f}"
                )
                running_loss = 0.0
                running_steps = 0
                window_started = time.monotonic()
            if global_step >= total_steps:
                stop = True
                break

        metrics = evaluate(model, validation_loader, device)
        metrics.update({"epoch": epoch + 1, "global_step": global_step})
        history.append(metrics)
        print(f"[validation] {json.dumps(metrics)}")
        checkpoint = {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "args": vars(args),
            "boundary_token": boundary_token,
            "global_step": global_step,
            "history": history,
            "data": data_metadata,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["loss_nats"] < best_validation:
            best_validation = metrics["loss_nats"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        if stop:
            break

    summary = {
        "configuration": {
            **vars(args),
            "tokens_dir": str(args.tokens_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "boundary_token": boundary_token,
            "parameters": parameter_count,
        },
        "data": data_metadata,
        "training": {
            "global_step": global_step,
            "elapsed_seconds": time.monotonic() - training_started,
            "best_validation_loss_nats": best_validation,
            "history": history,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary["training"], indent=2))


if __name__ == "__main__":
    main()
