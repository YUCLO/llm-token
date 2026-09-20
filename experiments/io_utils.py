"""Shared filesystem and embedded-audio helpers for experiment scripts."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


LIBRISPEECH_COLUMNS = ("audio", "id", "text", "speaker_id", "chapter_id")


def file_sha256(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of ``path`` without loading it all into memory."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_audio(audio: dict) -> tuple[np.ndarray, int]:
    """Decode a Hugging Face/ModelScope embedded-audio cell to mono float32."""

    payload = audio.get("bytes")
    if payload is None:
        raise ValueError("Parquet row does not contain embedded audio bytes")
    waveform, sample_rate = sf.read(
        io.BytesIO(payload),
        dtype="float32",
        always_2d=True,
    )
    mono = np.ascontiguousarray(waveform.mean(axis=1))
    return mono, int(sample_rate)


def iter_audio_rows(
    path: Path,
    *,
    columns: Sequence[str] = LIBRISPEECH_COLUMNS,
) -> Iterator[dict]:
    """Stream embedded-audio Parquet rows one at a time in stable file order."""

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1, columns=list(columns)):
        yield batch.to_pylist()[0]
