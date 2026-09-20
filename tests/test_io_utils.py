from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from experiments.io_utils import decode_audio, file_sha256


def test_file_sha256_streams_known_content(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert file_sha256(path, block_size=2) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_decode_audio_downmixes_embedded_stereo_wav():
    stereo = np.asarray([[0.25, -0.25], [0.5, 0.0]], dtype=np.float32)
    payload = io.BytesIO()
    sf.write(payload, stereo, 16_000, format="WAV", subtype="FLOAT")

    waveform, sample_rate = decode_audio({"bytes": payload.getvalue()})

    assert sample_rate == 16_000
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous
    np.testing.assert_allclose(waveform, np.asarray([0.0, 0.25], dtype=np.float32))


def test_decode_audio_rejects_missing_bytes():
    try:
        decode_audio({"path": "not-embedded.wav"})
    except ValueError as error:
        assert "embedded audio bytes" in str(error)
    else:
        raise AssertionError("decode_audio should reject path-only rows")
