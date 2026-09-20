# Boundary-stable speech-token experiments

This workspace studies whether directly regularizing the deployed vector
quantizer's decision boundary produces more stable and language-model-friendly
speech tokens than continuous latent consistency alone.

## Repository layout

```text
experiments/   training, geometry, reconstruction, and token-LM programs
scripts/       dataset download utilities
tests/         CPU unit tests for geometry, QBC, I/O, and statistics
checkpoints/   local pretrained weights (ignored by Git)
data/          local LibriSpeech/WenetSpeech data (ignored by Git)
artifacts/     checkpoints, CSV/JSON results, plots, and reports (ignored by Git)
sources/       local AUV and LLM-Codec source checkouts
```

See [experiments/README.md](experiments/README.md) for the module-level map and
the invariants that matched experiments must preserve.

The full Chinese research rationale, mathematical formulation, experimental
process, evidence boundary, and next-step plan are documented in
[docs/research_idea_and_process_zh.md](docs/research_idea_and_process_zh.md).

The third-party source trees are Git submodules. After cloning this repository,
initialize them with:

```bash
git submodule update --init --recursive
```

## Environment

The validated local environment uses Python 3.11, PyTorch/Torchaudio
`2.3.1+cu121`, and the remaining pinned packages in `requirements.txt`.
Install the CUDA-matched PyTorch pair first; do not let a generic dependency
installation replace it with a different build. The AUV and LLM-Codec sources
are loaded from `sources/` by the experiment entrypoints.

## Validated pilot recipe

The matched control and QBC arms use the same seed and sample order:

```bash
.venv/bin/python experiments/train_qbc_codec.py \
  --input-dir data/LibriSpeech/modelscope_parquet/all/train.clean.100 \
  --auv-ckpt checkpoints/llm-codec.pt \
  --output-dir artifacts/qbc_control_seed23 \
  --device cuda:0 --seed 23 --max-steps 1000 --grad-accum-steps 8 \
  --lambda-mel 5 --lambda-vq 1 --lambda-qbc 0

.venv/bin/python experiments/train_qbc_codec.py \
  --input-dir data/LibriSpeech/modelscope_parquet/all/train.clean.100 \
  --auv-ckpt checkpoints/llm-codec.pt \
  --output-dir artifacts/qbc_qbc10_seed23 \
  --device cuda:1 --seed 23 --max-steps 1000 --grad-accum-steps 8 \
  --lambda-mel 5 --lambda-vq 1 --lambda-qbc 10
```

Run a speaker-balanced diagnostic with:

```bash
.venv/bin/python experiments/qbc_diagnostic.py \
  --parquet data/LibriSpeech/modelscope_parquet/all/test.clean/0000.parquet \
  --auv-ckpt artifacts/qbc_qbc10_seed23/final.pt \
  --output-dir artifacts/qbc_qbc10_seed23/test_clean_128_spkcap4 \
  --device cuda:0 --max-utterances 128 --max-utterances-per-speaker 4 \
  --seed 17 --frame-chunk 512
```

Use `compare_diagnostics.py` for a paired control/QBC comparison and
`aggregate_qbc_multiseed.py` only after every seed/split comparison is complete.

## Current evidence boundary

- Three training seeds reduce full-vs-slice flip rate by `2.26 +/- 0.51` pp on
  LibriSpeech test-clean and `2.00 +/- 0.48` pp on test-other.
- The matched spectral reconstruction cost is approximately `0.5--1.1%`.
- The full intrinsic token-LM experiment exists only for seed 23.
- These results establish a reproducible mechanism-scale intervention, not yet
  broad multilingual or production-scale generalization.

The repository-safe result ledger is [docs/results.md](docs/results.md). Large
checkpoints, frame CSVs, plots, and raw result artifacts stay local and are not
committed.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile experiments/*.py scripts/*.py
```
