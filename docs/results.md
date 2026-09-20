# Validated pilot results

This document records the compact, repository-safe summary of the QBC pilot.
Large checkpoints, frame CSVs, plots, and raw artifacts remain local under
`artifacts/` and are intentionally not committed.

## Protocol

- Base codec: released LLM-Codec AUV checkpoint.
- Training data: LibriSpeech `train-clean-100`.
- Training: 1,000 optimizer updates, gradient accumulation 8, 8,000 valid
  four-second crops per arm.
- Intervention: `lambda_mel=5`, `lambda_vq=1`, `lambda_qbc=10`.
- Control: identical initialization, seed, crop order, optimizer, and update
  count with `lambda_qbc=0`.
- Training seeds: 23, 41, and 71.
- Evaluation: 128 utterances from 32 speakers per split, at most four
  utterances per speaker; 25,600 aligned frames and 19,200 interior frames.
- Uncertainty: paired speaker-cluster bootstrap.

## Token stability

| Seed | test-clean flip change | Speaker 95% CI | test-other flip change | Speaker 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 23 | -2.25 pp | [-3.08, -1.40] | -2.52 pp | [-3.70, -1.41] |
| 41 | -2.77 pp | [-3.71, -1.88] | -1.91 pp | [-2.98, -0.82] |
| 71 | -1.75 pp | [-2.53, -0.98] | -1.57 pp | [-2.58, -0.62] |
| Mean +/- seed SD | **-2.26 +/- 0.51 pp** | 3/3 improve | **-2.00 +/- 0.48 pp** | 3/3 improve |

Interior-frame improvements average `-2.17 +/- 0.64` pp on test-clean and
`-1.64 +/- 0.34` pp on test-other.

## Matched reconstruction cost

| Split | Metric | Mean relative QBC cost | Seed SD | Seed range |
| --- | --- | ---: | ---: | ---: |
| test-clean | log-Mel | +1.11% | 0.03% | +1.07% to +1.14% |
| test-clean | MR-STFT | +0.57% | 0.11% | +0.50% to +0.70% |
| test-other | log-Mel | +1.01% | 0.13% | +0.91% to +1.15% |
| test-other | MR-STFT | +0.69% | 0.11% | +0.58% to +0.80% |

The current recipe therefore has a small but reproducible stability versus
reconstruction trade-off; it is not a lossless improvement.

## Language-model evidence

For the seed-23 checkpoint, a version-matched compact LM trained from scratch
on the complete 100.591-hour token extraction reduced validation PPL from
3,803.8 to 3,065.5 relative to released-codec tokens (`-19.41%`). All 20,480
deployed code IDs remained active. This result is seed-23-only and should not
be described as multi-seed causal evidence.

## Claim boundary

The completed experiments establish a reproducible mechanism-scale result:
direct signed-boundary training in the deployed VQ space reduces nuisance-view
token flips. They do not yet establish multilingual, full-960-hour, or
production-scale generalization. Required next steps are full LibriSpeech test
evaluation, LibriSpeech 960-hour training, and independent WenetSpeech-M
evaluation.
