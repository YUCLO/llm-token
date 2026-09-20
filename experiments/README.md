# Experiment module map

The scripts remain flat so existing commands and artifact provenance do not
break. Their responsibilities are grouped below.

## Geometry and intervention

- `vq_geometry.py`: exact normalized VQ boundary geometry and the `D < R`
  certificate.
- `qbc_loss.py`: signed teacher-cell margin and detached-codebook QBC hinge.
- `qbc_diagnostic.py`: full-vs-slice token stability diagnostics.
- `train_qbc_codec.py`: matched codec fine-tuning with optional QBC and gradient
  accumulation.

## Reconstruction and statistics

- `eval_codec_reconstruction.py`: paired reconstruction evaluation.
- `compare_diagnostics.py`: frame-paired stability comparison with speaker- or
  utterance-cluster bootstrap.
- `aggregate_qbc_multiseed.py`: effect aggregation across independent training
  seeds.
- `stats_utils.py`: shared bootstrap/distribution helpers. Always reduce frames
  to the intended independent unit before calling the bootstrap helper.
- `io_utils.py`: shared embedded-audio decoding, Parquet iteration, and hashing.

## Token-language-model evaluation

- `extract_codec_tokens.py`: codec-token extraction from local Parquet audio.
- `train_small_token_lm.py`: version-matched compact causal LM training.
- `common_context_small_lm.py`: fixed-prefix substitution cost under that LM.
- `compare_common_context_runs.py`: paired comparison of common-context runs.
- `compare_token_lm_runs.py`: intrinsic token-LM comparison.
- `common_context_nll.py`, `compare_token_versions.py`: released-model/version
  diagnostics retained for provenance.
- `plot_qbc_lm_chain.py`: final mechanism-chain plots.

## Invariants

- Control and QBC arms must share initialization, seed, crop order, optimizer,
  reconstruction loss, and update count; only the QBC coefficient differs.
- Quantization geometry is measured after `in_proj` and L2 normalization over
  the complete deployed codebook.
- QBC detaches the codebook only on the QBC path; the original VQ loss retains
  its normal update path.
- Crop starts are multiples of the 320-sample hop.
- Primary uncertainty intervals resample speakers, not frames.
- Reconstruction is compared with the matched control, not only the released
  checkpoint.
