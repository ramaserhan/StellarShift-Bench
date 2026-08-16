# Executed LAMOST DR2 to DESI DR1 result

This directory is the frozen v1.2 evidence from the prespecified quality-gated run in
`configs/lamost_to_desi.yaml`.

Key facts:

- Gate passed: 1,032 source train, 259 source holdout, 467 target adaptation,
  and 1,088 model-held-out target-evaluation stars.
- Shared labels: calibrated APOGEE DR12 ASPCAP v603 `TEFF`, `LOGG`, and
  `PARAM_M_H`; the files' `feh` target means global `[M/H]`.
- Source-only cross-survey MAE change: +108.4% (`teff`), +110.0% (`logg`), and
  +241.6% (`[M/H]`).
- Labeled target retraining changes MAE by -28.3%, -40.9%, and -60.0%.
- Source-calibrated nominal 90% coverage falls to 67.8%, 75.2%, and 59.0%;
  disjoint target recalibration returns it near 90%.

Start with:

- `publication_gate.json` - prespecified readiness thresholds and observed counts.
- `domain_shift_summary.csv` - direct LAMOST holdout versus DESI transfer effect.
- `domain_shift_intervals.csv` - independent two-sample star-bootstrap interval,
  written on a v1.2.3 full-data rerun (not present in this earlier frozen run).
- `source_holdout_per_star_predictions.csv` - auditable source bootstrap input,
  written on a v1.2.3 full-data rerun (not present in this earlier frozen run).
- `transfer_metrics.csv` - all method/target point metrics.
- `star_bootstrap_intervals.csv` - target-domain uncertainty intervals.
- `adaptation_effect_intervals.csv` - paired same-star method effects.
- `calibration_cross_survey.csv` - coverage and interval width by calibration domain.
- `support_overlap_metrics.csv` - 918-star within-source-label-support sensitivity.
- `reference_label_sensitivity.csv` - APOGEE formal-error propagation.
- `per_star_predictions.csv` - auditable target-evaluation predictions.
- `manifest.json` and `split_manifest.csv` - exact computational provenance.

The primary result estimates operational cross-survey transfer and combines
instrument/reduction/S/N and population shift. The common APOGEE label scale is
not independent ground truth. See `docs/METHODOLOGY.md` for the full claim
boundary.
