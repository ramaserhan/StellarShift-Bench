# StellarShift-Bench v1.2.3 evidence bundle

The bundle is designed for scientific review without redistributing raw survey
spectra. All paths are relative to `stellar-benchmark/`.

## Start here

1. `output/pdf/StellarShift_Bench_v1.2.3_Portfolio_Brief.pdf` - one-page result.
2. `output/pdf/StellarShift_Bench_v1.2.3_Technical_Report.pdf` - methods, results,
   uncertainty, limitations, and references.
3. `examples/StellarShift_v1.2.3_Instant_Results.ipynb` - a runnable notebook
   that loads the archived tables and figures, with verified outputs embedded.
4. `results/lamost_dr2_to_desi_dr1_apogee_dr12/README.md` - artifact index.

## Scientific evidence

- `publication_gate.json`: passed prespecified minimum support.
- `domain_shift_summary.csv`: source holdout versus target transfer.
- `domain_shift_intervals.csv`: two-sample star bootstrap over both the source
  holdout and target evaluation set, written by a v1.2.3 full-data rerun. It is
  not present in the earlier frozen result directory packaged here.
- `source_holdout_per_star_predictions.csv`: auditable source-holdout inputs to
  that bootstrap, with the same full-rerun status.
- `per_star_predictions.csv`: auditable predictions for all methods.
- `star_bootstrap_intervals.csv`: target-domain metric intervals.
- `adaptation_effect_intervals.csv`: paired method effects and conclusions.
- `calibration_cross_survey.csv`: coverage and width by method/calibration set.
- `support_overlap_metrics.csv`: within-source-label-support sensitivity.
- `reference_label_sensitivity.csv`: formal APOGEE error propagation.
- `label_budget_trials.csv`: repeated target-label acquisition trials.
- `risk_coverage.csv`, `subgroup_metrics.csv`, `model_ablation.csv`, and
  `physical_plausibility.json`: robustness checks.

The earlier controlled DESI reliability directory is included so reviewers can
reproduce the real-versus-synthetic shift comparison.

## Provenance

- `configs/lamost_to_desi.yaml`: exact experiment configuration.
- `data/acquisition_logs/acquisition_manifest.json`: source URLs, catalog query,
  counts, matching decisions, and immutable hashes.
- `data/acquisition_logs/apogee_dr12_desi_dr1_target_selected.csv`: frozen public
  target selection.
- `data/acquisition_logs/apogee_dr12_desi_dr1_selection_flow.csv`: explicit
  4,662,192 -> 1,827 -> 1,629 -> 1,576 -> 1,555 selection audit.
- `scripts/build_apogee_desi_crossmatch.py`: executable positional crossmatch
  and source-overlap removal workflow.
- `data/acquisition_logs/sparcl_find_target_rv_response.json`: frozen DESI
  redshift/warning metadata used by the contract builder.
- `manifest.json` and `split_manifest.csv`: computation and partition record.

Raw LAMOST/DESI spectra and generated NPZ arrays are not included. Rebuild them
using `data/README.md` and the acquisition scripts in the source release.

## Semantic warning

The legacy result target named `feh` contains APOGEE DR12 calibrated
`PARAM_M_H`, global `[M/H]`, not elemental `FE_H`. The primary shift combines
instrument/reduction/S/N and population shift. The common APOGEE scale is not
independent absolute truth.
