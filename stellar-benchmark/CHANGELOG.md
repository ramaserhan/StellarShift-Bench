# Changelog

## 1.2.3 - 2026-08-16

- Replaced the stale standalone notebook with the release-built notebook and
  made every code cell runnable against archived result tables and figures.
- Added an independent two-sample star bootstrap for source-holdout to
  target-evaluation MAE changes, plus auditable source-holdout predictions on
  full data reruns.
- Added the missing APOGEE DR12 to DESI DR1 crossmatch builder and an explicit
  `4,662,192 -> 1,827 -> 1,629 -> 1,576 -> 1,555` selection audit.
- Replaced ambiguous publication language with a prespecified quality-gate
  description and clarified that target evaluation is model-held-out.
- Expanded the deterministic suite to 51 tests and retained the v1.2.2 release
  integrity and portable Colab fixes.

## 1.2.2 - 2026-08-16

- Made the release builder portable outside the original nested workspace,
  eliminating a test-collection failure at `/content/stellar-benchmark`.
- Added a fail-fast Google Colab workflow that validates acquisition inputs and
  stops on the first failed subprocess instead of surfacing a later missing file.
- Isolated SPARCL 1.3.0 in its own virtual environment so its pandas constraint
  cannot downgrade or corrupt the current Google Colab scientific stack.
- Added automatic LAMOST extraction-path detection and explicit four-chunk
  SPARCL validation before building the cross-survey contracts.

## 1.2.1 - 2026-08-16

- Rebuilt the instant-results notebook and checksum manifest from the same
  deterministic release step, closing the reported notebook-integrity gap.
- Added an atomic release builder for the notebook, figures, PDFs, source ZIP,
  evidence ZIP, and SHA-256 manifest with fixed archive timestamps and ordering.
- Added fail-closed verification that standalone artifacts match both ZIP copies
  and that every checksum still matches the final release bytes.
- Added regression tests for deterministic archives, checksum parsing, checksum
  verification, and archive-member identity.

## 1.2.0 - 2026-08-16

- Executed the prespecified quality-gated LAMOST DR2 to DESI DR1 transfer on 1,291
  usable source stars and 1,555 target stars with a 1,088-star model-held-out target
  evaluation set.
- Put both surveys on the exact APOGEE DR12 ASPCAP v603 calibrated label scale;
  the legacy `feh` contract slot is explicitly documented as global
  `PARAM_M_H` ([M/H]), not elemental `FE_H`.
- Added reproducible public-data acquisition and contract-building workflows,
  immutable input hashes, object-disjoint manifests, and a fail-closed gate that
  passed every prespecified sample and subgroup threshold.
- Added direct source-holdout versus target-evaluation shift estimates,
  shared-label-support sensitivity, formal reference-label-error propagation,
  target-label-budget trials, calibration audits, paired adaptation intervals,
  model-family ablations, OOD risk-coverage, and physical hard-bound checks.
- Found large real-survey degradation, severe source-calibrated undercoverage,
  substantial recovery from labeled target retraining, and metallicity harm
  from CORAL. Small labeled budgets can hurt, while larger budgets help.
- Updated the technical report, portfolio brief, instant-results notebook,
  methods, execution status, critique response, and archival release.

## 1.1.0 - 2026-08-16

- Reframed the public subtitle as a verified DESI S/N reliability case study
  plus a prespecified quality-gated cross-survey protocol; no real survey result is
  implied by the benchmark name.
- Added paired same-star adaptation-effect intervals. CORAL's temperature
  effect is now reported as inconclusive, not neutral or equivalent.
- Added a prospective precision plan with minimum detectable effects and sample
  sizes. The current variance implies about 344 target-evaluation stars to
  resolve a 5% CORAL temperature-MAE effect at 80% power.
- Added a 5,000-replicate sensitivity analysis that propagates reported RVSpecFit
  formal errors while explicitly excluding pipeline-systematic validation.
- Added a fail-closed cross-survey quality gate requiring a common exact
  label scale, adequate source/target partitions, and giant/metal-poor support.
- Added a cross-survey publication runbook, an adaptation-effect figure, five
  tests, and revised report, notebook, and interview language.

## 1.0.0 - 2026-08-16

- Added per-star predictions and 95% intervals for every clean, shifted, and
  adapted core metric.
- Added nested star×noise-seed bootstrap inference so perturbation realizations
  are not treated as independent stellar evidence.
- Added split-conformal 68%, 90%, and 95% interval audits under clean and shifted
  data, plus disjoint target-domain recalibration.
- Added source-fitted Mahalanobis OOD scores, per-seed risk–coverage curves,
  subgroup support accounting, and hard physical-bound checks.
- Added source noise augmentation, unlabeled CORAL, labeled retraining, and
  repeated target-label-budget comparisons with explicit access tags.
- Added Ridge/ExtraTrees/MLP and PCA-capacity ablations.
- Added a survey-neutral LAMOST→DESI optical transfer engine with strict
  object-disjoint splits, shared log-wavelength resampling, resolution matching,
  calibration, OOD, bootstrap, adaptation, subgroup, and plausibility audits.
- Generalized DESI extraction to the B, R, and Z arms and added declared B-arm
  and coadd→single-epoch follow-up configurations.
- Added four executed-result figures, an instant-results notebook, a technical
  report, expanded methodology, and 38 passing tests.
- Preserved the v0.3 scientific result while clearly separating executed DESI
  claims from data-ready but unexecuted real cross-survey work.

## 0.3.0 - 2026-08-15

- Added validated DESI DR1 R-arm FITS extraction and compact NPZ caching.
- Added explicit quality cuts, duplicate handling, temperature/S/N
  stratification, and leakage checks keyed by `TARGETID`.
- Added source-fitted continuum normalization, artifact masking, imputation,
  PCA, and separate ExtraTrees regressors for each physical target.
- Added deterministic per-star S/N perturbations across five noise levels and
  ten random realizations.
- Added labeled target retraining and target-label-free noise augmentation;
  retained negative and nonsignificant adaptation results.
- Added real-case-study tables, two figures, a Colab workflow, methodology and
  interview documentation, CI, and six additional tests (24 total).

## 0.2.0 - 2026-08-15

- Added an installable `stellar-benchmark` CLI and validated YAML configuration.
- Added deterministic source-domain, zero-shot target-domain, and supervised
  target-retraining smoke experiments with six saved artifacts.
- Added leakage-safe splitting that groups repeated spectra by `source_id`.
- Added MAE and RMSE, robust input validation, safer stratification, and an
  identical-residual guard for matched significance testing.
- Renamed the implemented adaptation method as retraining; retained a
  deprecated compatibility wrapper for the former fine-tuning name.
- Clarified that bootstrap ensemble spread is an uncertainty proxy.
- Added a validated shared-schema CSV loader and a tiny committed fixture.
- Expanded coverage to 18 configuration, metric, split, loader, model, harness, adaptation,
  and end-to-end smoke tests.
- Replaced aspirational README claims with verified commands and an explicit
  real-data roadmap.
