# Real LAMOST-to-DESI publication runbook

The v1.2 run completed this protocol without relaxing a threshold after seeing
the results.

## 1. Frozen estimands

Primary estimand: the change in error from a LAMOST source holdout to a disjoint
DESI target evaluation set when the source-trained model receives no target
data. Secondary estimands cover interval coverage, paired adaptation effects,
label budgets, OOD risk, subgroups, and physical checks.

Both surveys use the exact version-pinned label scale
`APOGEE_DR12_ASPCAP_v603_calibrated_TEFF_LOGG_PARAM_M_H`. The legacy `feh`
contract field is calibrated `PARAM_M_H` ([M/H]), not elemental `FE_H`.

## 2. Frozen data contract

- LAMOST: 1,387 high-S/N public Ho et al. tutorial spectra; 1,291 pass the
  4000–5500 Å validity requirement after APOGEE quality filtering.
- DESI: APOGEE DR12 position matches within 1.5 arcsec to `desi_dr1.zpix`
  primary stars; 1,555 SPARCL spectra pass `redshift_warning == 0` and spectral
  validity checks.
- Canonical identity: `APOGEE_ID`, with zero cross-domain overlap permitted.
- Common representation: rest-frame 4000–5500 Å, R=1800, Gaussian line-spread
  approximation, source-fitted normalization and PCA.

The URLs, queries, hashes, duplicate rule, and counts are frozen in
`data/acquisition_logs/acquisition_manifest.json`.

The catalog path is independently executable through
`scripts/build_apogee_desi_crossmatch.py`. Its frozen audit table records the
full selection flow: 4,662,192 queried DESI primary stars -> 1,827 positional
matches -> 1,629 clean-ASPCAP matches -> 1,576 source-disjoint SPARCL targets
-> 1,555 final target contracts.

## 3. Fail-closed gate and observed counts

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Shared exact label scale | identical | identical | pass |
| Source training | ≥1,000 | 1,032 | pass |
| Source holdout | ≥200 | 259 | pass |
| Target adaptation | ≥100 | 467 | pass |
| Target evaluation | ≥350 | 1,088 | pass |
| Target-evaluation giants (`logg < 3.5`) | ≥50 | 1,024 | pass |
| Target-evaluation metal-poor (`[M/H] < -1.5`) | ≥50 | 55 | pass |

The quality gate is evaluated and written before model fitting. The full record is
`results/lamost_dr2_to_desi_dr1_apogee_dr12/publication_gate.json`.

## 4. Leakage and access accounting

All target APOGEE IDs are removed from source fitting. Target adaptation and
evaluation are disjoint. The feature pipeline is fit on source training only.
Methods declare one of four access levels: no target data, unlabeled target
features, target labels only for calibration, or labeled target fitting. No
target-evaluation label is used for training, tuning, or interval calibration.

## 5. Required archived evidence

The evidence bundle contains the exact configuration, input hashes, gate, split
manifest, per-star predictions, source-to-target shift table, star-bootstrap
intervals, paired adaptation effects, calibration, label budgets,
support-overlap sensitivity, formal-label-error sensitivity, OOD risk-coverage,
subgroups, ablations, physical checks, plots, notebook, and report. Raw survey
spectra are not redistributed.

## 6. Interpretation rule

The full result estimates operational LAMOST→DESI transfer, not a pure
instrument-only effect. Because source and target populations differ, quote the
full-domain and within-source-label-support findings together. The common
APOGEE scale prevents a label-version mismatch, but shared ASPCAP systematics
are not independently validated.
