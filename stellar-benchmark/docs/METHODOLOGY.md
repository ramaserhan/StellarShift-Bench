# Methodology, estimands, and limitations

## Scientific questions

The benchmark separates two questions:

1. How does controlled additional noise change predictions within DESI?
2. How does a LAMOST-trained model behave on genuinely different DESI spectra?

The primary real-transfer estimand is the source-only change from LAMOST
holdout MAE to disjoint DESI target-evaluation MAE. Secondary estimands cover
calibration, paired adaptation effects, target-label budgets, OOD
risk-coverage, support overlap, subgroups, and physical checks.

## Real cross-survey data

The source is the public 1,387-star high-S/N LAMOST DR2 tutorial set accompanying
Ho et al. (2017). Its labels are direct APOGEE DR12 ASPCAP values. APOGEE DR12
`allStar-v603` supplies quality flags and formal errors.

The target begins with every `zcat_primary` stellar row in NOIRLab's
`desi_dr1.zpix` table. Unique APOGEE DR12 stars are position-matched within 1.5
arcsec, deduplicated, filtered, and retrieved through SPARCL. DESI Redrock
`redshift_warning == 0` is required. Acquisition occurred on 2026-08-16; URLs,
queries, counts, and SHA-256 hashes are frozen in the acquisition manifest.

Both domains use calibrated APOGEE DR12 ASPCAP v603 `TEFF`, `LOGG`, and
`PARAM_M_H`. The internal `feh` slot is retained for schema compatibility but
contains global `[M/H]`, not elemental `FE_H`. Formal errors are carried into a
separate sensitivity analysis.

After spectral and label quality checks, 1,291 LAMOST and 1,555 DESI stars are
available. Exact `APOGEE_ID` overlap is prohibited. Source is split 1,032/259
for training/holdout. Target is split 467/1,088 for adaptation/evaluation with
joint rare-population and temperature stratification. The evaluation set
contains 1,024 giants and 55 stars below `[M/H] = -1.5`.

## Spectrum harmonization and leakage boundary

Spectra are moved to the rest frame and resampled to a 3900–5600 Å linear
pregrid. The model uses 4000–5500 Å on a constant-log-wavelength grid. LAMOST
is treated as R=1800 and DESI as R=2500; both are matched to R=1800 using a
Gaussian line-spread approximation. The implementation refuses sharpening.

Continuum normalization, outlier thresholds, imputation, and 64-component PCA
are fit on source training only. The default model is a separate 400-tree
ExtraTrees regressor for each target. No target-evaluation spectrum or label is
used for fitting, tuning, or calibration.

## Accuracy and uncertainty

Point metrics are MAE, RMSE, bias, robust scatter, and R². Target-domain metric
intervals use 1,000 star-bootstrap replicates; adaptation effects resample
paired per-star loss differences. A 5% baseline-MAE planning margin prevents
failure to reject zero from being described as equivalence.

The primary source-to-target MAE change uses an independent two-sample
bootstrap. Each replicate resamples the 259 source-holdout stars and the 1,088
target-evaluation stars separately, recomputes both MAEs, and records their
difference and percentage change. This propagates finite-sample uncertainty
from both domains instead of treating the source MAE as fixed. A full-data rerun
writes `source_holdout_per_star_predictions.csv` and
`domain_shift_intervals.csv`; the earlier frozen run predates those artifacts.

Split-conformal intervals are fit separately per method and target on source
holdout residuals, then evaluated on DESI. A second audit recalibrates the
source-only and CORAL intervals using only the target adaptation partition.
Nominal coverage is 90%.

## Adaptation and label budgets

- Source-only: no target information.
- CORAL: aligns source feature covariance to 467 unlabeled target-adaptation
  spectra; the target labels remain hidden.
- Labeled retraining: combines source training and 467 target-adaptation stars,
  with domain-balanced sample weights.
- Label budgets: repeated random subsets of 5, 10, 25, 50, and 100 target labels,
  ten draws per budget, evaluated on the same frozen target set.

## Support, OOD, and subgroups

The primary effect intentionally reflects operational transfer, including
population shift. A prespecified sensitivity restricts target evaluation to
stars jointly within the source-training minima/maxima for all three labels;
918/1,088 remain. This rectangular support definition is transparent and
conservative, not a claim of perfect density overlap.

A source-fitted Mahalanobis score ranks target spectra for risk-coverage curves.
Subgroup tables stratify by target S/N and true-label terciles. Estimator-family
ablation compares Ridge, ExtraTrees, and MLP under the same source/target split.

## Reference-label and physical checks

The APOGEE formal-error sensitivity resamples stars and adds independent,
unbiased Gaussian noise using the reported one-sigma errors while preserving
the same perturbed label across methods. This tests sensitivity to reported
random error only; it does not validate APOGEE, model correlated errors, or
address shared pipeline systematics.

Hard bounds are checked for `teff`, `logg`, and `[M/H]`. No violations occur.
The optional isochrone-manifold check requires an explicitly cited,
population-appropriate grid; none was supplied, so that audit is reported as
not evaluated.

## Main results

Source-only MAE increases from 52.5 to 109.4 K (`teff`), 0.124 to 0.260 dex
(`logg`), and 0.073 to 0.250 dex (`[M/H]`): +108.4%, +110.0%, and +241.6%.
Within joint source-label support, target MAEs are 82.0 K, 0.163 dex, and 0.129
dex, still +56.3%, +32.3%, and +76.4% relative to source holdout.

CORAL changes MAE by +3.7%, +1.6%, and +57.5%; only the `[M/H]` harm is clearly
detected. Labeled retraining changes MAE by −28.3%, −40.9%, and −60.0%, with all
paired intervals excluding zero. Source-calibrated 90% coverage is 67.8%,
75.2%, and 59.0%; target-adaptation recalibration restores source-only coverage
to 90.3%, 90.2%, and 89.9%.

The controlled DESI R-arm experiment remains separate: at 2× noise, `teff` MAE
increases by 29.8% with a nested 95% interval of +19.4% to +43.6%. It is a
measurement-shift comparison, not a causal decomposition of the real survey
effect.

## Limitations and allowed claims

Allowed: executed LAMOST→DESI transfer on a shared APOGEE DR12 scale; real-shift
accuracy and calibration degradation; paired adaptation findings; label-budget,
support, OOD, subgroup, ablation, formal-error-sensitivity, and hard-bound
results for the frozen data.

Not allowed: independent APOGEE label validation; a pure instrument-only causal
effect; generalization to all LAMOST/DESI populations, other releases, arms, or
label pipelines; exact line-spread-function conclusions; isochrone consistency;
or a DOI until the archive is deposited.
