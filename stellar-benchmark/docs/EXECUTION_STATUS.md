# Execution status

This table prevents implemented capability from being confused with executed
scientific evidence.

| Workstream | Status | Evidence in v1.2 | Claim allowed |
|---|---|---|---|
| LAMOST DR2→DESI DR1 transfer | **Executed; gate passed** | 1,291 usable source, 1,555 target, 1,088 target-evaluation stars | Accuracy degradation on the shared APOGEE DR12 scale |
| Cross-survey calibration | **Executed** | Method-specific source calibration plus disjoint target recalibration | Source intervals under-cover; target recalibration restores approximately nominal coverage |
| CORAL and labeled retraining | **Executed** | Paired same-star effects on 1,088 target stars | CORAL harms `[M/H]`; labeled retraining improves all targets; Teff/logg CORAL effects remain inconclusive |
| Target label budgets | **Executed** | 5–100 labels, ten repeated draws | Very small budgets can hurt; 100 labels improves median MAE for all targets |
| Shared-support sensitivity | **Executed** | 918 target stars inside joint source label bounds | Population support explains part, not all, of the transfer loss |
| APOGEE formal-error sensitivity | **Executed** | 1,000 paired star/error replicates | Adaptation conclusions are stable to reported random formal errors under stated assumptions |
| Controlled DESI R-arm S/N | **Executed** | 901 selected stars, five factors, ten seeds | Synthetic measurement-shift accuracy and calibration findings |
| Model-family ablation | **Executed** | Ridge, ExtraTrees, MLP | No single family has the best target MAE on every parameter |
| OOD and subgroup audit | **Executed** | Risk-coverage and S/N/label-regime tables | Selective prediction is useful but target-dependent |
| Hard physical bounds | **Executed** | `physical_plausibility.json` | No hard-bound violations in the evaluated predictions |
| Isochrone-manifold plausibility | Implemented, not evaluated | Optional cited-grid interface | No isochrone-consistency claim |
| B+R+Z feature fusion | Not executed | Per-arm extraction exists, joint model absent | No multi-arm claim |
| DOI archive | Prepared, not deposited | Citation metadata and release bundle | No DOI until human deposit |

The common APOGEE label scale is a consistency device, not independent truth.
The primary cross-survey result combines instrument and population/covariate
shift; `support_overlap_metrics.csv` is the narrower support-controlled result.
