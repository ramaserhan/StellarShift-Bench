# Critique response

The v1.2 work directly addresses the earlier ceiling on scientific scope,
statistical power, and generalizability.

| Earlier critique | v1.2 response | Evidence |
|---|---|---|
| Benchmark name exceeded the executed scope | Ran genuine LAMOST DR2→DESI DR1 optical transfer | Passed quality gate and 21-artifact result directory |
| Only 91 target-evaluation stars | Expanded real-transfer evaluation to 1,088 stars | Star-bootstrap MAE intervals are materially narrower |
| One survey and one synthetic shift | Put controlled DESI noise and real cross-survey shift side by side | `domain_shift_summary.csv` and report comparison |
| One label pipeline and possible scale mismatch | Used exact APOGEE DR12 ASPCAP v603 labels in both domains | Version-pinned label-scale gate and input hashes |
| Accuracy without honest uncertainty under shift | Audited split-conformal coverage and disjoint target recalibration | 90% source coverage falls to 59%–75%; recalibration returns near 90% |
| Weak adaptation comparison | Compared no adaptation, CORAL, and labeled retraining with paired CIs | CORAL harms `[M/H]`; retraining improves every target |
| Unknown label efficiency | Repeated target-label budgets from 5 to 100 | Tiny budgets can hurt; 100 labels helps all targets |
| Population mismatch could explain the result | Added joint source-label-support sensitivity | 918 supported targets still show 32%–76% degradation |
| No reference-label uncertainty analysis | Propagated APOGEE formal errors in paired simulations | Main adaptation signs remain stable under stated assumptions |
| Project lacked a standalone artifact | Updated report, one-page brief, notebook, provenance, and archive | v1.2 evidence bundle |

## What the result now shows

Source-only cross-survey MAE increases by 108.4% for `teff`, 110.0% for `logg`,
and 241.6% for `[M/H]`. Labeled retraining with 467 target stars recovers
28.3%, 40.9%, and 60.0% relative to source-only transfer. Nominal 90% intervals
calibrated in the source domain become overconfident on DESI; target adaptation
labels can recalibrate them without using the evaluation set.

The controlled 2× DESI noise experiment raised `teff` MAE by 29.8%, so the real
survey shift is substantially more damaging in this benchmark. This comparison
is descriptive rather than causal because the shifts differ in mechanism.

## Remaining ceilings

The result is not independent validation of APOGEE labels, and the full-domain
effect mixes survey/instrument, reduction, S/N, and population shift. The Ho
high-S/N source lacks `[M/H] < -1.5` stars after quality cuts; those target stars
are intentionally visible as out-of-support cases. The target is dominated by
giants and DESI backup-program spectra. Resolution matching is approximate, and
no population-appropriate isochrone grid was evaluated. These are next-paper
questions rather than hidden caveats.
