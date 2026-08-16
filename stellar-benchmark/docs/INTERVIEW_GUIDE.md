# Interview and portfolio guide

## Thirty-second explanation

“I built a reliability benchmark for stellar spectroscopy and then closed its
biggest evidence gap with a real LAMOST-to-DESI transfer. I matched both surveys
to the exact APOGEE DR12 label scale, enforced object-disjoint splits and a
fail-closed sample gate, and evaluated 1,088 model-held-out DESI stars. Cross-survey
MAE roughly doubled for temperature and gravity and more than tripled for
metallicity. The model also became overconfident. Labeled target retraining
recovered much of the loss, while CORAL badly harmed metallicity. Every claim is
backed by per-star predictions, paired intervals, calibration, provenance, and
support sensitivity.”

## Resume bullets

- Built and released a prespecified quality-gated LAMOST DR2→DESI DR1 stellar-spectrum
  transfer benchmark using 2,846 quality-controlled survey contracts and a
  1,088-star model-held-out target evaluation set.
- Quantified 108%–242% cross-survey MAE degradation on a shared APOGEE DR12
  label scale; added star-bootstrap inference and a 918-star shared-support
  sensitivity analysis.
- Audited conformal uncertainty under real domain shift: nominal 90% coverage
  fell to 59%–75%, then returned near nominal after disjoint target-domain
  recalibration.
- Compared source-only, CORAL, and supervised target retraining with paired
  same-star effects; retraining improved MAE by 28%–60%, while CORAL caused
  detectable metallicity harm.
- Engineered reproducible public-data acquisition, rest-frame/resolution
  harmonization, immutable manifests, leakage controls, 51 tests, CLI workflows,
  and an archival evidence bundle.

## Role-specific framing

| Role | Lead with |
|---|---|
| Research scientist | estimands, shared reference scale, support sensitivity, calibration failure, negative CORAL result |
| ML scientist | domain adaptation, conformal recalibration, OOD risk, paired effects, label efficiency |
| ML engineer | modular contracts, deterministic splits, CLI, tests, fail-closed gate, provenance |
| Data scientist | matched comparisons, bootstrap intervals, subgroup support, budget tradeoffs |
| Research/software engineer | public survey APIs, spectrum harmonization, reproducible artifacts, honest operational limits |

## Decisions worth defending

- Why LAMOST→DESI rather than APOGEE→LAMOST spectra? Both model inputs are
  optical, so a shared wavelength experiment is physically coherent; APOGEE is
  used as the common reference-label scale.
- Why keep a source holdout? It distinguishes ordinary in-domain error from the
  operational target-domain penalty.
- Why not call CORAL neutral? Its temperature and gravity intervals include
  small benefit and harm, so the correct conclusion is inconclusive; its
  `[M/H]` harm is detectable.
- Why include out-of-support targets? They are part of operational transfer.
  The separate 918-star within-support analysis shows what changes when the
  population extrapolation is removed.
- Why is `[M/H]` labeled `feh` in files? Backward compatibility. Public language
  and manifests explicitly define the slot as APOGEE `PARAM_M_H`.

## Honest boundaries

The project does not establish absolute stellar-label truth, isolate a pure
instrument causal effect, model exact per-spectrum line-spread functions, prove
isochrone consistency, or cover B+R+Z fusion. Those limits are explicit and
machine-readable.
