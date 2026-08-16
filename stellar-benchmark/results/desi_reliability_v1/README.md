# Verified DESI v1.1 result bundle

Run date: 2026-08-16. Input: 1,040 extracted DESI DR1 R-arm spectra; 901
quality-selected unique stars. Partitions: 540 source training, 180 source
holdout, 90 target adaptation, 91 target evaluation.

Primary result at 2x noise:

| Target | Shifted MAE | 95% interval | Change from clean | Change 95% interval |
|---|---:|---:|---:|---:|
| Teff | 159.190 K | 132.553-186.654 | +29.818% | +19.352%-+43.553% |
| Logg | 0.226 dex | 0.182-0.280 | +4.195% | -0.589%-+10.227% |
| FeH | 0.215 dex | 0.179-0.250 | +3.448% | -1.626%-+9.607% |

Nominal 90% source-calibrated Teff coverage is 92.3% on clean target spectra,
85.9% at 2x noise, and 77.3% at 3x noise. At the fixed 2x perturbation seed,
source noise augmentation reduces Teff MAE from 171.062 K to 156.676 K
(-8.410%; paired 95% interval: -23.819 to -5.042 K). CORAL's paired effect is
+0.469 K with a 95% interval from -11.416 to +12.069 K; this is inconclusive,
not evidence of neutrality or equivalence.

The prospective precision plan estimates that 344 target-evaluation stars are
needed for 80% power to resolve a 5% CORAL effect on Teff MAE. Propagating the
reported independent Gaussian RVSpecFit formal errors yields a median Teff
change of +28.077% (95% interval: +18.041% to +41.640%). This sensitivity check
does not validate the reference pipeline or model shared systematics.

`manifest.json` records the complete experiment contract and caveats. The
science-facing interpretation belongs to `output/pdf/StellarShift_Bench_v1_Technical_Report.pdf`.
No cross-arm or real LAMOST-to-DESI claim is included in this result directory.
