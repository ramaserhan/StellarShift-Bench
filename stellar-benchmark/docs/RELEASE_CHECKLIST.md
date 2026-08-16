# v1.2.3 archive checklist

- [x] Freeze version `1.2.3` in package, citation, Zenodo, and changelog metadata.
- [x] Pass 51 tests and compile all package, script, and test modules.
- [x] Pass the prespecified cross-survey quality gate before fitting.
- [x] Freeze source/target hashes, data provenance, split manifest, and config.
- [x] Archive per-star predictions, metric intervals, paired effects, calibration,
  label budgets, support sensitivity, OOD, subgroups, ablations, and physical checks.
- [x] State that `feh` is APOGEE DR12 `PARAM_M_H` ([M/H]).
- [x] Separate full operational shift from within-source-support sensitivity.
- [x] Update README, methodology, critique response, interview guide, report,
  one-page brief, and instant-results notebook.
- [x] Exclude raw survey spectra and processed spectral arrays from public ZIPs.
- [x] Generate the notebook, PDFs, ZIPs, and checksum manifest from one atomic,
  deterministic command; verify standalone bytes against both archive copies.
- [x] Regenerate all derived artifacts and confirm identical SHA-256 hashes.
- [x] Verify PDFs visually.
- [x] Validate the release from a shallow `/content/stellar-benchmark`-style path.
- [x] Isolate SPARCL dependencies from the Google Colab scientific environment.
- [ ] Create a signed repository tag `v1.2.3` after syncing to the external repo.
- [ ] Deposit the release on Zenodo and add the assigned DOI to metadata.
