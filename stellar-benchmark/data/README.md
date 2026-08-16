# Data acquisition and local contracts

Raw spectra and processed spectral arrays are intentionally excluded from the
public source archive. The release does include the frozen target selection,
SPARCL query metadata, checksums, acquisition code, and contract builder needed
to reconstruct the v1.2 LAMOST DR2 to DESI DR1 experiment.

## Real cross-survey data

Download and verify the version-pinned Ho et al. tutorial files and APOGEE DR12
catalog:

```bash
python scripts/acquire_real_cross_survey.py static \
  --output-dir data/raw/catalogs \
  --extract-lamost
```

The exact NOIRLab Data Lab SQL used to enumerate DESI DR1 primary stars is
frozen in the acquisition script:

```bash
python scripts/acquire_real_cross_survey.py catalog-queries
```

Save the two query results using the filenames recorded in the acquisition
manifest, then independently rebuild the 1.5-arcsec APOGEE-DESI crossmatch and
the source-disjoint SPARCL target selection:

```bash
python scripts/build_apogee_desi_crossmatch.py \
  --apogee-allstar data/raw/catalogs/apogee_dr12_allStar-v603.fits \
  --desi-catalog \
    data/raw/catalogs/desi_dr1_main_bright_stars.csv \
    data/raw/catalogs/desi_dr1_other_primary_stars.csv \
  --lamost-labels data/raw/catalogs/lamost_labels_apogee_dr12.fits \
  --crossmatch-output data/acquisition_logs/apogee_dr12_desi_dr1_crossmatch.csv \
  --selection-output data/acquisition_logs/apogee_dr12_desi_dr1_target_selected.csv \
  --flow-output data/acquisition_logs/apogee_dr12_desi_dr1_selection_flow.csv
```

The frozen selection flow is explicit: 4,662,192 queried primary stellar rows
produce 1,827 APOGEE matches; 198 nonzero ASPCAPFLAG rows and 53 LAMOST-source
overlaps are removed, leaving 1,576 SPARCL requests. Twenty-one nonzero
Redrock-warning rows are then excluded, leaving the 1,555-row target contract.

The expensive positional crossmatch was frozen at
`data/acquisition_logs/apogee_dr12_desi_dr1_target_selected.csv`. Retrieve those
exact 1,576 DESI spectra anonymously through SPARCL:

```bash
python scripts/acquire_real_cross_survey.py sparcl \
  --selected-csv data/acquisition_logs/apogee_dr12_desi_dr1_target_selected.csv \
  --output-dir data/acquisition_logs/sparcl_chunks
```

Build the object-disjoint contracts:

```bash
python scripts/build_apogee_dr12_contracts.py \
  --lamost-labels data/raw/catalogs/lamost_labels_apogee_dr12.fits \
  --lamost-spectra data/raw/catalogs/lamost_ho2017_spectra/spectra \
  --apogee-allstar data/raw/catalogs/apogee_dr12_allStar-v603.fits \
  --desi-selection data/acquisition_logs/apogee_dr12_desi_dr1_target_selected.csv \
  --desi-pickle-glob 'data/acquisition_logs/sparcl_chunks/spectra_*.pkl' \
  --desi-redshifts data/acquisition_logs/sparcl_find_target_rv_response.json \
  --output-dir data/processed
```

Verify all immutable inputs and sample counts against
`data/acquisition_logs/acquisition_manifest.json`. Both survey contracts use
calibrated APOGEE DR12 ASPCAP v603 `TEFF`, `LOGG`, and `PARAM_M_H`. The legacy
`feh` field stores `PARAM_M_H` ([M/H]), not elemental `FE_H`.

## Earlier DESI-only extraction

The general DESI FITS extractor remains available for controlled S/N studies:

```bash
python examples/02_download_desi_subset.py --output-dir data/desi_fits
stellar-benchmark desi-extract \
  --input-glob "data/desi_fits/*.fits" \
  --output data/processed/desi_r_raw.npz
```
