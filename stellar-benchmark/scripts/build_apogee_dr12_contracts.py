"""Build the real LAMOST→DESI APOGEE-DR12 survey contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from stellar_benchmark.data.apogee_dr12 import build_apogee_dr12_contracts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lamost-labels", required=True, type=Path)
    parser.add_argument("--lamost-spectra", required=True, type=Path)
    parser.add_argument("--apogee-allstar", required=True, type=Path)
    parser.add_argument("--desi-selection", required=True, type=Path)
    parser.add_argument("--desi-pickle-glob", required=True)
    parser.add_argument("--desi-redshifts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    pickles = sorted(Path().glob(args.desi_pickle_glob))
    if not pickles:
        raise FileNotFoundError(f"no SPARCL pickle files matched {args.desi_pickle_glob}")
    output = args.output_dir
    artifacts = build_apogee_dr12_contracts(
        lamost_labels_fits=args.lamost_labels,
        lamost_spectra_dir=args.lamost_spectra,
        apogee_allstar_fits=args.apogee_allstar,
        desi_selected_csv=args.desi_selection,
        desi_sparcl_pickles=pickles,
        desi_redshift_json=args.desi_redshifts,
        source_output=output / "lamost_apogee_dr12.npz",
        target_output=output / "desi_apogee_dr12.npz",
        source_manifest_output=output / "lamost_apogee_dr12_manifest.csv",
        target_manifest_output=output / "desi_apogee_dr12_manifest.csv",
    )
    for name, value in artifacts.items():
        print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
