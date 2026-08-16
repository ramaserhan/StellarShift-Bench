"""Acquire the public inputs for the v1.2 LAMOST-to-DESI experiment.

This script deliberately separates the multi-million-row DESI catalog query
from the version-pinned spectral retrieval.  The public release includes the
frozen 1,576-row target selection and SPARCL find response, so the expensive
catalog crossmatch does not have to be repeated merely to reproduce the model
run.  See ``data/acquisition_logs/acquisition_manifest.json`` for hashes and
the exact selection contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import tarfile
from urllib.request import urlopen

import pandas as pd


STATIC_FILES = {
    "lamost_spectra_ho2017_tutorial.tar.gz": (
        "https://annayqho.github.io/TheCannon/_downloads/lamost_spectra.tar.gz",
        "6dd1476597dc9900268090ee4413e814e81bc9655fe3db9df27c0ce945558c9f",
    ),
    "lamost_labels_apogee_dr12.fits": (
        "https://annayqho.github.io/TheCannon/_downloads/lamost_labels.fits",
        "993f5197aa5e9a1367ccc4871bc32af96f7ef10a995d4a44060f44ae157221a7",
    ),
    "apogee_dr12_allStar-v603.fits": (
        "https://data.sdss3.org/sas/dr12/apogee/spectro/redux/r5/"
        "allStar-v603.fits",
        "c7010203583201dbae29268212d17614d29047dd6bce89d1983676dfdd3e8851",
    ),
}

DESI_COLUMNS = (
    "targetid,mean_fiber_ra,mean_fiber_dec,survey,program,healpix,"
    "tsnr2_gpbbright,tsnr2_gpbbackup,coadd_exptime"
)
DESI_DATA_LAB_QUERIES = {
    "main_bright": (
        f"select {DESI_COLUMNS} from desi_dr1.zpix "
        "where zcat_primary = true and spectype = 'STAR' "
        "and survey = 'main' and program = 'bright'"
    ),
    "other_primary": (
        f"select {DESI_COLUMNS} from desi_dr1.zpix "
        "where zcat_primary = true and spectype = 'STAR' "
        "and not (survey = 'main' and program = 'bright')"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_static(output_dir: Path, extract_lamost: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected) in STATIC_FILES.items():
        destination = output_dir / filename
        if destination.is_file() and sha256(destination) == expected:
            print(f"verified existing {destination}")
            continue
        partial = destination.with_suffix(destination.suffix + ".part")
        with urlopen(url) as response, partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        actual = sha256(partial)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {filename}: expected {expected}, got {actual}"
            )
        partial.replace(destination)
        print(f"downloaded and verified {destination}")
    if extract_lamost:
        archive = output_dir / "lamost_spectra_ho2017_tutorial.tar.gz"
        destination = output_dir / "lamost_ho2017_spectra"
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as stream:
            stream.extractall(destination, filter="data")
        print(f"extracted {archive} to {destination}")


def print_catalog_queries() -> None:
    print("Run these anonymous queries with NOIRLab Astro Data Lab, CSV output:")
    for name, query in DESI_DATA_LAB_QUERIES.items():
        print(f"\n{name}:\n{query}")


def _record_dict(record) -> dict[str, object]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "_asdict"):
        return dict(record._asdict())
    return {name: getattr(record, name) for name in record.keys()}


def retrieve_sparcl(
    selected_csv: Path,
    output_dir: Path,
    chunk_size: int,
) -> None:
    try:
        from sparcl.client import SparclClient
    except ImportError as exc:
        raise ImportError(
            "SPARCL acquisition requires sparclclient; install the data extra"
        ) from exc

    selected = pd.read_csv(selected_csv)
    targetids = selected["desi_targetid"].astype("int64").tolist()
    if len(targetids) != len(set(targetids)):
        raise ValueError("selected DESI target IDs must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    client = SparclClient(announcement=False)
    fields = [
        "sparcl_id",
        "targetid",
        "data_release",
        "ra",
        "dec",
        "redshift",
        "redshift_warning",
        "specprimary",
    ]
    found = client.find(
        outfields=fields,
        constraints={"targetid": targetids},
        limit=len(targetids) + 10,
        units=False,
    )
    found_records = [_record_dict(record) for record in found.records]
    found_by_target = {int(record["targetid"]): record for record in found_records}
    missing = sorted(set(targetids) - set(found_by_target))
    if missing:
        raise ValueError(f"SPARCL did not find {len(missing)} selected targets")
    response_path = output_dir / "sparcl_find_target_rv_response.json"
    response_path.write_text(
        json.dumps(
            [
                {
                    "status": {
                        "success": True,
                        "requested": len(targetids),
                        "found": len(found_records),
                        "dataset": "DESI-DR1",
                    }
                },
                *found_records,
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    ordered_ids = [found_by_target[targetid]["sparcl_id"] for targetid in targetids]
    for start in range(0, len(ordered_ids), chunk_size):
        chunk = ordered_ids[start : start + chunk_size]
        retrieved = client.retrieve(
            chunk,
            include=["targetid", "sparcl_id", "specid", "flux", "ivar", "mask"],
            dataset_list=["DESI-DR1"],
            limit=len(chunk),
            units=False,
        )
        records = [_record_dict(record) for record in retrieved.records]
        payload = [
            {
                "status": {
                    "success": len(records) == len(chunk),
                    "info": [f"retrieved {len(records)} DESI-DR1 records"],
                }
            },
            *records,
        ]
        path = output_dir / f"spectra_{start // chunk_size:02d}.pkl"
        with path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved {len(records)} spectra to {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static", help="download version-pinned files")
    static.add_argument("--output-dir", type=Path, required=True)
    static.add_argument("--extract-lamost", action="store_true")
    subparsers.add_parser("catalog-queries", help="print frozen Data Lab SQL")
    sparcl = subparsers.add_parser("sparcl", help="retrieve frozen DESI targets")
    sparcl.add_argument("--selected-csv", type=Path, required=True)
    sparcl.add_argument("--output-dir", type=Path, required=True)
    sparcl.add_argument("--chunk-size", type=int, default=400)
    args = parser.parse_args()
    if args.command == "static":
        download_static(args.output_dir, args.extract_lamost)
    elif args.command == "catalog-queries":
        print_catalog_queries()
    else:
        if args.chunk_size < 1 or args.chunk_size > 500:
            raise ValueError("chunk-size must be between 1 and 500")
        retrieve_sparcl(args.selected_csv, args.output_dir, args.chunk_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
