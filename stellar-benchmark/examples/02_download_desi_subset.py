"""Download the ten-file public DESI subset used by the v0.3 case study."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


BASE_URL = (
    "https://data.desi.lbl.gov/public/dr1/vac/dr1/stellar-reddening/"
    "v1.0/spectra/main/bright"
)


def download(healpix: int, output_dir: Path) -> Path:
    url = f"{BASE_URL}/{healpix // 100}/{healpix}/rvspecfit-main-bright-{healpix}.fits"
    destination = output_dir / f"rvspecfit-main-bright-{healpix}.fits"
    temporary = destination.with_suffix(".fits.part")
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/desi_fits"))
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=9)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.first < 0 or args.last < args.first:
        parser.error("expected 0 <= first <= last")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    healpix_values = list(range(args.first, args.last + 1))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        paths = list(
            executor.map(
                lambda value: download(value, args.output_dir), healpix_values
            )
        )
    for path in paths:
        print(path, f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
