"""Build the v1.2.3 Google Colab launcher for review and full reproduction."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


cells = [
    markdown(
        """
# StellarShift-Bench v1.2.3 — Google Colab launcher

This notebook supports two workflows:

1. **Review and validate the frozen release** — upload the v1.2.3 source ZIP,
   install it, run all 51 tests, and inspect the already-verified cross-survey
   evidence. This takes only a few minutes and does not download raw spectra.
2. **Recompute the real LAMOST DR2 → DESI DR1 experiment** — reacquire the
   public survey inputs, rebuild both contracts, and rerun the prespecified quality-gated
   harness. Budget roughly 2–3 GB of runtime storage and a longer CPU session.

Use a **high-RAM CPU runtime** if available. A GPU is not required: the primary
models are tree ensembles and the bottlenecks are downloads, preprocessing,
bootstrap analysis, and CPU fitting.

The older `StellarShift_DESI_Colab.ipynb` in the repository is the historical
v0.3 DESI-only controlled-noise study. It is not the v1.2.3 cross-survey run.
"""
    ),
    markdown(
        """
## 1. Upload the v1.2.3 source release

Download `stellar-benchmark-v1.2.3.zip` from the release and upload it when
prompted. You may upload `SHA256SUMS-v1.2.3.txt` at the same time; if present,
the cell verifies the ZIP before extraction.
"""
    ),
    code(
        """
from google.colab import files as colab_files
from pathlib import Path
import hashlib
import zipfile

uploaded = colab_files.upload()
zip_names = [name for name in uploaded if name == "stellar-benchmark-v1.2.3.zip"]
assert len(zip_names) == 1, "Upload exactly stellar-benchmark-v1.2.3.zip"
project_zip = Path(zip_names[0])

manifest_name = "SHA256SUMS-v1.2.3.txt"
if manifest_name in uploaded:
    expected = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in uploaded[manifest_name].decode().splitlines()
        if "  " in line
    }[project_zip.name]
    observed = hashlib.sha256(project_zip.read_bytes()).hexdigest()
    assert observed == expected, f"ZIP checksum mismatch: {observed} != {expected}"
    print("Source ZIP checksum verified:", observed)

with zipfile.ZipFile(project_zip) as archive:
    assert archive.testzip() is None, "ZIP CRC check failed"
    archive.extractall("/content")

project_dir = Path("/content/stellar-benchmark")
assert project_dir.is_dir(), "Expected /content/stellar-benchmark inside the ZIP"
%cd /content/stellar-benchmark
"""
    ),
    markdown("## 2. Install the benchmark and run its complete test suite"),
    code(
        """
%pip install -q -e ".[dev,data]"

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)

# SPARCL 1.3.0 declares pandas<2.2 on Python 3.12, while current Colab requires
# pandas==2.2.2. Install it in a separate environment so it cannot downgrade
# or destabilize the notebook runtime.
sparcl_environment = Path("/content/stellarshift-sparcl-env")
SPARCL_PYTHON = sparcl_environment / "bin" / "python"
if not SPARCL_PYTHON.is_file():
    subprocess.run(
        [sys.executable, "-m", "venv", str(sparcl_environment)], check=True
    )
    subprocess.run(
        [
            str(SPARCL_PYTHON),
            "-m",
            "pip",
            "install",
            "-q",
            "sparclclient==1.3.0",
        ],
        check=True,
    )
subprocess.run(
    [str(SPARCL_PYTHON), "-c", "from sparcl.client import SparclClient"],
    check=True,
)
print("SPARCL isolated environment ready:", SPARCL_PYTHON)
"""
    ),
    markdown(
        """
## 3A. Fast path: inspect the verified cross-survey result

The source release already contains the prespecified quality-gated result tables,
per-star predictions, figures, provenance, and report. The following cell reads
those frozen artifacts; it does not refit a model.
"""
    ),
    code(
        """
from IPython.display import Image, display
from pathlib import Path

result_dir = Path("results/lamost_dr2_to_desi_dr1_apogee_dr12")
print((result_dir / "summary.txt").read_text())
display(Image(filename=str(result_dir / "cross_survey_transfer.png")))
display(Image(filename=str(result_dir / "calibration_cross_survey.png")))
display(Image(filename=str(result_dir / "label_budget.png")))
"""
    ),
    markdown(
        """
The standalone `StellarShift_v1.2.3_Instant_Results.ipynb` is even quicker for
review: upload that notebook directly through **File → Upload notebook** in
Colab. Its verified tables and figures are embedded before any cell is run.

Stop here if you only want to inspect, present, or validate the release.
Continue below only for a clean public-data reconstruction.
"""
    ),
    markdown(
        """
# Full public-data reconstruction

The release deliberately excludes raw spectra and processed NPZ arrays. These
cells retrieve the frozen 1,576-star DESI selection rather than repeating the
multi-million-row catalog crossmatch. Colab storage is temporary, so copy the
final results to Drive or download them before the runtime disconnects.
"""
    ),
    markdown("## 4. Check runtime storage"),
    code(
        """
import shutil

total, used, free = shutil.disk_usage("/content")
print(f"Free Colab storage: {free / 1024**3:.1f} GB")
assert free >= 6 * 1024**3, "Use a fresh runtime with at least 6 GB free"
"""
    ),
    markdown("## 5. Download and verify the LAMOST/APOGEE static inputs"),
    code(
        """
subprocess.run(
    [
        sys.executable,
        "scripts/acquire_real_cross_survey.py",
        "static",
        "--output-dir",
        "data/raw/catalogs",
        "--extract-lamost",
    ],
    check=True,
)
"""
    ),
    markdown(
        """
## 6. Retrieve the frozen DESI DR1 target spectra

SPARCL is public and anonymous. The request downloads the exact target IDs in
the archived selection in four chunks. If the public service is temporarily
rate-limited, wait and rerun this cell; completed files remain in the runtime.
"""
    ),
    code(
        """
subprocess.run(
    [
        str(SPARCL_PYTHON),
        "scripts/acquire_real_cross_survey.py",
        "sparcl",
        "--selected-csv",
        "data/acquisition_logs/apogee_dr12_desi_dr1_target_selected.csv",
        "--output-dir",
        "data/acquisition_logs/sparcl_chunks",
        "--chunk-size",
        "400",
    ],
    check=True,
)
"""
    ),
    markdown("## 7. Build the leakage-safe source and target contracts"),
    code(
        """
lamost_candidates = [
    Path("data/raw/catalogs/lamost_ho2017_spectra/spectra"),
    Path("data/raw/lamost_ho2017_spectra/spectra"),
]
lamost_spectra = next((path for path in lamost_candidates if path.is_dir()), None)
if lamost_spectra is None:
    raise FileNotFoundError(
        "LAMOST spectra were not extracted. Rerun section 5 before continuing."
    )

required_files = {
    "LAMOST labels": Path("data/raw/catalogs/lamost_labels_apogee_dr12.fits"),
    "APOGEE allStar": Path("data/raw/catalogs/apogee_dr12_allStar-v603.fits"),
    "DESI target selection": Path(
        "data/acquisition_logs/apogee_dr12_desi_dr1_target_selected.csv"
    ),
    "DESI redshifts": Path(
        "data/acquisition_logs/sparcl_find_target_rv_response.json"
    ),
}
missing = [name for name, path in required_files.items() if not path.is_file()]
if missing:
    raise FileNotFoundError(
        "Missing acquisition inputs: " + ", ".join(missing) +
        ". Rerun sections 5 and 6."
    )

sparcl_chunks = sorted(
    Path("data/acquisition_logs/sparcl_chunks").glob("spectra_*.pkl")
)
if len(sparcl_chunks) != 4:
    raise FileNotFoundError(
        f"Expected 4 SPARCL spectrum chunks, found {len(sparcl_chunks)}. "
        "Rerun section 6 and inspect its first error."
    )

print("LAMOST spectra:", lamost_spectra)
print("SPARCL chunks:", [path.name for path in sparcl_chunks])

subprocess.run(
    [
        sys.executable,
        "scripts/build_apogee_dr12_contracts.py",
        "--lamost-labels",
        str(required_files["LAMOST labels"]),
        "--lamost-spectra",
        str(lamost_spectra),
        "--apogee-allstar",
        str(required_files["APOGEE allStar"]),
        "--desi-selection",
        str(required_files["DESI target selection"]),
        "--desi-pickle-glob",
        "data/acquisition_logs/sparcl_chunks/spectra_*.pkl",
        "--desi-redshifts",
        str(required_files["DESI redshifts"]),
        "--output-dir",
        "data/processed",
    ],
    check=True,
)

expected_contracts = [
    Path("data/processed/lamost_apogee_dr12.npz"),
    Path("data/processed/desi_apogee_dr12.npz"),
]
for contract in expected_contracts:
    if not contract.is_file():
        raise RuntimeError(f"Contract builder completed without creating {contract}")
    print("created", contract, f"({contract.stat().st_size / 1024**2:.1f} MB)")
"""
    ),
    markdown("## 8. Verify reconstructed input hashes against the frozen manifest"),
    code(
        """
import json

manifest = json.loads(Path("data/acquisition_logs/acquisition_manifest.json").read_text())
for entry in manifest["processed_contracts"]:
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} was not created. Rerun section 7 and stop on its first error."
        )
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed == entry["sha256"], (
        f"Processed-contract mismatch for {path}: {observed} != {entry['sha256']}"
    )
    print("verified", path, observed)
"""
    ),
    markdown("## 9. Run the prespecified quality-gated LAMOST → DESI benchmark"),
    code(
        """
subprocess.run(
    [
        sys.executable,
        "-m",
        "stellar_benchmark",
        "cross-survey-run",
        "--config",
        "configs/lamost_to_desi.yaml",
    ],
    check=True,
)
"""
    ),
    markdown("## 10. Inspect the recomputed result and gate"),
    code(
        """
import pandas as pd

result_dir = Path("results/lamost_dr2_to_desi_dr1_apogee_dr12")
gate = json.loads((result_dir / "publication_gate.json").read_text())
assert gate["passed"], gate
print((result_dir / "summary.txt").read_text())
display(pd.read_csv(result_dir / "domain_shift_summary.csv"))
display(pd.read_csv(result_dir / "calibration_cross_survey.csv"))
display(Image(filename=str(result_dir / "cross_survey_transfer.png")))
"""
    ),
    markdown("## 11. Download the recomputed evidence"),
    code(
        """
import shutil

archive = shutil.make_archive(
    "/content/stellarshift-v1.2.3-colab-results",
    "zip",
    root_dir="results",
    base_dir="lamost_dr2_to_desi_dr1_apogee_dr12",
)
colab_files.download(archive)
"""
    ),
    markdown(
        """
### Optional: persist to Google Drive

Run this instead of—or before—the download cell if you want the results to
survive a disconnected Colab runtime.
"""
    ),
    code(
        """
from google.colab import drive
drive.mount("/content/drive")
shutil.copy2(
    "/content/stellarshift-v1.2.3-colab-results.zip",
    "/content/drive/MyDrive/stellarshift-v1.2.3-colab-results.zip",
)
print("Saved to Google Drive")
"""
    ),
    markdown(
        """
## Reproducibility note

The frozen release hashes prove the identity of the published artifacts.
Recomputing in a future Colab image can introduce very small low-level numeric
differences when NumPy, SciPy, or scikit-learn versions change. Compare the
reported metrics and quality-gate counts, not regenerated PDF/CSV hashes,
unless you reproduce the original environment exactly.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "StellarShift_Bench_v1.2.3_Colab.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = ROOT / "examples" / "StellarShift_Bench_v1.2.3_Colab.ipynb"
destination.write_text(
    json.dumps(notebook, indent=1, sort_keys=True) + "\n", encoding="utf-8"
)
print(destination)
