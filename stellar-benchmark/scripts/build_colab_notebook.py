"""Regenerate the compact Colab walkthrough shipped with the repository."""

from __future__ import annotations

import json
from pathlib import Path


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
# StellarShift-Bench: DESI controlled S/N case study

This notebook runs the packaged v0.3 workflow rather than duplicating the
scientific logic in ad-hoc cells. It downloads approximately 446 MB of public
DESI DR1 FITS files, extracts the R arm, runs 24 tests, and produces the clean,
multi-seed shift, noise-augmentation, and labeled-target-retraining results.

The experiment uses RVSpecFit outputs as **reference labels**, not independent
ground truth. The shift is controlled Gaussian S/N degradation on real spectra,
not a natural cross-survey result.
"""
    ),
    markdown(
        """
## 1. Upload and unpack the project

Upload `stellar-benchmark-v0.3.0.zip` when prompted.
"""
    ),
    code(
        """
from google.colab import files
from pathlib import Path
import zipfile

uploaded = files.upload()
project_zip = next(name for name in uploaded if name.endswith(".zip"))

with zipfile.ZipFile(project_zip) as archive:
    archive.extractall("/content")

project_dir = Path("/content/stellar-benchmark")
assert project_dir.is_dir(), "Expected /content/stellar-benchmark in the ZIP"
%cd /content/stellar-benchmark
"""
    ),
    markdown("## 2. Install and validate the software"),
    code(
        """
!python -m pip install -q -e ".[dev,desi]"
!python -m pytest -q
!stellar-benchmark run --config configs/synthetic_demo.yaml
"""
    ),
    markdown(
        """
## 3. Download the public DESI subset

The ten files contain roughly 1,000 stellar spectra and occupy approximately
446 MB. Existing completed files are reused.
"""
    ),
    code(
        """
!python examples/02_download_desi_subset.py --output-dir /content/desi_files
"""
    ),
    markdown("## 4. Extract aligned, quality-flagged R-arm arrays"),
    code(
        """
!stellar-benchmark desi-extract \
  --input-glob "/content/desi_files/*.fits" \
  --output data/processed/desi_r_raw.npz
"""
    ),
    markdown(
        """
## 5. Run the full case study

This fits preprocessing on source training only, trains source-only and
noise-augmented models, evaluates five shift levels across ten deterministic
noise realizations, and runs the disjoint 90-star target-retraining baseline.
"""
    ),
    code(
        """
!stellar-benchmark desi-run --config configs/desi_snr.yaml
"""
    ),
    markdown("## 6. Inspect the auditable summary and figures"),
    code(
        """
from IPython.display import Image, display

print(Path("results/desi_case_study/summary.txt").read_text())
display(Image(filename="results/desi_case_study/snr_robustness_curve.png"))
display(Image(filename="results/desi_case_study/mitigation_comparison.png"))
"""
    ),
    markdown("## 7. Download the compact results"),
    code(
        """
import shutil

archive = shutil.make_archive(
    "/content/stellarshift-desi-results",
    "zip",
    root_dir="results",
    base_dir="desi_case_study"
)
files.download(archive)
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "StellarShift_DESI_Colab.ipynb", "provenance": []},
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

destination = Path(__file__).resolve().parents[1] / "examples" / "StellarShift_DESI_Colab.ipynb"
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(destination)
