"""Build and verify every public StellarShift-Bench v1.2.3 artifact.

This is the only supported release path. Derived files are regenerated twice,
both archives use fixed ordering and metadata, and the checksum manifest is
published only after standalone/archive byte identity has been verified.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = Path(
    os.environ.get("STELLARSHIFT_DELIVERABLES", ROOT / "dist")
).resolve()
VERSION = "1.2.3"
ARCHIVE_ROOT = "stellar-benchmark"
FIXED_ZIP_TIME = (2026, 8, 16, 0, 0, 0)

NOTEBOOK = ROOT / "examples" / "StellarShift_v1.2.3_Instant_Results.ipynb"
COLAB = ROOT / "examples" / "StellarShift_Bench_v1.2.3_Colab.ipynb"
REPORT = ROOT / "output" / "pdf" / "StellarShift_Bench_v1.2.3_Technical_Report.pdf"
BRIEF = ROOT / "output" / "pdf" / "StellarShift_Bench_v1.2.3_Portfolio_Brief.pdf"
FIGURES = (
    ROOT / "output" / "figures" / "real_vs_controlled_shift.png",
    ROOT / "output" / "figures" / "support_sensitivity.png",
    ROOT / "output" / "figures" / "adaptation_relative_effects.png",
)

SOURCE_ARCHIVE = DELIVERABLES / f"stellar-benchmark-v{VERSION}.zip"
EVIDENCE_ARCHIVE = DELIVERABLES / f"stellarshift-v{VERSION}-evidence-bundle.zip"
CHECKSUM_FILE = DELIVERABLES / f"SHA256SUMS-v{VERSION}.txt"
PUBLIC_NOTEBOOK = DELIVERABLES / NOTEBOOK.name
PUBLIC_COLAB = DELIVERABLES / COLAB.name
PUBLIC_REPORT = DELIVERABLES / REPORT.name
PUBLIC_BRIEF = DELIVERABLES / BRIEF.name

PUBLIC_ACQUISITION_FILES = {
    "acquisition_manifest.json",
    "apogee_dr12_desi_dr1_crossmatch.csv",
    "apogee_dr12_desi_dr1_selection_flow.csv",
    "apogee_dr12_desi_dr1_target_selected.csv",
    "sparcl_find_target_rv_payload.json",
    "sparcl_find_target_rv_response.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(text: str) -> dict[str, str]:
    """Parse a conventional two-space-separated SHA256SUMS manifest."""

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            digest, name = raw_line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"invalid manifest line {line_number}: {raw_line!r}") from error
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid SHA-256 on manifest line {line_number}")
        if not name or Path(name).name != name:
            raise ValueError(f"unsafe manifest filename on line {line_number}: {name!r}")
        if name in entries:
            raise ValueError(f"duplicate manifest filename: {name}")
        entries[name] = digest
    return entries


def verify_manifest(text: str, directory: Path) -> None:
    """Fail if any manifest entry is missing or differs from its final bytes."""

    for name, expected in parse_manifest(text).items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"checksum mismatch for {name}: {observed} != {expected}")


def deterministic_zip(destination: Path, members: Mapping[str, Path]) -> None:
    """Write a ZIP with stable names, ordering, timestamps, and permissions."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for archive_name in sorted(members):
            source = members[archive_name]
            info = zipfile.ZipInfo(archive_name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(info, source.read_bytes(), compresslevel=9)


def archive_member_matches(archive_path: Path, member_name: str, source: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"CRC failure in {archive_path.name}")
        archived = archive.read(member_name)
    current = source.read_bytes()
    if archived != current:
        raise RuntimeError(
            f"archive member mismatch: {archive_path.name}:{member_name}"
        )


def _is_public_source(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    parts = relative.parts
    if not path.is_file():
        return False
    if any(part in {".git", ".pytest_cache", "__pycache__", ".venv", "build", "dist", "tmp"} for part in parts):
        return False
    if path.suffix in {".pyc", ".pkl"}:
        return False
    if parts[:2] in {("data", "private"), ("data", "processed"), ("data", "raw")}:
        return False
    if relative == Path("data/desi_r_raw_private.npz"):
        return False
    if parts[:2] == ("data", "acquisition_logs"):
        return len(parts) == 3 and parts[2] in PUBLIC_ACQUISITION_FILES
    if parts[:2] == ("output", "pdf"):
        return path in {REPORT, BRIEF}
    if parts[:1] == ("examples",) and path.name.endswith("_Instant_Results.ipynb"):
        return path == NOTEBOOK
    if parts[:1] == ("examples",) and path.name.startswith("StellarShift_Bench_v") and path.name.endswith("_Colab.ipynb"):
        return path == COLAB
    return True


def source_members() -> dict[str, Path]:
    return {
        f"{ARCHIVE_ROOT}/{path.relative_to(ROOT).as_posix()}": path
        for path in ROOT.rglob("*")
        if _is_public_source(path)
    }


def evidence_files() -> Iterable[Path]:
    explicit = [
        ROOT / ".zenodo.json",
        ROOT / "CHANGELOG.md",
        ROOT / "CITATION.cff",
        ROOT / "LICENSE",
        ROOT / "README.md",
        ROOT / "configs" / "desi_reliability.yaml",
        ROOT / "configs" / "lamost_to_desi.yaml",
        ROOT / "data" / "README.md",
        ROOT / "docs" / "CRITIQUE_RESPONSE.md",
        ROOT / "docs" / "CROSS_SURVEY_RUNBOOK.md",
        ROOT / "docs" / "EVIDENCE_BUNDLE.md",
        ROOT / "docs" / "EXECUTION_STATUS.md",
        ROOT / "docs" / "METHODOLOGY.md",
        ROOT / "docs" / "RELEASE_CHECKLIST.md",
        ROOT / "scripts" / "build_apogee_desi_crossmatch.py",
        NOTEBOOK,
        COLAB,
        REPORT,
        BRIEF,
        *FIGURES,
    ]
    for name in sorted(PUBLIC_ACQUISITION_FILES):
        explicit.append(ROOT / "data" / "acquisition_logs" / name)
    for directory in (
        ROOT / "results" / "desi_reliability_v1",
        ROOT / "results" / "lamost_dr2_to_desi_dr1_apogee_dr12",
    ):
        explicit.extend(path for path in directory.rglob("*") if path.is_file())
    return explicit


def evidence_members() -> dict[str, Path]:
    return {
        f"{ARCHIVE_ROOT}/{path.relative_to(ROOT).as_posix()}": path
        for path in evidence_files()
    }


def _run_builder(script_name: str, environment: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name)],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def regenerate_and_prove_stability() -> None:
    """Generate all derived artifacts twice and require byte-identical output."""

    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = "1786838400"
    environment.setdefault("PYTHONHASHSEED", "0")
    with tempfile.TemporaryDirectory(prefix="stellarshift-mpl-") as mpl_dir:
        environment["MPLCONFIGDIR"] = mpl_dir
        for script in (
            "build_v1_report.py",
            "build_instant_results_notebook.py",
            "build_v121_colab_notebook.py",
        ):
            _run_builder(script, environment)
        derived = (NOTEBOOK, COLAB, REPORT, BRIEF, *FIGURES)
        first = {path: sha256_file(path) for path in derived}
        for script in (
            "build_v1_report.py",
            "build_instant_results_notebook.py",
            "build_v121_colab_notebook.py",
        ):
            _run_builder(script, environment)
        second = {path: sha256_file(path) for path in derived}
    if first != second:
        unstable = [path.relative_to(ROOT).as_posix() for path in derived if first[path] != second[path]]
        raise RuntimeError(f"non-deterministic derived artifacts: {unstable}")


def _manifest_text(paths: Iterable[Path]) -> str:
    return "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths)


def build_release() -> None:
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    regenerate_and_prove_stability()
    with tempfile.TemporaryDirectory(prefix="stellarshift-release-", dir=DELIVERABLES) as stage:
        stage_dir = Path(stage)
        staged_source = stage_dir / SOURCE_ARCHIVE.name
        staged_evidence = stage_dir / EVIDENCE_ARCHIVE.name
        staged_checksums = stage_dir / CHECKSUM_FILE.name
        staged_notebook = stage_dir / PUBLIC_NOTEBOOK.name
        staged_colab = stage_dir / PUBLIC_COLAB.name
        staged_report = stage_dir / PUBLIC_REPORT.name
        staged_brief = stage_dir / PUBLIC_BRIEF.name
        deterministic_zip(staged_source, source_members())
        deterministic_zip(staged_evidence, evidence_members())

        staged_notebook.write_bytes(NOTEBOOK.read_bytes())
        staged_colab.write_bytes(COLAB.read_bytes())
        staged_report.write_bytes(REPORT.read_bytes())
        staged_brief.write_bytes(BRIEF.read_bytes())

        tracked = (
            staged_source,
            staged_evidence,
            staged_report,
            staged_brief,
            staged_notebook,
            staged_colab,
        )
        staged_checksums.write_text(_manifest_text(tracked), encoding="utf-8")

        notebook_member = f"{ARCHIVE_ROOT}/{NOTEBOOK.relative_to(ROOT).as_posix()}"
        report_member = f"{ARCHIVE_ROOT}/{REPORT.relative_to(ROOT).as_posix()}"
        brief_member = f"{ARCHIVE_ROOT}/{BRIEF.relative_to(ROOT).as_posix()}"
        for archive in (staged_source, staged_evidence):
            archive_member_matches(archive, notebook_member, NOTEBOOK)
            archive_member_matches(archive, report_member, REPORT)
            archive_member_matches(archive, brief_member, BRIEF)

        os.replace(staged_source, SOURCE_ARCHIVE)
        os.replace(staged_evidence, EVIDENCE_ARCHIVE)
        os.replace(staged_report, PUBLIC_REPORT)
        os.replace(staged_brief, PUBLIC_BRIEF)
        os.replace(staged_notebook, PUBLIC_NOTEBOOK)
        os.replace(staged_colab, PUBLIC_COLAB)
        os.replace(staged_checksums, CHECKSUM_FILE)

    verify_manifest(CHECKSUM_FILE.read_text(encoding="utf-8"), DELIVERABLES)
    for path in (
        SOURCE_ARCHIVE,
        EVIDENCE_ARCHIVE,
        PUBLIC_REPORT,
        PUBLIC_BRIEF,
        PUBLIC_NOTEBOOK,
        PUBLIC_COLAB,
        CHECKSUM_FILE,
    ):
        print(f"{sha256_file(path)}  {path}")


if __name__ == "__main__":
    build_release()
