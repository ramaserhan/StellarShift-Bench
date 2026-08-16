from pathlib import Path
import zipfile

import pytest

from scripts.build_release import (
    archive_member_matches,
    deterministic_zip,
    parse_manifest,
    sha256_file,
    verify_manifest,
)


def test_deterministic_zip_ignores_source_mtime(tmp_path: Path) -> None:
    source = tmp_path / "result.txt"
    source.write_text("verified result\n", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    members = {"stellar-benchmark/result.txt": source}
    deterministic_zip(first, members)
    source.touch()
    deterministic_zip(second, members)
    assert first.read_bytes() == second.read_bytes()


def test_parse_manifest_rejects_unsafe_filename() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        parse_manifest("0" * 64 + "  ../notebook.ipynb\n")


def test_verify_manifest_detects_post_hash_edit(tmp_path: Path) -> None:
    notebook = tmp_path / "results.ipynb"
    notebook.write_text("before\n", encoding="utf-8")
    manifest = f"{sha256_file(notebook)}  {notebook.name}\n"
    notebook.write_text("after\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_manifest(manifest, tmp_path)


def test_archive_member_identity_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "results.ipynb"
    source.write_text("standalone\n", encoding="utf-8")
    archive_path = tmp_path / "evidence.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("stellar-benchmark/results.ipynb", "different\n")
    with pytest.raises(RuntimeError, match="archive member mismatch"):
        archive_member_matches(
            archive_path, "stellar-benchmark/results.ipynb", source
        )
