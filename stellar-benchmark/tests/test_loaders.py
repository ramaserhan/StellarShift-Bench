from pathlib import Path

from stellar_benchmark.data.loaders import load_benchmark_csv, load_vizier_table
from stellar_benchmark.demo import FEATURE_COLUMNS


def test_committed_miniature_dataset_matches_shared_schema():
    path = Path(__file__).parents[1] / "examples" / "data" / "mini_stellar_spectra.csv"
    frame = load_benchmark_csv(path, FEATURE_COLUMNS)
    assert len(frame) == 24
    assert frame["source_id"].is_unique
    assert set(["teff", "logg", "feh", "snr"]).issubset(frame.columns)


def test_vizier_loader_validates_before_network_access():
    for catalog_id, max_rows in (("", 10), ("catalog", 0)):
        try:
            load_vizier_table(catalog_id, max_rows=max_rows)
            assert False, "expected ValueError"
        except ValueError:
            pass
