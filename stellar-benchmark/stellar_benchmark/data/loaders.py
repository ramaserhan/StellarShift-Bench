"""Survey data loaders.

The network helper returns a version-declared public table without guessing a
release-specific label mapping. Spectral transfer uses the stricter NPZ
contract in ``cross_survey.py``.

Design note: every loader returns a pandas DataFrame with a common minimal
schema (teff, logg, feh, snr, source_id, ...) so downstream code never has to
know which survey produced a row. This shared schema is what actually makes
cross-survey comparison possible -- most of the "domain shift" pain in this
field comes from surveys disagreeing on column names, units, and null
conventions before a single model is ever trained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COMMON_SCHEMA = ["source_id", "teff", "logg", "feh", "snr", "stellar_type"]


def load_benchmark_csv(
    path: str | Path,
    feature_columns: list[str],
    target_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a local benchmark table and validate its shared schema.

    This loader supports the committed miniature dataset and preprocessed real
    survey tables. Raw FITS spectrum ingestion will be implemented separately
    for each data release so that release-specific choices stay explicit.
    """

    target_columns = target_columns or ["teff", "logg", "feh"]
    required = ["source_id", "snr", *feature_columns, *target_columns]
    if len(required) != len(set(required)):
        raise ValueError("feature and target column lists must not overlap or contain duplicates")

    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"benchmark CSV not found: {data_path}")
    df = pd.read_csv(data_path)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"benchmark CSV is missing required columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("benchmark CSV contains missing values in required columns")

    numeric_columns = ["snr", *feature_columns, *target_columns]
    for column in numeric_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy()).all():
            raise ValueError(f"column {column!r} must contain only finite numeric values")
        df[column] = converted
    return df


def load_vizier_table(catalog_id: str, max_rows: int = 20000) -> pd.DataFrame:
    """Download one explicit VizieR table without inventing column semantics.

    Callers must record ``catalog_id`` and perform an explicit, tested mapping
    into either the scalar benchmark schema or the spectral NPZ contract.  This
    avoids silently treating a published label catalog as a spectral dataset.
    """

    if not catalog_id.strip():
        raise ValueError("catalog_id must not be empty")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    try:
        from astroquery.vizier import Vizier
    except ImportError as exc:  # pragma: no cover - optional network dependency
        raise ImportError(
            "VizieR access requires the data extra: "
            "python -m pip install -e '.[data]'"
        ) from exc
    client = Vizier(columns=["*"], row_limit=max_rows)
    tables = client.get_catalogs(catalog_id)
    if len(tables) == 0:
        raise ValueError(f"VizieR returned no tables for {catalog_id!r}")
    frame = tables[0].to_pandas()
    if len(frame) == 0:
        raise ValueError(f"VizieR table {catalog_id!r} is empty")
    frame.attrs["vizier_catalog_id"] = catalog_id
    return frame


def load_crossmatch(df_a: pd.DataFrame, df_b: pd.DataFrame, on: str = "source_id") -> pd.DataFrame:
    """Join two survey dataframes on a shared identifier (e.g. Gaia source_id
    or 2MASS ID) to find stars observed by both surveys -- this crossmatch is
    what gives you ground truth for measuring cross-survey label agreement,
    the same approach used by Ho et al. (2017) with 9,952 common stars.
    """
    return df_a.merge(df_b, on=on, suffixes=("_a", "_b"))
