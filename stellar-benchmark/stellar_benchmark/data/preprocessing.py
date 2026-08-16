"""Shared preprocessing utilities.

Applying the SAME preprocessing function to every survey (rather than each
survey's own pipeline defaults) is one of the simplest but most overlooked
ways to reduce spurious "domain shift" that is really just inconsistent
preprocessing -- the literature review this project is based on flags this
explicitly (section 4.3: preprocessing is part of the effective model, not a
neutral preliminary step).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def apply_snr_cut(df: pd.DataFrame, snr_min: float, snr_col: str = "snr") -> pd.DataFrame:
    if snr_col not in df.columns:
        raise KeyError(f"missing S/N column: {snr_col!r}")
    if snr_min < 0:
        raise ValueError("snr_min must be non-negative")
    return df[df[snr_col] >= snr_min].reset_index(drop=True)


def add_snr_bins(df: pd.DataFrame, snr_col: str = "snr", bins=(0, 50, 100, 200, np.inf)) -> pd.DataFrame:
    if snr_col not in df.columns:
        raise KeyError(f"missing S/N column: {snr_col!r}")
    if len(bins) < 2 or any(a >= b for a, b in zip(bins, bins[1:])):
        raise ValueError("bins must contain at least two strictly increasing edges")
    labels = [f"{int(bins[i])}-{int(bins[i+1]) if bins[i+1] != np.inf else 'inf'}" for i in range(len(bins) - 1)]
    df = df.copy()
    df["snr_bin"] = pd.cut(df[snr_col], bins=bins, labels=labels)
    return df


def add_feh_bins(df: pd.DataFrame, feh_col: str = "feh", bins=(-3, -1, -0.5, 0, 0.5)) -> pd.DataFrame:
    if feh_col not in df.columns:
        raise KeyError(f"missing metallicity column: {feh_col!r}")
    if len(bins) < 2 or any(a >= b for a, b in zip(bins, bins[1:])):
        raise ValueError("bins must contain at least two strictly increasing edges")
    labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]
    df = df.copy()
    df["feh_bin"] = pd.cut(df[feh_col], bins=bins, labels=labels)
    return df


def train_test_split_by_column(
    df: pd.DataFrame, test_fraction: float = 0.2, random_state: int = 42
):
    """Deprecated row-level random split.

    Use :func:`train_test_split_by_group` for spectra because repeated
    observations of the same star must never appear in both partitions.
    """
    warnings.warn(
        "train_test_split_by_column performs a row-level split; use "
        "train_test_split_by_group for leakage-safe stellar data",
        DeprecationWarning,
        stacklevel=2,
    )
    _validate_fraction(test_fraction)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_fraction)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def train_test_split_by_group(
    df: pd.DataFrame,
    group_col: str = "source_id",
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically split complete stars rather than individual rows.

    Every row sharing ``group_col`` is assigned to the same partition. This
    prevents repeated spectra of one star from leaking into both training and
    evaluation sets.
    """

    _validate_fraction(test_fraction)
    if group_col not in df.columns:
        raise KeyError(f"missing group column: {group_col!r}")
    if df[group_col].isna().any():
        raise ValueError(f"group column {group_col!r} contains missing values")

    groups = pd.unique(df[group_col])
    if len(groups) < 2:
        raise ValueError("at least two distinct groups are required")

    rng = np.random.default_rng(random_state)
    shuffled_groups = rng.permutation(groups)
    n_test = min(max(1, int(round(len(groups) * test_fraction))), len(groups) - 1)
    test_groups = set(shuffled_groups[:n_test].tolist())
    is_test = df[group_col].isin(test_groups)

    train = df.loc[~is_test].reset_index(drop=True)
    test = df.loc[is_test].reset_index(drop=True)
    return train, test


def _validate_fraction(value: float) -> None:
    if not 0 < value < 1:
        raise ValueError("test_fraction must be between 0 and 1")
