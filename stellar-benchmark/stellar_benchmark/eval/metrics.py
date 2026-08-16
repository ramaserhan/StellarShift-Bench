"""Metrics for evaluating stellar-parameter predictions.

These follow the conventions used in the cross-survey ML literature (e.g. bias
and scatter reported by Ho et al. 2017, R^2 reported by Zhao et al. 2026) plus
a calibration metric that most published studies skip.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed residual (prediction - truth). Reported in the same units
    as the target (e.g. K for Teff, dex for [Fe/H])."""
    y_true, y_pred = _validate_pair(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute residual in the target's physical units."""

    y_true, y_pred = _validate_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared residual in the target's physical units."""

    y_true, y_pred = _validate_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def scatter(y_true: np.ndarray, y_pred: np.ndarray, robust: bool = True) -> float:
    """Spread of the residuals. Robust=True uses 1.4826 * MAD, which is
    standard in the stellar-parameter literature because it is less sensitive
    to outlier stars than a plain standard deviation."""
    y_true, y_pred = _validate_pair(y_true, y_pred)
    resid = y_pred - y_true
    if robust:
        mad = np.median(np.abs(resid - np.median(resid)))
        return float(1.4826 * mad)
    return float(np.std(resid))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_pair(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def calibration_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
    z: float = 1.0,
) -> float:
    """Fraction of true values falling within +/- z predicted standard
    deviations of the prediction. For a well-calibrated model with Gaussian
    errors and z=1, this should be close to 0.68.

    This directly targets a gap the paper flags: model calibration is rarely
    reported alongside point-estimate accuracy.
    """
    y_true, y_pred = _validate_pair(y_true, y_pred)
    y_pred_std = np.asarray(y_pred_std, dtype=float).reshape(-1)
    if len(y_pred_std) != len(y_true):
        raise ValueError("y_pred_std must have the same length as y_true")
    if not np.isfinite(y_pred_std).all() or (y_pred_std < 0).any():
        raise ValueError("y_pred_std must contain finite, non-negative values")
    if z <= 0:
        raise ValueError("z must be positive")
    lower = y_pred - z * y_pred_std
    upper = y_pred + z * y_pred_std
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray | None = None,
) -> dict:
    """Compute the standard set of metrics for one target parameter."""
    out = {
        "n": int(len(y_true)),
        "bias": bias(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "scatter": scatter(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }
    if y_pred_std is not None:
        out["coverage_1sigma"] = calibration_coverage(y_true, y_pred, y_pred_std, z=1.0)
        out["coverage_2sigma"] = calibration_coverage(y_true, y_pred, y_pred_std, z=2.0)
    return out


def paired_significance(resid_a: np.ndarray, resid_b: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value comparing two sets of residuals on the
    SAME stars (e.g. baseline vs. post-adaptation). Use this before claiming
    an adaptation method 'improved' performance -- a lower scatter alone can
    be noise on a small held-out set.
    """
    resid_a, resid_b = _validate_pair(resid_a, resid_b)
    absolute_difference = np.abs(resid_a) - np.abs(resid_b)
    if np.allclose(absolute_difference, 0):
        return 1.0
    _, p = stats.wilcoxon(np.abs(resid_a), np.abs(resid_b))
    return float(p)


def _validate_pair(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first, dtype=float).reshape(-1)
    second = np.asarray(second, dtype=float).reshape(-1)
    if len(first) == 0:
        raise ValueError("metric inputs must not be empty")
    if len(first) != len(second):
        raise ValueError("metric inputs must have the same length")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("metric inputs must contain only finite values")
    return first, second
