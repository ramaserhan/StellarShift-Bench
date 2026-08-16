"""Reliability diagnostics for stellar models under domain shift.

The functions in this module deliberately operate on predictions rather than
on a particular estimator.  This keeps interval calibration, bootstrap
uncertainty, OOD scoring, and selective prediction comparable across model
families and survey pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .metrics import summary_metrics


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    """Return a finite-sample conformal quantile using the higher order statistic."""

    values = _finite_vector(values, "values")
    if not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    rank = int(np.ceil(probability * len(values)))
    rank = min(max(rank, 1), len(values))
    return float(np.partition(values, rank - 1)[rank - 1])


@dataclass
class SplitConformalCalibrator:
    """Model-agnostic symmetric split-conformal prediction intervals.

    Coverage is calibrated on absolute residuals.  Marginal finite-sample
    coverage is justified only when calibration and evaluation examples are
    exchangeable; measuring its failure under domain shift is a benchmark
    result, not a violation of the implementation.
    """

    alpha: float = 0.10
    radius_: float | None = None
    calibration_size_: int = 0

    def fit(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> "SplitConformalCalibrator":
        y_true = _finite_vector(y_true, "y_true")
        y_pred = _finite_vector(y_pred, "y_pred")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        residuals = np.abs(y_true - y_pred)
        probability = min(1.0, np.ceil((len(residuals) + 1) * (1 - self.alpha)) / len(residuals))
        self.radius_ = _higher_quantile(residuals, probability)
        self.calibration_size_ = len(residuals)
        return self

    def predict_interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.radius_ is None:
            raise RuntimeError("calibrator must be fitted before prediction")
        y_pred = _finite_vector(y_pred, "y_pred")
        return y_pred - self.radius_, y_pred + self.radius_


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float | int]:
    """Coverage, width, and tail failures for prediction intervals."""

    y_true = _finite_vector(y_true, "y_true")
    lower = _finite_vector(lower, "lower")
    upper = _finite_vector(upper, "upper")
    if not (len(y_true) == len(lower) == len(upper)):
        raise ValueError("interval inputs must have the same length")
    if np.any(lower > upper):
        raise ValueError("lower interval bounds must not exceed upper bounds")
    width = upper - lower
    return {
        "n": int(len(y_true)),
        "coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "below_interval": float(np.mean(y_true < lower)),
        "above_interval": float(np.mean(y_true > upper)),
    }


@dataclass
class MahalanobisOODScorer:
    """Shrinkage-covariance distance in a source-fitted feature space."""

    location_: np.ndarray | None = None
    precision_: np.ndarray | None = None
    train_scores_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MahalanobisOODScorer":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 3 or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D array with at least three rows")
        covariance = LedoitWolf().fit(X)
        self.location_ = covariance.location_.copy()
        self.precision_ = covariance.precision_.copy()
        self.train_scores_ = self.score(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.location_ is None or self.precision_ is None:
            raise RuntimeError("OOD scorer must be fitted before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.location_) or not np.isfinite(X).all():
            raise ValueError("X must be finite and match the fitted feature width")
        centered = X - self.location_
        squared = np.einsum("ij,jk,ik->i", centered, self.precision_, centered)
        return np.sqrt(np.maximum(squared, 0.0))

    def percentile(self, X: np.ndarray) -> np.ndarray:
        if self.train_scores_ is None:
            raise RuntimeError("OOD scorer must be fitted before scoring")
        scores = self.score(X)
        ordered = np.sort(self.train_scores_)
        return np.searchsorted(ordered, scores, side="right") / len(ordered)


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ood_score: np.ndarray,
    coverages: Iterable[float] = (1.0, 0.9, 0.8, 0.7, 0.5),
) -> pd.DataFrame:
    """Evaluate MAE after retaining the least-OOD fraction of examples."""

    y_true = _finite_vector(y_true, "y_true")
    y_pred = _finite_vector(y_pred, "y_pred")
    ood_score = _finite_vector(ood_score, "ood_score")
    if not (len(y_true) == len(y_pred) == len(ood_score)):
        raise ValueError("risk-coverage inputs must have the same length")
    order = np.argsort(ood_score, kind="stable")
    rows: list[dict[str, float | int]] = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverages must be in (0, 1]")
        retained = max(1, int(np.ceil(coverage * len(order))))
        indices = order[:retained]
        metrics = summary_metrics(y_true[indices], y_pred[indices])
        rows.append(
            {
                "requested_coverage": float(coverage),
                "retained_fraction": float(retained / len(order)),
                "n": retained,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "ood_threshold": float(ood_score[indices].max()),
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Percentile intervals from a star-level cluster bootstrap."""

    y_true = _finite_vector(y_true, "y_true")
    y_pred = _finite_vector(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups).reshape(-1)
    if len(groups) != len(y_true):
        raise ValueError("groups must align with predictions")
    unique_groups = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(random_state)
    names = ("mae", "rmse", "bias", "scatter", "r2")
    samples = {name: np.empty(n_bootstrap, dtype=float) for name in names}
    for iteration in range(n_bootstrap):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([positions[group] for group in drawn])
        values = summary_metrics(y_true[indices], y_pred[indices])
        for name in names:
            samples[name][iteration] = values[name]
    estimate = summary_metrics(y_true, y_pred)
    tail = (1 - confidence) / 2
    return pd.DataFrame(
        [
            {
                "metric": name,
                "estimate": estimate[name],
                "ci_lower": float(np.quantile(samples[name], tail)),
                "ci_upper": float(np.quantile(samples[name], 1 - tail)),
                "confidence": confidence,
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_unit": "group",
                "group_count": int(len(unique_groups)),
            }
            for name in names
        ]
    )


def two_sample_mae_shift_bootstrap(
    source_y_true: np.ndarray,
    source_y_pred: np.ndarray,
    target_y_true: np.ndarray,
    target_y_pred: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    """Bootstrap the source-to-target MAE change over both stellar samples.

    Source-holdout and target-evaluation stars are resampled independently on
    every replicate. This carries uncertainty from both finite samples into the
    absolute MAE difference and relative cross-survey change, instead of treating
    the source-domain MAE as a fixed denominator.
    """

    source_y_true = _finite_vector(source_y_true, "source_y_true")
    source_y_pred = _finite_vector(source_y_pred, "source_y_pred")
    target_y_true = _finite_vector(target_y_true, "target_y_true")
    target_y_pred = _finite_vector(target_y_pred, "target_y_pred")
    if len(source_y_true) != len(source_y_pred):
        raise ValueError("source truth and predictions must have the same length")
    if len(target_y_true) != len(target_y_pred):
        raise ValueError("target truth and predictions must have the same length")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    source_errors = np.abs(source_y_pred - source_y_true)
    target_errors = np.abs(target_y_pred - target_y_true)
    source_mae = float(source_errors.mean())
    target_mae = float(target_errors.mean())
    if source_mae <= 0:
        raise ValueError("relative MAE change is undefined for zero source MAE")

    rng = np.random.default_rng(random_state)
    differences = np.empty(n_bootstrap, dtype=float)
    relative_changes = np.empty(n_bootstrap, dtype=float)
    for replicate in range(n_bootstrap):
        source_draw = float(
            source_errors[
                rng.integers(0, len(source_errors), size=len(source_errors))
            ].mean()
        )
        target_draw = float(
            target_errors[
                rng.integers(0, len(target_errors), size=len(target_errors))
            ].mean()
        )
        if source_draw <= 0:
            raise ValueError(
                "bootstrap produced zero source MAE; relative change is undefined"
            )
        differences[replicate] = target_draw - source_draw
        relative_changes[replicate] = 100.0 * (target_draw / source_draw - 1.0)

    tail = (1.0 - confidence) / 2.0
    return {
        "source_holdout_n": int(len(source_errors)),
        "target_evaluation_n": int(len(target_errors)),
        "source_holdout_mae": source_mae,
        "target_evaluation_mae": target_mae,
        "mae_difference": target_mae - source_mae,
        "mae_difference_ci_lower": float(np.quantile(differences, tail)),
        "mae_difference_ci_upper": float(np.quantile(differences, 1.0 - tail)),
        "cross_survey_mae_change_percent": 100.0
        * (target_mae / source_mae - 1.0),
        "cross_survey_change_ci_lower_percent": float(
            np.quantile(relative_changes, tail)
        ),
        "cross_survey_change_ci_upper_percent": float(
            np.quantile(relative_changes, 1.0 - tail)
        ),
        "confidence": confidence,
        "bootstrap_replicates": int(n_bootstrap),
        "bootstrap_scheme": "independent_source_and_target_stars",
    }


def two_axis_mae_bootstrap(
    prediction_rows: pd.DataFrame,
    clean_errors: pd.Series,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    """Bootstrap stars and perturbation seeds for a shifted MAE effect.

    ``prediction_rows`` must contain one absolute error for every seed-star
    pair.  ``clean_errors`` is indexed by the same star identifier.  Resampling
    both axes prevents noise-realization variability from masquerading as
    independent stellar evidence.
    """

    required = {"seed", "object_id", "abs_error"}
    missing = required - set(prediction_rows)
    if missing:
        raise ValueError(f"prediction_rows is missing columns: {sorted(missing)}")
    matrix = prediction_rows.pivot(index="seed", columns="object_id", values="abs_error")
    if matrix.isna().any().any():
        raise ValueError("every seed-star pair must have one absolute error")
    clean_errors = pd.Series(clean_errors, dtype=float)
    clean_errors = clean_errors.reindex(matrix.columns)
    if clean_errors.isna().any():
        raise ValueError("clean_errors must cover every object_id")
    rng = np.random.default_rng(random_state)
    values = matrix.to_numpy(dtype=float)
    clean = clean_errors.to_numpy(dtype=float)
    shifted_mae = np.empty(n_bootstrap)
    change_percent = np.empty(n_bootstrap)
    for iteration in range(n_bootstrap):
        seed_indices = rng.integers(0, values.shape[0], size=values.shape[0])
        star_indices = rng.integers(0, values.shape[1], size=values.shape[1])
        shifted = values[np.ix_(seed_indices, star_indices)].mean()
        baseline = clean[star_indices].mean()
        shifted_mae[iteration] = shifted
        change_percent[iteration] = 100 * (shifted / baseline - 1)
    tail = (1 - confidence) / 2
    observed_shifted = float(values.mean())
    observed_clean = float(clean.mean())
    return {
        "mae": observed_shifted,
        "mae_ci_lower": float(np.quantile(shifted_mae, tail)),
        "mae_ci_upper": float(np.quantile(shifted_mae, 1 - tail)),
        "clean_mae": observed_clean,
        "mae_change_percent": 100 * (observed_shifted / observed_clean - 1),
        "change_ci_lower": float(np.quantile(change_percent, tail)),
        "change_ci_upper": float(np.quantile(change_percent, 1 - tail)),
        "confidence": confidence,
        "bootstrap_replicates": n_bootstrap,
        "bootstrap_axes": "object_id,seed",
        "object_count": int(values.shape[1]),
        "seed_count": int(values.shape[0]),
    }


def two_axis_metric_bootstrap(
    prediction_rows: pd.DataFrame,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Intervals for all point metrics while resampling stars and shift seeds."""

    required = {"seed", "object_id", "y_true", "y_pred"}
    missing = required - set(prediction_rows)
    if missing:
        raise ValueError(f"prediction_rows is missing columns: {sorted(missing)}")
    truth = prediction_rows.pivot(index="seed", columns="object_id", values="y_true")
    prediction = prediction_rows.pivot(
        index="seed", columns="object_id", values="y_pred"
    )
    if not truth.index.equals(prediction.index) or not truth.columns.equals(
        prediction.columns
    ):
        raise ValueError("truth and predictions must share the seed-star grid")
    if truth.isna().any().any() or prediction.isna().any().any():
        raise ValueError("every seed-star pair must be present")
    if not np.allclose(truth.to_numpy(), truth.to_numpy()[0][None, :]):
        raise ValueError("y_true must be constant across seeds for each object")
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    truth_values = truth.to_numpy(dtype=float)
    prediction_values = prediction.to_numpy(dtype=float)
    names = ("mae", "rmse", "bias", "scatter", "r2")
    samples = {name: np.empty(n_bootstrap, dtype=float) for name in names}
    rng = np.random.default_rng(random_state)
    for iteration in range(n_bootstrap):
        seed_indices = rng.integers(
            0, prediction_values.shape[0], size=prediction_values.shape[0]
        )
        star_indices = rng.integers(
            0, prediction_values.shape[1], size=prediction_values.shape[1]
        )
        sampled_truth = truth_values[np.ix_(seed_indices, star_indices)].reshape(-1)
        sampled_prediction = prediction_values[
            np.ix_(seed_indices, star_indices)
        ].reshape(-1)
        metrics = summary_metrics(sampled_truth, sampled_prediction)
        for name in names:
            samples[name][iteration] = metrics[name]
    observed = summary_metrics(
        truth_values.reshape(-1), prediction_values.reshape(-1)
    )
    tail = (1 - confidence) / 2
    return pd.DataFrame(
        [
            {
                "metric": name,
                "estimate": observed[name],
                "ci_lower": float(np.quantile(samples[name], tail)),
                "ci_upper": float(np.quantile(samples[name], 1 - tail)),
                "confidence": confidence,
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_axes": "object_id,seed",
                "object_count": int(prediction_values.shape[1]),
                "seed_count": int(prediction_values.shape[0]),
            }
            for name in names
        ]
    )


@dataclass
class IsochroneManifold:
    """Nearest-manifold plausibility score for an explicit isochrone grid.

    This is an optional physical diagnostic.  No grid is bundled: callers must
    provide a cited grid appropriate for their population assumptions.  The
    threshold is calibrated from a reference sample rather than invented.
    """

    threshold_quantile: float = 0.99
    scaler_: StandardScaler | None = None
    neighbors_: NearestNeighbors | None = None
    threshold_: float | None = None

    def fit(
        self,
        isochrone_points: np.ndarray,
        reference_labels: np.ndarray,
    ) -> "IsochroneManifold":
        grid = np.asarray(isochrone_points, dtype=float)
        reference = np.asarray(reference_labels, dtype=float)
        if grid.ndim != 2 or grid.shape[1] != 3 or len(grid) < 4:
            raise ValueError("isochrone_points must have shape (n>=4, 3)")
        if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 4:
            raise ValueError("reference_labels must have shape (n>=4, 3)")
        if not np.isfinite(grid).all() or not np.isfinite(reference).all():
            raise ValueError("isochrone and reference values must be finite")
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("threshold_quantile must be in (0, 1)")
        scaler = StandardScaler().fit(reference)
        neighbors = NearestNeighbors(n_neighbors=1).fit(scaler.transform(grid))
        distances = neighbors.kneighbors(scaler.transform(reference))[0][:, 0]
        self.scaler_ = scaler
        self.neighbors_ = neighbors
        self.threshold_ = float(np.quantile(distances, self.threshold_quantile))
        return self

    def score(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.scaler_ is None or self.neighbors_ is None or self.threshold_ is None:
            raise RuntimeError("isochrone manifold must be fitted before scoring")
        labels = np.asarray(labels, dtype=float)
        if labels.ndim != 2 or labels.shape[1] != 3 or not np.isfinite(labels).all():
            raise ValueError("labels must be a finite array with three columns")
        distances = self.neighbors_.kneighbors(self.scaler_.transform(labels))[0][:, 0]
        return distances, distances > self.threshold_
