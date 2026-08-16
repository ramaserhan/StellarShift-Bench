"""Sensitivity of reported shift effects to formal reference-label errors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def formal_label_error_sensitivity(
    clean_predictions: pd.DataFrame,
    shifted_predictions: pd.DataFrame,
    label_uncertainties: pd.DataFrame,
    *,
    baseline_model: str = "source_only",
    noise_factor: float = 2.0,
    replicates: int = 2_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Propagate reported independent Gaussian label errors through MAE changes.

    This is a sensitivity analysis, not validation of the reference pipeline.
    It assumes the supplied formal errors are unbiased, calibrated one-sigma
    values and does not represent shared or parameter-dependent systematics.
    Each replicate resamples stars and shift seeds and draws one perturbed label
    per resampled star, preserving the clean-versus-shifted pairing.
    """

    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    required_prediction = {
        "object_id",
        "model",
        "target",
        "y_true",
        "y_pred",
        "noise_factor",
        "seed",
    }
    for name, frame in (
        ("clean_predictions", clean_predictions),
        ("shifted_predictions", shifted_predictions),
    ):
        missing = sorted(required_prediction - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    if "object_id" not in label_uncertainties:
        raise ValueError("label_uncertainties must contain object_id")

    rng = np.random.default_rng(random_state)
    alpha = 1.0 - confidence
    rows: list[dict[str, object]] = []
    clean = clean_predictions.loc[
        (clean_predictions["model"] == baseline_model)
        & (clean_predictions["noise_factor"] == 1.0)
    ]
    shifted = shifted_predictions.loc[
        (shifted_predictions["model"] == baseline_model)
        & np.isclose(shifted_predictions["noise_factor"], noise_factor)
    ]

    for target in sorted(set(clean["target"]) & set(shifted["target"])):
        error_column = f"{target}_err"
        if error_column not in label_uncertainties:
            raise ValueError(
                f"label_uncertainties is missing required column {error_column}"
            )
        clean_target = clean.loc[clean["target"] == target].copy()
        clean_target = clean_target.merge(
            label_uncertainties[["object_id", error_column]],
            on="object_id",
            how="inner",
            validate="one_to_one",
        ).sort_values("object_id")
        shifted_target = shifted.loc[shifted["target"] == target].copy()
        shifted_wide = shifted_target.pivot(
            index="seed", columns="object_id", values="y_pred"
        )
        object_ids = clean_target["object_id"].to_numpy()
        shifted_wide = shifted_wide.reindex(columns=object_ids)
        if shifted_wide.isna().any().any() or len(clean_target) == 0:
            raise ValueError(f"clean and shifted objects do not align for {target}")

        truth = clean_target["y_true"].to_numpy(dtype=float)
        clean_pred = clean_target["y_pred"].to_numpy(dtype=float)
        sigma = clean_target[error_column].to_numpy(dtype=float)
        shifted_pred = shifted_wide.to_numpy(dtype=float)
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError(f"formal label errors must be positive for {target}")

        point_clean = float(np.mean(np.abs(clean_pred - truth)))
        point_shifted = float(np.mean(np.abs(shifted_pred - truth[None, :])))
        point_change = 100.0 * (point_shifted / point_clean - 1.0)
        simulated = np.empty((replicates, 3), dtype=float)
        n_stars = len(truth)
        n_seeds = shifted_pred.shape[0]
        for replicate in range(replicates):
            star_index = rng.integers(0, n_stars, size=n_stars)
            seed_index = rng.integers(0, n_seeds, size=n_seeds)
            perturbed_truth = truth[star_index] + rng.normal(
                0.0, sigma[star_index]
            )
            clean_mae = float(
                np.mean(np.abs(clean_pred[star_index] - perturbed_truth))
            )
            shifted_mae = float(
                np.mean(
                    np.abs(
                        shifted_pred[seed_index][:, star_index]
                        - perturbed_truth[None, :]
                    )
                )
            )
            simulated[replicate] = (
                clean_mae,
                shifted_mae,
                100.0 * (shifted_mae / clean_mae - 1.0),
            )

        quantiles = np.quantile(
            simulated, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0
        )
        rows.append(
            {
                "model": baseline_model,
                "target": target,
                "noise_factor": noise_factor,
                "n_stars": n_stars,
                "n_shift_seeds": n_seeds,
                "point_clean_mae_original_labels": point_clean,
                "point_shifted_mae_original_labels": point_shifted,
                "point_change_percent_original_labels": point_change,
                "sensitivity_clean_mae_median": float(np.median(simulated[:, 0])),
                "sensitivity_clean_mae_ci_lower": float(quantiles[0, 0]),
                "sensitivity_clean_mae_ci_upper": float(quantiles[1, 0]),
                "sensitivity_shifted_mae_median": float(np.median(simulated[:, 1])),
                "sensitivity_shifted_mae_ci_lower": float(quantiles[0, 1]),
                "sensitivity_shifted_mae_ci_upper": float(quantiles[1, 1]),
                "sensitivity_change_percent_median": float(
                    np.median(simulated[:, 2])
                ),
                "sensitivity_change_ci_lower": float(quantiles[0, 2]),
                "sensitivity_change_ci_upper": float(quantiles[1, 2]),
                "fraction_change_above_zero": float(
                    np.mean(simulated[:, 2] > 0)
                ),
                "replicates": replicates,
                "confidence": confidence,
                "assumption": "independent_unbiased_gaussian_formal_errors",
                "scope": "formal_error_sensitivity_not_independent_label_validation",
            }
        )
    return pd.DataFrame(rows)


def formal_label_error_transfer_sensitivity(
    predictions: pd.DataFrame,
    label_uncertainties: pd.DataFrame,
    *,
    baseline_method: str = "source_only",
    replicates: int = 2_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Propagate formal label errors through cross-survey MAE comparisons.

    Stars are resampled and one perturbed reference label is shared across all
    methods within each replicate.  The resulting paired method differences
    therefore retain the benchmark's object-level pairing.  As in
    :func:`formal_label_error_sensitivity`, this does not model common APOGEE
    systematics or validate the reference scale independently.
    """

    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    required = {"object_id", "method", "target", "y_true", "y_pred"}
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"predictions is missing columns: {missing}")
    if "object_id" not in label_uncertainties:
        raise ValueError("label_uncertainties must contain object_id")

    rng = np.random.default_rng(random_state)
    alpha = 1.0 - confidence
    rows: list[dict[str, object]] = []
    for target, target_frame in predictions.groupby("target"):
        error_column = f"{target}_err"
        if error_column not in label_uncertainties:
            raise ValueError(
                f"label_uncertainties is missing required column {error_column}"
            )
        wide = target_frame.pivot(
            index="object_id", columns="method", values="y_pred"
        ).sort_index()
        truth = (
            target_frame.drop_duplicates("object_id")
            .set_index("object_id")["y_true"]
            .reindex(wide.index)
            .to_numpy(dtype=float)
        )
        sigma = (
            label_uncertainties.set_index("object_id")[error_column]
            .reindex(wide.index)
            .to_numpy(dtype=float)
        )
        if baseline_method not in wide or not np.isfinite(wide.to_numpy()).all():
            raise ValueError(f"predictions do not align for target {target}")
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError(f"formal label errors must be positive for {target}")

        methods = list(wide.columns)
        predictions_array = wide.to_numpy(dtype=float)
        baseline_index = methods.index(baseline_method)
        point_mae = np.mean(np.abs(predictions_array - truth[:, None]), axis=0)
        simulated_mae = np.empty((replicates, len(methods)), dtype=float)
        n_stars = len(truth)
        for replicate in range(replicates):
            indices = rng.integers(0, n_stars, size=n_stars)
            perturbed_truth = truth[indices] + rng.normal(0.0, sigma[indices])
            simulated_mae[replicate] = np.mean(
                np.abs(predictions_array[indices] - perturbed_truth[:, None]),
                axis=0,
            )

        for method_index, method in enumerate(methods):
            mae_values = simulated_mae[:, method_index]
            difference_values = mae_values - simulated_mae[:, baseline_index]
            mae_ci = np.quantile(
                mae_values, [alpha / 2.0, 1.0 - alpha / 2.0]
            )
            difference_ci = np.quantile(
                difference_values, [alpha / 2.0, 1.0 - alpha / 2.0]
            )
            rows.append(
                {
                    "method": method,
                    "baseline": baseline_method,
                    "target": target,
                    "n_stars": n_stars,
                    "point_mae_original_labels": float(point_mae[method_index]),
                    "point_mae_difference_vs_source": float(
                        point_mae[method_index] - point_mae[baseline_index]
                    ),
                    "sensitivity_mae_median": float(np.median(mae_values)),
                    "sensitivity_mae_ci_lower": float(mae_ci[0]),
                    "sensitivity_mae_ci_upper": float(mae_ci[1]),
                    "sensitivity_difference_median": float(
                        np.median(difference_values)
                    ),
                    "sensitivity_difference_ci_lower": float(difference_ci[0]),
                    "sensitivity_difference_ci_upper": float(difference_ci[1]),
                    "fraction_difference_below_zero": float(
                        np.mean(difference_values < 0)
                    ),
                    "replicates": replicates,
                    "confidence": confidence,
                    "assumption": "independent_unbiased_gaussian_formal_errors",
                    "scope": (
                        "formal_error_sensitivity_not_independent_label_validation"
                    ),
                }
            )
    return pd.DataFrame(rows)
