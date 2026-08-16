"""End-to-end controlled S/N benchmark on real DESI R-arm spectra."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import DesiSNRConfig
from .data.desi import (
    config_manifest,
    continuum_normalize,
    create_split_manifest,
    inject_noise,
    load_desi_npz,
)
from .eval.metrics import paired_significance, summary_metrics
from .models.spectral import SeparateExtraTreesRegressor, SpectralFeaturePipeline


TARGET_NAMES = ("teff", "logg", "feh")


def run_desi_snr_experiment(config: DesiSNRConfig) -> dict[str, Path]:
    """Run clean, zero-shot S/N-shift, retraining, and augmentation baselines."""

    data = load_desi_npz(config.input_npz)
    manifest = create_split_manifest(data, config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    selected_manifest = (
        manifest.loc[manifest["selected"]]
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    selected_rows = selected_manifest["row_index"].to_numpy(dtype=int)
    split = selected_manifest["split"].to_numpy(dtype=str)
    partitions = {
        name: np.flatnonzero(split == name)
        for name in (
            "source_train",
            "source_holdout",
            "target_adaptation",
            "target_evaluation",
        )
    }
    targets = {
        name: np.asarray(data[name][selected_rows], dtype=np.float32)
        for name in TARGET_NAMES
    }

    normalized_clean, valid_clean = continuum_normalize(
        data["flux"][selected_rows],
        data["valid"][selected_rows],
        config.continuum_window,
        config.continuum_polyorder,
    )
    train = partitions["source_train"]
    feature_pipeline = SpectralFeaturePipeline(
        wavelength_min=config.wavelength_min,
        wavelength_max=config.wavelength_max,
        outlier_percentiles=tuple(config.outlier_percentiles),
        pca_components=config.pca_components,
        random_state=config.random_state,
    ).fit(
        normalized_clean[train],
        valid_clean[train],
        data["wavelength"],
    )
    clean_features, clean_extreme_count = feature_pipeline.transform(
        normalized_clean, valid_clean
    )

    source_model = _new_model(config).fit(
        clean_features[train], _slice_targets(targets, train)
    )
    augmented_model = _fit_noise_augmented_model(
        config,
        data,
        selected_rows,
        train,
        clean_features,
        targets,
        feature_pipeline,
    )
    models = {"source_only": source_model}
    if augmented_model is not None:
        models["noise_augmented"] = augmented_model

    clean_metrics, clean_predictions = _evaluate_clean_partitions(
        models, clean_features, targets, partitions
    )
    clean_metrics_path = output_dir / "clean_metrics.csv"
    clean_metrics.to_csv(clean_metrics_path, index=False)

    trials = _run_robustness_sweep(
        config,
        data,
        selected_rows,
        partitions["target_evaluation"],
        targets,
        models,
        clean_predictions,
        feature_pipeline,
    )
    trials_path = output_dir / "snr_trials.csv"
    trials.to_csv(trials_path, index=False)
    robustness_summary = _summarize_trials(trials)
    robustness_path = output_dir / "snr_summary.csv"
    robustness_summary.to_csv(robustness_path, index=False)

    curve_path = output_dir / "snr_robustness_curve.png"
    _plot_source_only_curve(robustness_summary, curve_path)
    mitigation_path = output_dir / "mitigation_comparison.png"
    _plot_mitigation_comparison(
        robustness_summary, clean_metrics, mitigation_path
    )

    adaptation_metrics, adaptation_comparison = _run_target_retraining(
        config,
        data,
        selected_rows,
        partitions,
        targets,
        clean_features,
        source_model,
        feature_pipeline,
    )
    adaptation_metrics_path = output_dir / "adaptation_metrics.csv"
    adaptation_comparison_path = output_dir / "adaptation_comparison.csv"
    adaptation_metrics.to_csv(adaptation_metrics_path, index=False)
    adaptation_comparison.to_csv(adaptation_comparison_path, index=False)

    experiment_manifest = config_manifest(config)
    experiment_manifest.update(
        {
            "partition_rows": {
                name: int(len(indices)) for name, indices in partitions.items()
            },
            "excluded_rows": int((manifest["split"] == "excluded").sum()),
            "pca_explained_variance": feature_pipeline.explained_variance_ratio,
            "train_derived_outlier_bounds": list(
                feature_pipeline.outlier_bounds_ or ()
            ),
            "clean_extreme_pixels_masked": clean_extreme_count,
            "augmentation_strategy": (
                "two deterministic noisy views with clean and augmented domains "
                "receiving equal total sample weight"
                if config.augmentation_views
                else "disabled"
            ),
        }
    )
    experiment_manifest_path = output_dir / "manifest.json"
    experiment_manifest_path.write_text(
        json.dumps(experiment_manifest, indent=2), encoding="utf-8"
    )

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(
        _render_summary(
            config,
            manifest,
            clean_metrics,
            robustness_summary,
            adaptation_comparison,
            feature_pipeline.explained_variance_ratio,
        ),
        encoding="utf-8",
    )
    return {
        "manifest": experiment_manifest_path,
        "split_manifest": manifest_path,
        "clean_metrics": clean_metrics_path,
        "snr_trials": trials_path,
        "snr_summary": robustness_path,
        "adaptation_metrics": adaptation_metrics_path,
        "adaptation_comparison": adaptation_comparison_path,
        "robustness_curve": curve_path,
        "mitigation_comparison": mitigation_path,
        "summary": summary_path,
    }


def _new_model(config: DesiSNRConfig) -> SeparateExtraTreesRegressor:
    return SeparateExtraTreesRegressor(
        target_names=TARGET_NAMES,
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=config.random_state,
    )


def _fit_noise_augmented_model(
    config: DesiSNRConfig,
    data: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    train: np.ndarray,
    clean_features: np.ndarray,
    targets: dict[str, np.ndarray],
    feature_pipeline: SpectralFeaturePipeline,
) -> SeparateExtraTreesRegressor | None:
    if config.augmentation_views == 0:
        return None
    source_rows = selected_rows[train]
    augmented_features: list[np.ndarray] = []
    factor_values = np.asarray(config.noise_factors, dtype=float)
    for view in range(config.augmentation_views):
        factors = np.resize(factor_values, len(source_rows)).copy()
        rng = np.random.default_rng(config.augmentation_seed_start + view)
        rng.shuffle(factors)
        shifted_flux = inject_noise(
            data["flux"][source_rows],
            data["ivar"][source_rows],
            data["valid"][source_rows],
            data["targetid"][source_rows],
            factor=factors,
            seed=config.augmentation_seed_start + view,
        )
        normalized, valid = continuum_normalize(
            shifted_flux,
            data["valid"][source_rows],
            config.continuum_window,
            config.continuum_polyorder,
        )
        features, _ = feature_pipeline.transform(normalized, valid)
        augmented_features.append(features)

    training_features = np.vstack(
        [clean_features[train], *augmented_features]
    )
    training_targets = {
        name: np.concatenate(
            [targets[name][train]] * (1 + config.augmentation_views)
        )
        for name in TARGET_NAMES
    }
    # Clean spectra and all noisy views each receive half the total weight.
    weights = np.concatenate(
        [
            np.ones(len(train)),
            *[
                np.full(len(train), 1.0 / config.augmentation_views)
                for _ in range(config.augmentation_views)
            ],
        ]
    )
    return _new_model(config).fit(
        training_features, training_targets, sample_weight=weights
    )


def _evaluate_clean_partitions(
    models: dict[str, SeparateExtraTreesRegressor],
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    partitions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, object]] = []
    target_eval_predictions: dict[tuple[str, str], np.ndarray] = {}
    for evaluation, partition_name in (
        ("source_holdout_clean", "source_holdout"),
        ("target_evaluation_clean", "target_evaluation"),
    ):
        indices = partitions[partition_name]
        for model_name, model in models.items():
            predictions = model.predict(features[indices])
            for target_name in TARGET_NAMES:
                metrics = summary_metrics(
                    targets[target_name][indices], predictions[target_name]
                )
                rows.append(
                    {
                        "evaluation": evaluation,
                        "model": model_name,
                        "target": target_name,
                        **metrics,
                    }
                )
                if partition_name == "target_evaluation":
                    target_eval_predictions[(model_name, target_name)] = predictions[
                        target_name
                    ]
    return pd.DataFrame(rows), target_eval_predictions


def _run_robustness_sweep(
    config: DesiSNRConfig,
    data: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    target_eval: np.ndarray,
    targets: dict[str, np.ndarray],
    models: dict[str, SeparateExtraTreesRegressor],
    clean_predictions: dict[tuple[str, str], np.ndarray],
    feature_pipeline: SpectralFeaturePipeline,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    raw_rows = selected_rows[target_eval]
    y_true = {name: targets[name][target_eval] for name in TARGET_NAMES}
    for seed in range(
        config.noise_seed_start, config.noise_seed_start + config.noise_seed_count
    ):
        for factor in config.noise_factors:
            shifted_flux = inject_noise(
                data["flux"][raw_rows],
                data["ivar"][raw_rows],
                data["valid"][raw_rows],
                data["targetid"][raw_rows],
                factor=factor,
                seed=seed,
            )
            normalized, valid = continuum_normalize(
                shifted_flux,
                data["valid"][raw_rows],
                config.continuum_window,
                config.continuum_polyorder,
            )
            features, extreme_count = feature_pipeline.transform(normalized, valid)
            predictions_by_model = {
                model_name: model.predict(features)
                for model_name, model in models.items()
            }
            for model_name, predictions in predictions_by_model.items():
                for target_name in TARGET_NAMES:
                    metrics = summary_metrics(
                        y_true[target_name], predictions[target_name]
                    )
                    clean_prediction = clean_predictions[(model_name, target_name)]
                    clean_metrics = summary_metrics(
                        y_true[target_name], clean_prediction
                    )
                    p_value = paired_significance(
                        predictions[target_name] - y_true[target_name],
                        clean_prediction - y_true[target_name],
                    )
                    source_prediction = predictions_by_model["source_only"][target_name]
                    source_metrics = summary_metrics(
                        y_true[target_name], source_prediction
                    )
                    if model_name == "source_only":
                        model_comparison_p = 1.0
                    else:
                        model_comparison_p = paired_significance(
                            predictions[target_name] - y_true[target_name],
                            source_prediction - y_true[target_name],
                        )
                    rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "noise_factor": factor,
                            "approximate_median_snr": float(
                                np.median(data["sn_r"][raw_rows] / factor)
                            ),
                            "target": target_name,
                            **metrics,
                            "clean_mae": clean_metrics["mae"],
                            "mae_change_percent": 100
                            * (metrics["mae"] / clean_metrics["mae"] - 1),
                            "wilcoxon_p": p_value,
                            "significant_at_0.05": p_value < 0.05,
                            "source_only_mae": source_metrics["mae"],
                            "mae_vs_source_percent": 100
                            * (metrics["mae"] / source_metrics["mae"] - 1),
                            "model_comparison_p": model_comparison_p,
                            "significant_improvement_vs_source": (
                                model_name != "source_only"
                                and metrics["mae"] < source_metrics["mae"]
                                and model_comparison_p < 0.05
                            ),
                            "extreme_pixels_masked": extreme_count,
                        }
                    )
    return pd.DataFrame(rows)


def _summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["model", "target", "noise_factor"], as_index=False)
        .agg(
            approximate_median_snr=("approximate_median_snr", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_change_percent_mean=("mae_change_percent", "mean"),
            mae_change_percent_std=("mae_change_percent", "std"),
            significant_runs=("significant_at_0.05", "sum"),
            mae_vs_source_percent_mean=("mae_vs_source_percent", "mean"),
            mae_vs_source_percent_std=("mae_vs_source_percent", "std"),
            significant_improvement_runs=(
                "significant_improvement_vs_source",
                "sum",
            ),
        )
        .sort_values(["model", "target", "noise_factor"])
        .reset_index(drop=True)
    )


def _run_target_retraining(
    config: DesiSNRConfig,
    data: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    partitions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    clean_features: np.ndarray,
    source_model: SeparateExtraTreesRegressor,
    feature_pipeline: SpectralFeaturePipeline,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_train = partitions["source_train"]
    target_adapt = partitions["target_adaptation"]
    target_eval = partitions["target_evaluation"]
    shifted_partitions: dict[str, np.ndarray] = {}
    for name, indices in (
        ("target_adaptation", target_adapt),
        ("target_evaluation", target_eval),
    ):
        raw_rows = selected_rows[indices]
        shifted_flux = inject_noise(
            data["flux"][raw_rows],
            data["ivar"][raw_rows],
            data["valid"][raw_rows],
            data["targetid"][raw_rows],
            factor=config.adaptation_noise_factor,
            seed=config.adaptation_noise_seed,
        )
        normalized, valid = continuum_normalize(
            shifted_flux,
            data["valid"][raw_rows],
            config.continuum_window,
            config.continuum_polyorder,
        )
        shifted_partitions[name], _ = feature_pipeline.transform(normalized, valid)

    target_weight = len(source_train) / len(target_adapt)
    training_features = np.vstack(
        [clean_features[source_train], shifted_partitions["target_adaptation"]]
    )
    training_targets = {
        name: np.concatenate(
            [targets[name][source_train], targets[name][target_adapt]]
        )
        for name in TARGET_NAMES
    }
    weights = np.concatenate(
        [np.ones(len(source_train)), np.full(len(target_adapt), target_weight)]
    )
    retrained_model = _new_model(config).fit(
        training_features, training_targets, sample_weight=weights
    )

    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for evaluation, indices, features in (
        (
            "source_holdout_clean",
            partitions["source_holdout"],
            clean_features[partitions["source_holdout"]],
        ),
        (
            "target_evaluation_shifted",
            target_eval,
            shifted_partitions["target_evaluation"],
        ),
    ):
        before = source_model.predict(features)
        after = retrained_model.predict(features)
        for target_name in TARGET_NAMES:
            y_true = targets[target_name][indices]
            before_metrics = summary_metrics(y_true, before[target_name])
            after_metrics = summary_metrics(y_true, after[target_name])
            metric_rows.extend(
                [
                    {
                        "evaluation": evaluation,
                        "model": "source_only",
                        "target": target_name,
                        **before_metrics,
                    },
                    {
                        "evaluation": evaluation,
                        "model": "source_plus_target_retrained",
                        "target": target_name,
                        **after_metrics,
                    },
                ]
            )
            p_value = paired_significance(
                before[target_name] - y_true, after[target_name] - y_true
            )
            comparison_rows.append(
                {
                    "evaluation": evaluation,
                    "target": target_name,
                    "mae_before": before_metrics["mae"],
                    "mae_after": after_metrics["mae"],
                    "mae_change_percent": 100
                    * (after_metrics["mae"] / before_metrics["mae"] - 1),
                    "rmse_before": before_metrics["rmse"],
                    "rmse_after": after_metrics["rmse"],
                    "bias_before": before_metrics["bias"],
                    "bias_after": after_metrics["bias"],
                    "wilcoxon_p": p_value,
                    "significant_at_0.05": p_value < 0.05,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(comparison_rows)


def _plot_source_only_curve(summary: pd.DataFrame, output_path: Path) -> None:
    source = summary.loc[summary["model"] == "source_only"]
    figure, axis = plt.subplots(figsize=(9, 6))
    for target_name in TARGET_NAMES:
        values = source.loc[source["target"] == target_name]
        axis.errorbar(
            values["noise_factor"],
            values["mae_change_percent_mean"],
            yerr=values["mae_change_percent_std"],
            marker="o",
            capsize=4,
            label=target_name.upper(),
        )
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Noise standard-deviation multiplier")
    axis.set_ylabel("Mean MAE change from clean baseline (%)")
    axis.set_title("Zero-shot sensitivity to controlled S/N degradation")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_mitigation_comparison(
    summary: pd.DataFrame, clean_metrics: pd.DataFrame, output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    model_labels = {
        "source_only": "Source only",
        "noise_augmented": "Noise augmented",
    }
    for axis, target_name in zip(axes, TARGET_NAMES):
        for model_name in summary["model"].unique():
            values = summary.loc[
                (summary["target"] == target_name)
                & (summary["model"] == model_name)
            ]
            clean_row = clean_metrics.loc[
                (clean_metrics["evaluation"] == "target_evaluation_clean")
                & (clean_metrics["model"] == model_name)
                & (clean_metrics["target"] == target_name)
            ]
            x = np.concatenate([[1.0], values["noise_factor"].to_numpy()])
            y = np.concatenate([[clean_row["mae"].iloc[0]], values["mae_mean"]])
            error = np.concatenate([[0.0], values["mae_std"]])
            axis.errorbar(
                x,
                y,
                yerr=error,
                marker="o",
                capsize=3,
                label=model_labels.get(model_name, model_name),
            )
        axis.set_title(target_name.upper())
        axis.set_xlabel("Noise multiplier")
        axis.set_ylabel("MAE")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("Noise augmentation mitigation on held-out DESI stars")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _render_summary(
    config: DesiSNRConfig,
    manifest: pd.DataFrame,
    clean_metrics: pd.DataFrame,
    robustness: pd.DataFrame,
    adaptation: pd.DataFrame,
    explained_variance: float,
) -> str:
    source_robustness = robustness.loc[robustness["model"] == "source_only"]
    factor_two = source_robustness.loc[
        np.isclose(source_robustness["noise_factor"], 2.0)
    ]
    return "\n\n".join(
        [
            f"=== {config.name} ===\n"
            f"Selected spectra: {int(manifest['selected'].sum())} / {len(manifest)}\n"
            f"PCA explained variance: {explained_variance:.4f}",
            "=== Clean held-out performance ===\n"
            + clean_metrics.to_string(index=False),
            "=== Source-only robustness at 2x noise ===\n"
            + factor_two.to_string(index=False),
            "=== Source+target retraining comparison ===\n"
            + adaptation.to_string(index=False),
            "Interpretation: this is a controlled Gaussian S/N perturbation on real "
            "DESI spectra. RVSpecFit outputs are reference labels, not independent "
            "ground truth. Negative or nonsignificant adaptation results are retained.",
        ]
    ) + "\n"


def _slice_targets(
    targets: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: targets[name][indices] for name in TARGET_NAMES}
