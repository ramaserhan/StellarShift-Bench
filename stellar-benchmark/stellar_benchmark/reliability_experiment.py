"""DESI reliability benchmark: calibration, OOD, uncertainty, and adaptation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .adaptation.coral import CORALAdapter
from .config import DesiReliabilityConfig
from .data.desi import (
    continuum_normalize,
    create_split_manifest,
    inject_noise,
    load_desi_npz,
)
from .desi_experiment import TARGET_NAMES, _fit_noise_augmented_model, _new_model
from .eval.metrics import paired_significance, summary_metrics
from .eval.precision import paired_adaptation_effects
from .eval.reference_labels import formal_label_error_sensitivity
from .eval.reliability import (
    IsochroneManifold,
    MahalanobisOODScorer,
    SplitConformalCalibrator,
    cluster_bootstrap_metrics,
    interval_metrics,
    risk_coverage_curve,
    two_axis_mae_bootstrap,
    two_axis_metric_bootstrap,
)
from .models.families import SeparateFamilyRegressor
from .models.spectral import SpectralFeaturePipeline


def run_desi_reliability_experiment(
    config: DesiReliabilityConfig,
) -> dict[str, Path]:
    """Run the extended reliability audit without changing v0.3 outputs."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_desi_npz(config.input_npz)
    manifest = create_split_manifest(data, config)
    split_manifest_path = output_dir / "split_manifest.csv"
    manifest.to_csv(split_manifest_path, index=False)

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
    source_train = partitions["source_train"]
    feature_pipeline = SpectralFeaturePipeline(
        wavelength_min=config.wavelength_min,
        wavelength_max=config.wavelength_max,
        outlier_percentiles=tuple(config.outlier_percentiles),
        pca_components=config.pca_components,
        random_state=config.random_state,
    ).fit(
        normalized_clean[source_train],
        valid_clean[source_train],
        data["wavelength"],
    )
    clean_features, _ = feature_pipeline.transform(normalized_clean, valid_clean)
    target_dict = _slice_targets(targets, source_train)
    source_model = _new_model(config).fit(clean_features[source_train], target_dict)
    augmented_model = _fit_noise_augmented_model(
        config,
        data,
        selected_rows,
        source_train,
        clean_features,
        targets,
        feature_pipeline,
    )
    models = {"source_only": source_model}
    if augmented_model is not None:
        models["noise_augmented"] = augmented_model

    ood_scorer = MahalanobisOODScorer().fit(clean_features[source_train])
    clean_rows, clean_metrics, clean_predictions = _evaluate_clean(
        data,
        selected_rows,
        selected_manifest,
        partitions,
        targets,
        clean_features,
        models,
        ood_scorer,
    )
    clean_predictions_path = output_dir / "per_star_clean_predictions.csv"
    clean_metrics_path = output_dir / "clean_metrics.csv"
    clean_rows.to_csv(clean_predictions_path, index=False)
    clean_metrics.to_csv(clean_metrics_path, index=False)
    clean_intervals = _cluster_metric_intervals(
        clean_rows.loc[
            clean_rows["evaluation"] == "target_evaluation_clean"
        ],
        ["model", "target"],
        config,
    )
    clean_intervals_path = output_dir / "clean_metric_intervals.csv"
    clean_intervals.to_csv(clean_intervals_path, index=False)

    calibrators = _calibrators_from_clean_rows(
        clean_rows, config.conformal_alphas
    )
    shifted_rows, shifted_metrics, calibration_rows = _evaluate_shifts(
        config,
        data,
        selected_rows,
        selected_manifest,
        partitions["target_evaluation"],
        targets,
        models,
        feature_pipeline,
        ood_scorer,
        calibrators,
    )
    clean_calibration = _evaluate_clean_target_calibration(
        clean_rows, calibrators, config.conformal_alphas
    )
    calibration_rows = pd.concat(
        [clean_calibration, calibration_rows], ignore_index=True
    )
    shifted_predictions_path = output_dir / "per_star_shift_predictions.csv"
    shifted_metrics_path = output_dir / "shift_metrics_by_seed.csv"
    calibration_metrics_path = output_dir / "calibration_under_shift.csv"
    shifted_rows.to_csv(shifted_predictions_path, index=False)
    shifted_metrics.to_csv(shifted_metrics_path, index=False)
    calibration_rows.to_csv(calibration_metrics_path, index=False)

    bootstrap = _bootstrap_shift_effects(config, clean_rows, shifted_rows)
    bootstrap_path = output_dir / "nested_bootstrap_intervals.csv"
    bootstrap.to_csv(bootstrap_path, index=False)
    shifted_intervals = _two_axis_metric_intervals(config, shifted_rows)
    shifted_intervals_path = output_dir / "shift_metric_intervals.csv"
    shifted_intervals.to_csv(shifted_intervals_path, index=False)
    label_uncertainties = selected_manifest[
        ["targetid", "teff_err", "logg_err", "feh_err"]
    ].rename(columns={"targetid": "object_id"})
    label_sensitivity = formal_label_error_sensitivity(
        clean_rows.loc[
            clean_rows["evaluation"] == "target_evaluation_clean"
        ],
        shifted_rows,
        label_uncertainties,
        noise_factor=config.adaptation_noise_factor,
        replicates=max(1000, config.bootstrap_replicates),
        confidence=config.bootstrap_confidence,
        random_state=config.random_state + 613,
    )
    label_sensitivity_path = output_dir / "reference_label_sensitivity.csv"
    label_sensitivity.to_csv(label_sensitivity_path, index=False)

    adaptation_rows, adaptation_metrics, target_calibration, label_budget = (
        _evaluate_adaptation_ladder(
            config,
            data,
            selected_rows,
            selected_manifest,
            partitions,
            targets,
            clean_features,
            source_model,
            augmented_model,
            feature_pipeline,
            calibrators,
            ood_scorer,
        )
    )
    adaptation_predictions_path = output_dir / "adaptation_per_star.csv"
    adaptation_metrics_path = output_dir / "adaptation_ladder.csv"
    target_calibration_path = output_dir / "target_recalibration.csv"
    label_budget_path = output_dir / "label_budget_trials.csv"
    adaptation_rows.to_csv(adaptation_predictions_path, index=False)
    adaptation_metrics.to_csv(adaptation_metrics_path, index=False)
    target_calibration.to_csv(target_calibration_path, index=False)
    label_budget.to_csv(label_budget_path, index=False)
    adaptation_intervals = _cluster_metric_intervals(
        adaptation_rows, ["method", "target"], config
    )
    adaptation_intervals_path = output_dir / "adaptation_metric_intervals.csv"
    adaptation_intervals.to_csv(adaptation_intervals_path, index=False)
    adaptation_effects, precision_plan = paired_adaptation_effects(
        adaptation_rows,
        bootstrap_replicates=max(1000, config.bootstrap_replicates),
        confidence=config.bootstrap_confidence,
        planning_margin_fraction=0.05,
        planning_power=0.80,
        random_state=config.random_state + 811,
    )
    adaptation_effects_path = output_dir / "adaptation_effect_intervals.csv"
    precision_plan_path = output_dir / "adaptation_precision_plan.csv"
    adaptation_effects.to_csv(adaptation_effects_path, index=False)
    precision_plan.to_csv(precision_plan_path, index=False)

    subgroup = _subgroup_metrics(config, shifted_rows)
    subgroup_path = output_dir / "subgroup_metrics.csv"
    subgroup.to_csv(subgroup_path, index=False)
    subgroup_support = _subgroup_support(
        selected_manifest, partitions["target_evaluation"], config
    )
    subgroup_support_path = output_dir / "subgroup_support.csv"
    subgroup_support.to_csv(subgroup_support_path, index=False)
    risk = _selective_prediction(config, shifted_rows)
    risk_path = output_dir / "risk_coverage.csv"
    risk.to_csv(risk_path, index=False)

    ablation = _model_ablation(
        config,
        data,
        selected_rows,
        partitions,
        targets,
        clean_features,
        feature_pipeline,
    )
    ablation_path = output_dir / "model_ablation.csv"
    ablation.to_csv(ablation_path, index=False)
    representation = _representation_ablation(
        config,
        data,
        selected_rows,
        partitions,
        targets,
        normalized_clean,
        valid_clean,
    )
    representation_path = output_dir / "representation_ablation.csv"
    representation.to_csv(representation_path, index=False)

    plausibility_path = output_dir / "physical_plausibility.json"
    _physical_plausibility(
        config,
        adaptation_rows,
        targets,
        partitions["source_train"],
        plausibility_path,
    )

    calibration_plot = output_dir / "calibration_under_shift.png"
    risk_plot = output_dir / "risk_coverage.png"
    label_budget_plot = output_dir / "label_budget.png"
    ablation_plot = output_dir / "model_ablation.png"
    representation_plot = output_dir / "representation_ablation.png"
    _plot_calibration(calibration_rows, calibration_plot)
    _plot_risk(risk, risk_plot)
    _plot_label_budget(label_budget, label_budget_plot)
    _plot_ablation(ablation, ablation_plot)
    _plot_representation(representation, representation_plot)

    manifest_path = output_dir / "manifest.json"
    experiment_manifest = {
        "experiment": asdict(config),
        "version": "1.2.3",
        "reference_labels": "DESI DR1 Stellar Reddening VAC RVSpecFit RVTAB",
        "label_caveat": (
            "RVSpecFit outputs are pipeline-derived reference labels, not "
            "independent astrophysical ground truth."
        ),
        "shift_caveat": (
            "This executed case study uses Gaussian S/N perturbations on real "
            "DESI spectra; the cross-survey workflow is a separate experiment."
        ),
        "bootstrap_axes": ["TARGETID", "noise_seed"],
        "calibration": "symmetric split conformal",
        "ood_score": "source-fitted Ledoit-Wolf Mahalanobis distance in PCA space",
        "partition_rows": {
            name: int(len(indices)) for name, indices in partitions.items()
        },
        "pca_explained_variance": feature_pipeline.explained_variance_ratio,
        "supplemental_analyses": {
            "paired_adaptation_effects": "adaptation_effect_intervals.csv",
            "prospective_precision_plan": "adaptation_precision_plan.csv",
            "reference_label_sensitivity": "reference_label_sensitivity.csv",
            "label_sensitivity_scope": (
                "formal errors only; not independent pipeline validation"
            ),
        },
    }
    manifest_path.write_text(json.dumps(experiment_manifest, indent=2), encoding="utf-8")

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(
        _render_summary(
            config,
            bootstrap,
            calibration_rows,
            adaptation_metrics,
            adaptation_effects,
            precision_plan,
            label_sensitivity,
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "split_manifest": split_manifest_path,
        "clean_predictions": clean_predictions_path,
        "clean_metrics": clean_metrics_path,
        "clean_intervals": clean_intervals_path,
        "shift_predictions": shifted_predictions_path,
        "shift_metrics": shifted_metrics_path,
        "calibration": calibration_metrics_path,
        "bootstrap": bootstrap_path,
        "shift_metric_intervals": shifted_intervals_path,
        "reference_label_sensitivity": label_sensitivity_path,
        "adaptation_predictions": adaptation_predictions_path,
        "adaptation": adaptation_metrics_path,
        "adaptation_intervals": adaptation_intervals_path,
        "adaptation_effects": adaptation_effects_path,
        "adaptation_precision_plan": precision_plan_path,
        "target_recalibration": target_calibration_path,
        "label_budget": label_budget_path,
        "subgroups": subgroup_path,
        "subgroup_support": subgroup_support_path,
        "risk_coverage": risk_path,
        "ablation": ablation_path,
        "representation_ablation": representation_path,
        "physical_plausibility": plausibility_path,
        "calibration_plot": calibration_plot,
        "risk_plot": risk_plot,
        "label_budget_plot": label_budget_plot,
        "ablation_plot": ablation_plot,
        "representation_plot": representation_plot,
    }


def _slice_targets(
    targets: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[indices] for name, values in targets.items()}


def _metadata(
    data: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    selected_manifest: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    rows = selected_rows[indices]
    frame = selected_manifest.iloc[indices][
        ["targetid", "source_file", "sn_r", "teff", "logg", "feh", "split"]
    ].reset_index(drop=True)
    frame = frame.rename(columns={"targetid": "object_id"})
    assert np.array_equal(frame["object_id"].to_numpy(), data["targetid"][rows])
    return frame


def _prediction_frame(
    metadata: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    model_name: str,
    evaluation: str,
    seed: int,
    noise_factor: float,
    ood_score: np.ndarray,
    ood_percentile: np.ndarray,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for target in TARGET_NAMES:
        frame = metadata.copy()
        frame["evaluation"] = evaluation
        frame["model"] = model_name
        frame["target"] = target
        frame["seed"] = seed
        frame["noise_factor"] = noise_factor
        frame["y_true"] = frame[target].to_numpy(dtype=float)
        frame["y_pred"] = np.asarray(predictions[target], dtype=float)
        frame["residual"] = frame["y_pred"] - frame["y_true"]
        frame["abs_error"] = frame["residual"].abs()
        frame["ood_score"] = ood_score
        frame["ood_percentile"] = ood_percentile
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _evaluate_clean(
    data: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    selected_manifest: pd.DataFrame,
    partitions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    features: np.ndarray,
    models: dict[str, object],
    ood_scorer: MahalanobisOODScorer,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    saved: dict[tuple[str, str, str], np.ndarray] = {}
    for partition_name in ("source_holdout", "target_adaptation", "target_evaluation"):
        indices = partitions[partition_name]
        metadata = _metadata(data, selected_rows, selected_manifest, indices)
        scores = ood_scorer.score(features[indices])
        percentiles = ood_scorer.percentile(features[indices])
        for model_name, model in models.items():
            predictions = model.predict(features[indices])
            prediction_frames.append(
                _prediction_frame(
                    metadata,
                    predictions,
                    model_name,
                    f"{partition_name}_clean",
                    seed=-1,
                    noise_factor=1.0,
                    ood_score=scores,
                    ood_percentile=percentiles,
                )
            )
            for target in TARGET_NAMES:
                saved[(partition_name, model_name, target)] = predictions[target]
                metric_rows.append(
                    {
                        "evaluation": f"{partition_name}_clean",
                        "model": model_name,
                        "target": target,
                        **summary_metrics(targets[target][indices], predictions[target]),
                    }
                )
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(metric_rows),
        saved,
    )


def _calibrators_from_clean_rows(
    clean_rows: pd.DataFrame, alphas: list[float]
) -> dict[tuple[str, str, float], SplitConformalCalibrator]:
    calibrators: dict[tuple[str, str, float], SplitConformalCalibrator] = {}
    calibration = clean_rows.loc[
        clean_rows["evaluation"] == "source_holdout_clean"
    ]
    for (model, target), group in calibration.groupby(["model", "target"]):
        for alpha in alphas:
            calibrators[(str(model), str(target), float(alpha))] = (
                SplitConformalCalibrator(alpha=float(alpha)).fit(
                    group["y_true"].to_numpy(), group["y_pred"].to_numpy()
                )
            )
    return calibrators


def _evaluate_clean_target_calibration(
    clean_rows: pd.DataFrame,
    calibrators: dict[tuple[str, str, float], SplitConformalCalibrator],
    alphas: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    evaluation = clean_rows.loc[
        clean_rows["evaluation"] == "target_evaluation_clean"
    ]
    for (model, target), group in evaluation.groupby(["model", "target"]):
        for alpha in alphas:
            calibrator = calibrators[(str(model), str(target), float(alpha))]
            lower, upper = calibrator.predict_interval(group["y_pred"])
            rows.append(
                {
                    "evaluation": "target_evaluation_clean",
                    "calibration_source": "source_holdout_clean",
                    "model": model,
                    "target": target,
                    "seed": -1,
                    "noise_factor": 1.0,
                    "alpha": float(alpha),
                    "nominal_coverage": 1 - float(alpha),
                    "calibration_size": calibrator.calibration_size_,
                    "interval_radius": calibrator.radius_,
                    **interval_metrics(group["y_true"], lower, upper),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_shifts(
    config: DesiReliabilityConfig,
    data: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    selected_manifest: pd.DataFrame,
    target_eval: np.ndarray,
    targets: dict[str, np.ndarray],
    models: dict[str, object],
    feature_pipeline: SpectralFeaturePipeline,
    ood_scorer: MahalanobisOODScorer,
    calibrators: dict[tuple[str, str, float], SplitConformalCalibrator],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    metadata = _metadata(data, selected_rows, selected_manifest, target_eval)
    true = _slice_targets(targets, target_eval)

    for seed in range(
        config.noise_seed_start,
        config.noise_seed_start + config.noise_seed_count,
    ):
        for factor in config.noise_factors:
            features = _shift_features(
                config,
                data,
                selected_rows,
                target_eval,
                float(factor),
                seed,
                feature_pipeline,
            )
            scores = ood_scorer.score(features)
            percentiles = ood_scorer.percentile(features)
            for model_name, model in models.items():
                predictions = model.predict(features)
                frame = _prediction_frame(
                    metadata,
                    predictions,
                    model_name,
                    "target_evaluation_shifted",
                    seed,
                    float(factor),
                    scores,
                    percentiles,
                )
                frame["pi90_lower"] = np.nan
                frame["pi90_upper"] = np.nan
                prediction_frames.append(frame)
                for target in TARGET_NAMES:
                    metrics = summary_metrics(true[target], predictions[target])
                    metric_rows.append(
                        {
                            "evaluation": "target_evaluation_shifted",
                            "model": model_name,
                            "target": target,
                            "seed": seed,
                            "noise_factor": float(factor),
                            "mean_ood_score": float(np.mean(scores)),
                            "mean_ood_percentile": float(np.mean(percentiles)),
                            **metrics,
                        }
                    )
                    for alpha in config.conformal_alphas:
                        calibrator = calibrators[
                            (model_name, target, float(alpha))
                        ]
                        lower, upper = calibrator.predict_interval(
                            predictions[target]
                        )
                        calibration_rows.append(
                            {
                                "evaluation": "target_evaluation_shifted",
                                "calibration_source": "source_holdout_clean",
                                "model": model_name,
                                "target": target,
                                "seed": seed,
                                "noise_factor": float(factor),
                                "alpha": float(alpha),
                                "nominal_coverage": 1 - float(alpha),
                                "calibration_size": calibrator.calibration_size_,
                                "interval_radius": calibrator.radius_,
                                **interval_metrics(true[target], lower, upper),
                            }
                        )
                        if np.isclose(alpha, 0.10):
                            target_mask = frame["target"] == target
                            frame.loc[target_mask, "pi90_lower"] = lower
                            frame.loc[target_mask, "pi90_upper"] = upper
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(calibration_rows),
    )


def _bootstrap_shift_effects(
    config: DesiReliabilityConfig,
    clean_rows: pd.DataFrame,
    shifted_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clean_eval = clean_rows.loc[
        clean_rows["evaluation"] == "target_evaluation_clean"
    ]
    for (model, target, factor), group in shifted_rows.groupby(
        ["model", "target", "noise_factor"]
    ):
        clean = clean_eval.loc[
            (clean_eval["model"] == model) & (clean_eval["target"] == target)
        ].set_index("object_id")["abs_error"]
        effect = two_axis_mae_bootstrap(
            group[["seed", "object_id", "abs_error"]],
            clean,
            n_bootstrap=config.bootstrap_replicates,
            confidence=config.bootstrap_confidence,
            random_state=config.random_state + int(round(float(factor) * 100)),
        )
        rows.append(
            {"model": model, "target": target, "noise_factor": factor, **effect}
        )
    return pd.DataFrame(rows)


def _two_axis_metric_intervals(
    config: DesiReliabilityConfig, shifted_rows: pd.DataFrame
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (model, target, factor), group in shifted_rows.groupby(
        ["model", "target", "noise_factor"]
    ):
        interval = two_axis_metric_bootstrap(
            group[["seed", "object_id", "y_true", "y_pred"]],
            n_bootstrap=config.bootstrap_replicates,
            confidence=config.bootstrap_confidence,
            random_state=(
                config.random_state
                + int(round(float(factor) * 100))
                + sum(ord(character) for character in f"{model}:{target}")
            ),
        )
        interval.insert(0, "noise_factor", factor)
        interval.insert(0, "target", target)
        interval.insert(0, "model", model)
        frames.append(interval)
    return pd.concat(frames, ignore_index=True)


def _cluster_metric_intervals(
    frame: pd.DataFrame,
    grouping: list[str],
    config: DesiReliabilityConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for keys, group in frame.groupby(grouping):
        keys = keys if isinstance(keys, tuple) else (keys,)
        interval = cluster_bootstrap_metrics(
            group["y_true"],
            group["y_pred"],
            groups=group["object_id"],
            n_bootstrap=config.bootstrap_replicates,
            confidence=config.bootstrap_confidence,
            random_state=(
                config.random_state
                + sum(ord(character) for character in ":".join(map(str, keys)))
            ),
        )
        for position, (name, value) in enumerate(zip(grouping, keys)):
            interval.insert(position, name, value)
        frames.append(interval)
    return pd.concat(frames, ignore_index=True)


def _shift_features(
    config: DesiReliabilityConfig,
    data: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    indices: np.ndarray,
    factor: float,
    seed: int,
    feature_pipeline: SpectralFeaturePipeline,
) -> np.ndarray:
    raw_rows = selected_rows[indices]
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
    return feature_pipeline.transform(normalized, valid)[0]


def _evaluate_adaptation_ladder(
    config: DesiReliabilityConfig,
    data: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    selected_manifest: pd.DataFrame,
    partitions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    clean_features: np.ndarray,
    source_model: object,
    augmented_model: object | None,
    feature_pipeline: SpectralFeaturePipeline,
    source_calibrators: dict[tuple[str, str, float], SplitConformalCalibrator],
    ood_scorer: MahalanobisOODScorer,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_train = partitions["source_train"]
    source_holdout = partitions["source_holdout"]
    target_adaptation = partitions["target_adaptation"]
    target_evaluation = partitions["target_evaluation"]
    factor = float(config.adaptation_noise_factor)
    seed = int(config.adaptation_noise_seed)
    shifted_adaptation = _shift_features(
        config,
        data,
        selected_rows,
        target_adaptation,
        factor,
        seed,
        feature_pipeline,
    )
    shifted_evaluation = _shift_features(
        config,
        data,
        selected_rows,
        target_evaluation,
        factor,
        seed,
        feature_pipeline,
    )

    coral = CORALAdapter(config.coral_regularization).fit(
        clean_features[source_train], shifted_adaptation
    )
    coral_model = _new_model(config).fit(
        coral.transform_source(clean_features[source_train]),
        _slice_targets(targets, source_train),
    )
    full_target_weight = len(source_train) / len(target_adaptation)
    retrained_model = _new_model(config).fit(
        np.vstack(
            [clean_features[source_train], shifted_adaptation]
        ),
        {
            target: np.concatenate(
                [targets[target][source_train], targets[target][target_adaptation]]
            )
            for target in TARGET_NAMES
        },
        sample_weight=np.concatenate(
            [
                np.ones(len(source_train)),
                np.full(len(target_adaptation), full_target_weight),
            ]
        ),
    )

    method_specs: list[tuple[str, str, object, np.ndarray]] = [
        ("source_only", "source_only", source_model, clean_features[source_holdout]),
    ]
    if augmented_model is not None:
        method_specs.append(
            (
                "noise_augmented",
                "source_only_synthetic_augmentation",
                augmented_model,
                clean_features[source_holdout],
            )
        )
    method_specs.extend(
        [
            (
                "coral_unlabeled",
                "unlabeled_target_features",
                coral_model,
                coral.transform_source(clean_features[source_holdout]),
            ),
            (
                "source_plus_target_retrained",
                "labeled_target",
                retrained_model,
                clean_features[source_holdout],
            ),
        ]
    )

    metadata = _metadata(
        data, selected_rows, selected_manifest, target_evaluation
    )
    scores = ood_scorer.score(shifted_evaluation)
    percentiles = ood_scorer.percentile(shifted_evaluation)
    true_evaluation = _slice_targets(targets, target_evaluation)
    true_holdout = _slice_targets(targets, source_holdout)
    frames: list[pd.DataFrame] = []
    predictions_by_method: dict[str, dict[str, np.ndarray]] = {}
    metrics_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []

    for method, access, model, calibration_features in method_specs:
        predictions = model.predict(shifted_evaluation)
        predictions_by_method[method] = predictions
        frame = _prediction_frame(
            metadata,
            predictions,
            method,
            "target_evaluation_shifted",
            seed,
            factor,
            scores,
            percentiles,
        ).rename(columns={"model": "method"})
        frame["target_access"] = access
        frames.append(frame)

        calibration_predictions = model.predict(calibration_features)
        for target in TARGET_NAMES:
            current = predictions[target]
            baseline = predictions_by_method["source_only"][target]
            current_metrics = summary_metrics(true_evaluation[target], current)
            baseline_mae = summary_metrics(
                true_evaluation[target], baseline
            )["mae"]
            metrics_rows.append(
                {
                    "evaluation": "target_evaluation_shifted",
                    "method": method,
                    "target_access": access,
                    "target": target,
                    "noise_factor": factor,
                    "seed": seed,
                    **current_metrics,
                    "mae_change_vs_source_percent": 100
                    * (current_metrics["mae"] / baseline_mae - 1),
                    "wilcoxon_p_vs_source": paired_significance(
                        current - true_evaluation[target],
                        baseline - true_evaluation[target],
                    ),
                }
            )
            for alpha in config.conformal_alphas:
                method_calibrator = SplitConformalCalibrator(float(alpha)).fit(
                    true_holdout[target], calibration_predictions[target]
                )
                lower, upper = method_calibrator.predict_interval(current)
                calibration_rows.append(
                    {
                        "evaluation": "target_evaluation_shifted",
                        "method": method,
                        "target_access": access,
                        "calibration_source": "source_holdout_method_specific",
                        "target": target,
                        "noise_factor": factor,
                        "seed": seed,
                        "alpha": float(alpha),
                        "nominal_coverage": 1 - float(alpha),
                        "calibration_size": method_calibrator.calibration_size_,
                        "interval_radius": method_calibrator.radius_,
                        **interval_metrics(true_evaluation[target], lower, upper),
                    }
                )

    # Target-domain recalibration is a separate, explicitly labeled use of
    # target labels.  The calibration stars remain disjoint from evaluation.
    for method, access, model, _ in method_specs:
        if method not in {"source_only", "noise_augmented", "coral_unlabeled"}:
            continue
        adaptation_predictions = model.predict(shifted_adaptation)
        evaluation_predictions = predictions_by_method[method]
        for target in TARGET_NAMES:
            for alpha in config.conformal_alphas:
                target_calibrator = SplitConformalCalibrator(float(alpha)).fit(
                    targets[target][target_adaptation],
                    adaptation_predictions[target],
                )
                lower, upper = target_calibrator.predict_interval(
                    evaluation_predictions[target]
                )
                calibration_rows.append(
                    {
                        "evaluation": "target_evaluation_shifted",
                        "method": method,
                        "target_access": "labeled_target_calibration",
                        "calibration_source": "target_adaptation_shifted",
                        "target": target,
                        "noise_factor": factor,
                        "seed": seed,
                        "alpha": float(alpha),
                        "nominal_coverage": 1 - float(alpha),
                        "calibration_size": target_calibrator.calibration_size_,
                        "interval_radius": target_calibrator.radius_,
                        **interval_metrics(true_evaluation[target], lower, upper),
                    }
                )

    label_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(config.random_state + 1701)
    budgets = [0] + sorted(
        {
            min(int(budget), len(target_adaptation))
            for budget in config.label_budgets
            if int(budget) > 0
        }
    )
    for budget in budgets:
        repeats = config.label_budget_repeats if budget > 0 else 1
        for repeat in range(repeats):
            if budget == 0:
                budget_model = source_model
            else:
                chosen = rng.choice(
                    target_adaptation, size=budget, replace=False
                )
                chosen_positions = np.searchsorted(target_adaptation, chosen)
                target_weight = len(source_train) / budget
                budget_model = _new_model(config).fit(
                    np.vstack(
                        [clean_features[source_train], shifted_adaptation[chosen_positions]]
                    ),
                    {
                        target: np.concatenate(
                            [targets[target][source_train], targets[target][chosen]]
                        )
                        for target in TARGET_NAMES
                    },
                    sample_weight=np.concatenate(
                        [
                            np.ones(len(source_train)),
                            np.full(budget, target_weight),
                        ]
                    ),
                )
            budget_predictions = budget_model.predict(shifted_evaluation)
            for target in TARGET_NAMES:
                current = budget_predictions[target]
                baseline = predictions_by_method["source_only"][target]
                current_metrics = summary_metrics(true_evaluation[target], current)
                baseline_mae = summary_metrics(
                    true_evaluation[target], baseline
                )["mae"]
                label_rows.append(
                    {
                        "evaluation": "target_evaluation_shifted",
                        "method": "source_plus_target_retrained",
                        "target_access": "labeled_target" if budget else "source_only",
                        "budget": budget,
                        "repeat": repeat,
                        "target": target,
                        "noise_factor": factor,
                        "seed": seed,
                        **current_metrics,
                        "mae_change_vs_source_percent": 100
                        * (current_metrics["mae"] / baseline_mae - 1),
                        "wilcoxon_p_vs_source": paired_significance(
                            current - true_evaluation[target],
                            baseline - true_evaluation[target],
                        ),
                    }
                )
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(metrics_rows),
        pd.DataFrame(calibration_rows),
        pd.DataFrame(label_rows),
    )


def _subgroup_metrics(
    config: DesiReliabilityConfig, shifted_rows: pd.DataFrame
) -> pd.DataFrame:
    audited = shifted_rows.copy()
    audited["temperature_regime"] = pd.cut(
        audited["teff"],
        [-np.inf, 4000, 6000, np.inf],
        labels=["cool_<4000", "mid_4000_6000", "hot_>6000"],
    )
    audited["gravity_regime"] = np.where(
        audited["logg"] < 3.5, "giant_<3.5", "dwarf_>=3.5"
    )
    audited["metallicity_regime"] = pd.cut(
        audited["feh"],
        [-np.inf, -1.5, -0.5, np.inf],
        labels=["metal_poor_<-1.5", "intermediate", "metal_rich_>=-0.5"],
    )
    audited["snr_regime"] = pd.cut(
        audited["sn_r"],
        [-np.inf, 15, 30, np.inf],
        labels=["snr_<15", "snr_15_30", "snr_>=30"],
    )
    rows: list[dict[str, object]] = []
    for keys, experiment in audited.groupby(
        ["model", "target", "noise_factor", "seed"]
    ):
        model, target, factor, seed = keys
        for column in (
            "temperature_regime",
            "gravity_regime",
            "metallicity_regime",
            "snr_regime",
        ):
            for group_name, group in experiment.groupby(column, observed=True):
                if len(group) < config.subgroup_minimum:
                    continue
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "noise_factor": factor,
                        "seed": seed,
                        "subgroup_dimension": column,
                        "subgroup": str(group_name),
                        **summary_metrics(group["y_true"], group["y_pred"]),
                    }
                )
    return pd.DataFrame(rows)


def _subgroup_support(
    selected_manifest: pd.DataFrame,
    target_evaluation: np.ndarray,
    config: DesiReliabilityConfig,
) -> pd.DataFrame:
    """Record sparse groups even when metric reporting would be misleading."""

    frame = selected_manifest.iloc[target_evaluation].copy()
    frame["temperature_regime"] = pd.cut(
        frame["teff"],
        [-np.inf, 4000, 6000, np.inf],
        labels=["cool_<4000", "mid_4000_6000", "hot_>6000"],
    )
    frame["gravity_regime"] = np.where(
        frame["logg"] < 3.5, "giant_<3.5", "dwarf_>=3.5"
    )
    frame["metallicity_regime"] = pd.cut(
        frame["feh"],
        [-np.inf, -1.5, -0.5, np.inf],
        labels=["metal_poor_<-1.5", "intermediate", "metal_rich_>=-0.5"],
    )
    frame["snr_regime"] = pd.cut(
        frame["sn_r"],
        [-np.inf, 15, 30, np.inf],
        labels=["snr_<15", "snr_15_30", "snr_>=30"],
    )
    rows: list[dict[str, object]] = []
    for dimension in (
        "temperature_regime",
        "gravity_regime",
        "metallicity_regime",
        "snr_regime",
    ):
        for subgroup, group in frame.groupby(dimension, observed=True):
            rows.append(
                {
                    "subgroup_dimension": dimension,
                    "subgroup": str(subgroup),
                    "n": int(len(group)),
                    "metric_status": (
                        "reported"
                        if len(group) >= config.subgroup_minimum
                        else "insufficient_support"
                    ),
                    "minimum_required": config.subgroup_minimum,
                }
            )
    return pd.DataFrame(rows)


def _selective_prediction(
    config: DesiReliabilityConfig, shifted_rows: pd.DataFrame
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (model, target, factor, seed), group in shifted_rows.groupby(
        ["model", "target", "noise_factor", "seed"]
    ):
        curve = risk_coverage_curve(
            group["y_true"],
            group["y_pred"],
            group["ood_score"],
            config.ood_coverages,
        )
        curve.insert(0, "noise_factor", factor)
        curve.insert(0, "target", target)
        curve.insert(0, "model", model)
        curve.insert(3, "seed", seed)
        rows.append(curve)
    return pd.concat(rows, ignore_index=True)


def _model_ablation(
    config: DesiReliabilityConfig,
    data: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    partitions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    clean_features: np.ndarray,
    feature_pipeline: SpectralFeaturePipeline,
) -> pd.DataFrame:
    source_train = partitions["source_train"]
    source_holdout = partitions["source_holdout"]
    target_evaluation = partitions["target_evaluation"]
    shifted_evaluation = _shift_features(
        config,
        data,
        selected_rows,
        target_evaluation,
        float(config.ablation_noise_factor),
        int(config.adaptation_noise_seed),
        feature_pipeline,
    )
    evaluations = {
        "source_holdout_clean": (clean_features[source_holdout], source_holdout),
        "target_evaluation_clean": (
            clean_features[target_evaluation],
            target_evaluation,
        ),
        "target_evaluation_shifted": (shifted_evaluation, target_evaluation),
    }
    rows: list[dict[str, object]] = []
    for family in config.ablation_models:
        model = SeparateFamilyRegressor(
            family=family,
            target_names=TARGET_NAMES,
            random_state=config.random_state,
            n_estimators=config.ablation_estimators,
        ).fit(
            clean_features[source_train],
            _slice_targets(targets, source_train),
        )
        family_metrics: dict[tuple[str, str], dict[str, float | int]] = {}
        for evaluation, (features, indices) in evaluations.items():
            predictions = model.predict(features)
            for target in TARGET_NAMES:
                metrics = summary_metrics(targets[target][indices], predictions[target])
                family_metrics[(evaluation, target)] = metrics
                clean_target_mae = family_metrics.get(
                    ("target_evaluation_clean", target), {}
                ).get("mae", np.nan)
                rows.append(
                    {
                        "family": family,
                        "representation": f"pca_{config.pca_components}",
                        "evaluation": evaluation,
                        "target": target,
                        "noise_factor": (
                            float(config.ablation_noise_factor)
                            if evaluation.endswith("shifted")
                            else 1.0
                        ),
                        **metrics,
                        "mae_change_vs_clean_target_percent": (
                            100 * (metrics["mae"] / clean_target_mae - 1)
                            if np.isfinite(clean_target_mae)
                            else np.nan
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    for family in frame["family"].unique():
        for target in TARGET_NAMES:
            clean_mask = (
                (frame["family"] == family)
                & (frame["target"] == target)
                & (frame["evaluation"] == "target_evaluation_clean")
            )
            shifted_mask = (
                (frame["family"] == family)
                & (frame["target"] == target)
                & (frame["evaluation"] == "target_evaluation_shifted")
            )
            clean_mae = float(frame.loc[clean_mask, "mae"].iloc[0])
            frame.loc[shifted_mask, "mae_change_vs_clean_target_percent"] = (
                100 * (frame.loc[shifted_mask, "mae"] / clean_mae - 1)
            )
    return frame


def _representation_ablation(
    config: DesiReliabilityConfig,
    data: dict[str, np.ndarray],
    selected_rows: np.ndarray,
    partitions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    normalized_clean: np.ndarray,
    valid_clean: np.ndarray,
) -> pd.DataFrame:
    """Measure whether the robustness conclusion depends on PCA capacity."""

    source_train = partitions["source_train"]
    target_evaluation = partitions["target_evaluation"]
    rows: list[dict[str, object]] = []
    for components in sorted(set(config.representation_components)):
        pipeline = SpectralFeaturePipeline(
            wavelength_min=config.wavelength_min,
            wavelength_max=config.wavelength_max,
            outlier_percentiles=tuple(config.outlier_percentiles),
            pca_components=int(components),
            random_state=config.random_state,
        ).fit(
            normalized_clean[source_train],
            valid_clean[source_train],
            data["wavelength"],
        )
        clean_features, _ = pipeline.transform(normalized_clean, valid_clean)
        shifted_features = _shift_features(
            config,
            data,
            selected_rows,
            target_evaluation,
            float(config.ablation_noise_factor),
            int(config.adaptation_noise_seed),
            pipeline,
        )
        model = SeparateFamilyRegressor(
            family="extra_trees",
            target_names=TARGET_NAMES,
            random_state=config.random_state,
            n_estimators=config.ablation_estimators,
        ).fit(
            clean_features[source_train],
            _slice_targets(targets, source_train),
        )
        for evaluation, features in (
            ("target_evaluation_clean", clean_features[target_evaluation]),
            ("target_evaluation_shifted", shifted_features),
        ):
            predictions = model.predict(features)
            for target in TARGET_NAMES:
                rows.append(
                    {
                        "family": "extra_trees",
                        "requested_components": int(components),
                        "fitted_components": int(pipeline.pca_.n_components_),
                        "explained_variance": pipeline.explained_variance_ratio,
                        "evaluation": evaluation,
                        "target": target,
                        "noise_factor": (
                            float(config.ablation_noise_factor)
                            if evaluation.endswith("shifted")
                            else 1.0
                        ),
                        **summary_metrics(
                            targets[target][target_evaluation], predictions[target]
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    clean = frame.loc[
        frame["evaluation"] == "target_evaluation_clean",
        ["requested_components", "target", "mae"],
    ].rename(columns={"mae": "clean_mae"})
    frame = frame.merge(clean, on=["requested_components", "target"], how="left")
    frame["mae_change_vs_clean_target_percent"] = 100 * (
        frame["mae"] / frame["clean_mae"] - 1
    )
    return frame


def _physical_plausibility(
    config: DesiReliabilityConfig,
    adaptation_rows: pd.DataFrame,
    targets: dict[str, np.ndarray],
    source_train: np.ndarray,
    output_path: Path,
) -> None:
    pivot = adaptation_rows.pivot_table(
        index=["method", "object_id"], columns="target", values="y_pred"
    ).reset_index()
    hard_bounds = {
        "teff": (2500.0, 50000.0),
        "logg": (-1.0, 6.5),
        "feh": (-6.0, 1.5),
    }
    benchmark_bounds = {
        "teff": tuple(config.teff_range),
        "logg": tuple(config.logg_range),
        "feh": tuple(config.feh_range),
    }
    hard_flag = np.zeros(len(pivot), dtype=bool)
    benchmark_flag = np.zeros(len(pivot), dtype=bool)
    for target in TARGET_NAMES:
        hard_lower, hard_upper = hard_bounds[target]
        domain_lower, domain_upper = benchmark_bounds[target]
        hard_flag |= ~pivot[target].between(hard_lower, hard_upper).to_numpy()
        benchmark_flag |= ~pivot[target].between(
            domain_lower, domain_upper
        ).to_numpy()
    base_payload: dict[str, object] = {
        "hard_physical_bounds": hard_bounds,
        "benchmark_domain_bounds": benchmark_bounds,
        "method_summary": [
            {
                "method": str(method),
                "n": int(len(group)),
                "hard_bound_violation_fraction": float(
                    hard_flag[group.index.to_numpy(dtype=int)].mean()
                ),
                "benchmark_domain_exit_fraction": float(
                    benchmark_flag[group.index.to_numpy(dtype=int)].mean()
                ),
            }
            for method, group in pivot.groupby("method")
        ],
    }
    if config.isochrone_grid_csv is None:
        base_payload.update(
            {
                "status": "bounds_evaluated_isochrone_not_evaluated",
                "isochrone_reason": (
                    "No cited isochrone grid was configured; empirical rarity is "
                    "not mislabeled as physical impossibility."
                ),
                "isochrone_required_columns": ["teff", "logg", "feh"],
            }
        )
        output_path.write_text(
            json.dumps(base_payload, indent=2),
            encoding="utf-8",
        )
        return
    grid_frame = pd.read_csv(config.isochrone_grid_csv)
    required = ["teff", "logg", "feh"]
    missing = set(required) - set(grid_frame)
    if missing:
        raise ValueError(f"isochrone grid is missing columns: {sorted(missing)}")
    reference = np.column_stack([targets[name][source_train] for name in TARGET_NAMES])
    checker = IsochroneManifold(config.isochrone_threshold_quantile).fit(
        grid_frame[required].to_numpy(), reference
    )
    distances, flagged = checker.score(pivot[required].to_numpy())
    base_payload.update({
        "status": "bounds_and_isochrone_evaluated",
        "grid_path": config.isochrone_grid_csv,
        "threshold_quantile": config.isochrone_threshold_quantile,
        "isochrone_method_summary": [
            {
                "method": str(method),
                "n": int(len(group)),
                "flagged_fraction": float(flagged[group.index].mean()),
                "mean_distance": float(distances[group.index].mean()),
            }
            for method, group in pivot.groupby("method")
        ],
    })
    output_path.write_text(json.dumps(base_payload, indent=2), encoding="utf-8")


def _plot_calibration(frame: pd.DataFrame, output_path: Path) -> None:
    selected = frame.loc[
        (frame["model"] == "source_only") & np.isclose(frame["alpha"], 0.10)
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for target, group in selected.groupby("target"):
        summary = group.groupby("noise_factor")["coverage"].agg(["mean", "std"])
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"].fillna(0),
            marker="o",
            capsize=3,
            label=str(target).upper(),
        )
    axis.axhline(0.9, color="black", linestyle="--", linewidth=1, label="Nominal 90%")
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Noise standard-deviation multiplier")
    axis.set_ylabel("Empirical interval coverage")
    axis.set_title("Source-calibrated uncertainty under S/N shift")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_risk(frame: pd.DataFrame, output_path: Path) -> None:
    selected = frame.loc[
        (frame["model"] == "source_only") & np.isclose(frame["noise_factor"], 2.0)
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, target in zip(axes, TARGET_NAMES):
        group = (
            selected.loc[selected["target"] == target]
            .groupby("retained_fraction", as_index=False)["mae"]
            .agg(["mean", "std"])
            .reset_index()
        )
        axis.errorbar(
            group["retained_fraction"],
            group["mean"],
            yerr=group["std"].fillna(0),
            marker="o",
            capsize=3,
        )
        axis.set_title(target.upper())
        axis.set_xlabel("Retained fraction")
        axis.set_ylabel("MAE")
        axis.grid(alpha=0.25)
    figure.suptitle("Selective prediction using source-fitted OOD scores at 2x noise")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_label_budget(frame: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, target in zip(axes, TARGET_NAMES):
        group = frame.loc[frame["target"] == target]
        summary = group.groupby("budget")["mae"].agg(["mean", "std"])
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"].fillna(0),
            marker="o",
            capsize=3,
        )
        axis.set_title(target.upper())
        axis.set_xlabel("Labeled target stars")
        axis.set_ylabel("MAE")
        axis.grid(alpha=0.25)
    figure.suptitle("Supervised adaptation label-budget curve")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_ablation(frame: pd.DataFrame, output_path: Path) -> None:
    pivot = frame.loc[frame["evaluation"] == "target_evaluation_shifted"].pivot(
        index="target", columns="family", values="mae"
    )
    axis = pivot.plot(kind="bar", figsize=(9, 5))
    axis.set_ylabel("MAE")
    axis.set_title("Model-family sensitivity at the declared shift")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(axis.figure)


def _plot_representation(frame: pd.DataFrame, output_path: Path) -> None:
    shifted = frame.loc[frame["evaluation"] == "target_evaluation_shifted"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, target in zip(axes, TARGET_NAMES):
        group = shifted.loc[shifted["target"] == target].sort_values(
            "requested_components"
        )
        axis.plot(group["requested_components"], group["mae"], marker="o")
        axis.set_title(target.upper())
        axis.set_xlabel("PCA components")
        axis.set_ylabel("Shifted MAE")
        axis.grid(alpha=0.25)
    figure.suptitle("Representation-capacity sensitivity at the declared shift")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _render_summary(
    config: DesiReliabilityConfig,
    bootstrap: pd.DataFrame,
    calibration: pd.DataFrame,
    adaptation: pd.DataFrame,
    adaptation_effects: pd.DataFrame,
    precision_plan: pd.DataFrame,
    label_sensitivity: pd.DataFrame,
) -> str:
    factor_two = bootstrap.loc[
        (bootstrap["model"] == "source_only")
        & (bootstrap["target"] == "teff")
        & np.isclose(bootstrap["noise_factor"], 2.0)
    ].iloc[0]
    coverage = calibration.loc[
        (calibration["model"] == "source_only")
        & (calibration["target"] == "teff")
        & np.isclose(calibration["noise_factor"], 2.0)
        & np.isclose(calibration["alpha"], 0.10)
    ]["coverage"].mean()
    clean_coverage = calibration.loc[
        (calibration["model"] == "source_only")
        & (calibration["target"] == "teff")
        & np.isclose(calibration["noise_factor"], 1.0)
        & np.isclose(calibration["alpha"], 0.10)
    ]["coverage"].mean()
    maximum_factor = max(config.noise_factors)
    maximum_coverage = calibration.loc[
        (calibration["model"] == "source_only")
        & (calibration["target"] == "teff")
        & np.isclose(calibration["noise_factor"], maximum_factor)
        & np.isclose(calibration["alpha"], 0.10)
    ]["coverage"].mean()
    methods = adaptation.loc[
        (adaptation["evaluation"] == "target_evaluation_shifted")
        & (adaptation["target"] == "teff")
    ][["method", "mae", "mae_change_vs_source_percent", "wilcoxon_p_vs_source"]]
    effect_teff = adaptation_effects.loc[
        adaptation_effects["target"] == "teff",
        [
            "method",
            "paired_mae_difference",
            "paired_difference_ci_lower",
            "paired_difference_ci_upper",
            "conclusion",
        ],
    ]
    coral_required_n = int(
        precision_plan.loc[
            (precision_plan["target"] == "teff")
            & (precision_plan["method"] == "coral_unlabeled"),
            "required_n_for_two_sided_detection",
        ].iloc[0]
    )
    label_teff = label_sensitivity.loc[
        label_sensitivity["target"] == "teff"
    ].iloc[0]
    return (
        "=== StellarShift-Bench v1.1 DESI reliability audit ===\n"
        f"2x-noise TEFF MAE: {factor_two['mae']:.2f} K "
        f"({config.bootstrap_confidence:.0%} nested-bootstrap CI "
        f"{factor_two['mae_ci_lower']:.2f} to {factor_two['mae_ci_upper']:.2f}).\n"
        f"Change from clean: {factor_two['mae_change_percent']:.1f}% "
        f"(CI {factor_two['change_ci_lower']:.1f}% to "
        f"{factor_two['change_ci_upper']:.1f}%).\n"
        "Source-calibrated nominal 90% TEFF coverage: "
        f"clean {clean_coverage:.3f}; 2x noise {coverage:.3f}; "
        f"{maximum_factor:g}x noise {maximum_coverage:.3f}.\n\n"
        "Adaptation ladder at the fixed 2x target shift:\n"
        + methods.to_string(index=False)
        + "\n\nPaired TEFF MAE effects (negative is better):\n"
        + effect_teff.to_string(index=False)
        + f"\n\nProspective CORAL target-evaluation requirement for a 5% effect: "
        f"{coral_required_n} stars.\n"
        + "Formal-label sensitivity TEFF change: "
        f"{label_teff['sensitivity_change_percent_median']:+.1f}% "
        f"(CI {label_teff['sensitivity_change_ci_lower']:+.1f}% to "
        f"{label_teff['sensitivity_change_ci_upper']:+.1f}%).\n\n"
        "Caveat: the executed shift remains controlled Gaussian S/N degradation "
        "on DESI R-arm spectra. Cross-arm and cross-survey conclusions require "
        "new data runs.\n"
    )
