"""Leakage-safe optical-spectrum transfer between two real surveys."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .adaptation.coral import CORALAdapter
from .config import CrossSurveyConfig
from .data.cross_survey import (
    TARGET_NAMES,
    SurveySpectra,
    harmonize_spectra,
    load_survey_npz,
    shared_log_wavelength_grid,
)
from .data.desi import continuum_normalize
from .eval.metrics import paired_significance, summary_metrics
from .eval.precision import paired_adaptation_effects
from .eval.publication import cross_survey_publication_gate
from .eval.reference_labels import formal_label_error_transfer_sensitivity
from .eval.reliability import (
    IsochroneManifold,
    MahalanobisOODScorer,
    SplitConformalCalibrator,
    cluster_bootstrap_metrics,
    interval_metrics,
    risk_coverage_curve,
    two_sample_mae_shift_bootstrap,
)
from .models.families import SeparateFamilyRegressor
from .models.spectral import SeparateExtraTreesRegressor, SpectralFeaturePipeline


def run_cross_survey_experiment(config: CrossSurveyConfig) -> dict[str, Path]:
    """Run source→target transfer with explicit target-access accounting."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_survey_npz(config.source_npz, config.source_survey)
    target = load_survey_npz(config.target_npz, config.target_survey)
    selections, split_manifest, overlap_removed = _build_splits(
        source, target, config
    )
    split_manifest_path = output_dir / "split_manifest.csv"
    split_manifest.to_csv(split_manifest_path, index=False)

    common_wavelength = shared_log_wavelength_grid(
        config.wavelength_min,
        config.wavelength_max,
        config.log_wavelength_step,
    )
    source_rows = selections["source_selected"]
    target_rows = selections["target_selected"]
    source_flux, source_valid = harmonize_spectra(
        source.wavelength,
        source.flux[source_rows],
        source.valid[source_rows],
        common_wavelength,
        config.source_resolving_power,
        config.common_resolving_power,
    )
    target_flux, target_valid = harmonize_spectra(
        target.wavelength,
        target.flux[target_rows],
        target.valid[target_rows],
        common_wavelength,
        config.target_resolving_power,
        config.common_resolving_power,
    )
    source_flux, source_valid = continuum_normalize(
        source_flux,
        source_valid,
        config.continuum_window,
        config.continuum_polyorder,
    )
    target_flux, target_valid = continuum_normalize(
        target_flux,
        target_valid,
        config.continuum_window,
        config.continuum_polyorder,
    )

    source_train = selections["source_train_local"]
    source_holdout = selections["source_holdout_local"]
    target_adaptation = selections["target_adaptation_local"]
    target_evaluation = selections["target_evaluation_local"]
    source_targets = {
        name: np.asarray(source.targets[name][source_rows], dtype=float)
        for name in TARGET_NAMES
    }
    target_targets = {
        name: np.asarray(target.targets[name][target_rows], dtype=float)
        for name in TARGET_NAMES
    }
    publication_gate = cross_survey_publication_gate(
        source_train_targets=_slice_targets(source_targets, source_train),
        source_holdout_count=len(source_holdout),
        target_adaptation_count=len(target_adaptation),
        target_evaluation_targets=_slice_targets(target_targets, target_evaluation),
        source_label_scale=config.source_label_scale,
        target_label_scale=config.target_label_scale,
        minimum_source_train=config.publication_min_source_train,
        minimum_source_holdout=config.publication_min_source_holdout,
        minimum_target_adaptation=config.publication_min_target_adaptation,
        minimum_target_evaluation=config.publication_min_target_evaluation,
        minimum_giants=config.publication_min_giants,
        minimum_metal_poor=config.publication_min_metal_poor,
    )
    publication_gate_path = output_dir / "publication_gate.json"
    publication_gate_path.write_text(
        json.dumps(publication_gate, indent=2), encoding="utf-8"
    )
    if config.enforce_publication_gate and not publication_gate["passed"]:
        blockers = ", ".join(publication_gate["blockers"])
        raise ValueError(
            "cross-survey quality gate failed; no scientific run started. "
            f"Blockers: {blockers}. See {publication_gate_path}."
        )
    feature_pipeline = SpectralFeaturePipeline(
        wavelength_min=config.wavelength_min,
        wavelength_max=config.wavelength_max,
        outlier_percentiles=tuple(config.outlier_percentiles),
        pca_components=config.pca_components,
        random_state=config.random_state,
    ).fit(
        source_flux[source_train], source_valid[source_train], common_wavelength
    )
    source_features, _ = feature_pipeline.transform(source_flux, source_valid)
    target_features, _ = feature_pipeline.transform(target_flux, target_valid)
    source_model = _new_model(config).fit(
        source_features[source_train],
        _slice_targets(source_targets, source_train),
    )
    coral = CORALAdapter(config.coral_regularization).fit(
        source_features[source_train], target_features[target_adaptation]
    )
    coral_model = _new_model(config).fit(
        coral.transform_source(source_features[source_train]),
        _slice_targets(source_targets, source_train),
    )
    target_weight = len(source_train) / len(target_adaptation)
    retrained_model = _new_model(config).fit(
        np.vstack(
            [source_features[source_train], target_features[target_adaptation]]
        ),
        {
            name: np.concatenate(
                [
                    source_targets[name][source_train],
                    target_targets[name][target_adaptation],
                ]
            )
            for name in TARGET_NAMES
        },
        sample_weight=np.concatenate(
            [
                np.ones(len(source_train)),
                np.full(len(target_adaptation), target_weight),
            ]
        ),
    )
    method_specs = [
        (
            "source_only",
            "source_only",
            source_model,
            source_features[source_holdout],
        ),
        (
            "coral_unlabeled",
            "unlabeled_target_features",
            coral_model,
            coral.transform_source(source_features[source_holdout]),
        ),
        (
            "source_plus_target_retrained",
            "labeled_target",
            retrained_model,
            source_features[source_holdout],
        ),
    ]

    ood = MahalanobisOODScorer().fit(source_features[source_train])
    target_ood = ood.score(target_features[target_evaluation])
    target_percentile = ood.percentile(target_features[target_evaluation])
    target_metadata = pd.DataFrame(
        {
            "survey": config.target_survey,
            "object_id": target.object_id[target_rows][target_evaluation],
            "snr": target.snr[target_rows][target_evaluation],
        }
    )
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    method_predictions: dict[str, dict[str, np.ndarray]] = {}
    true_eval = _slice_targets(target_targets, target_evaluation)
    true_source_holdout = _slice_targets(source_targets, source_holdout)

    source_holdout_predictions = source_model.predict(
        source_features[source_holdout]
    )
    source_holdout_prediction_frames: list[pd.DataFrame] = []
    for name in TARGET_NAMES:
        source_holdout_prediction_frames.append(
            pd.DataFrame(
                {
                    "survey": config.source_survey,
                    "object_id": source.object_id[source_rows][source_holdout],
                    "evaluation": "source_holdout_in_domain",
                    "method": "source_only",
                    "target_access": "source_only",
                    "target": name,
                    "y_true": true_source_holdout[name],
                    "y_pred": source_holdout_predictions[name],
                    "residual": (
                        source_holdout_predictions[name]
                        - true_source_holdout[name]
                    ),
                    "abs_error": np.abs(
                        source_holdout_predictions[name]
                        - true_source_holdout[name]
                    ),
                }
            )
        )
        metric_rows.append(
            {
                "evaluation": "source_holdout_in_domain",
                "method": "source_only",
                "target_access": "source_only",
                "target": name,
                **summary_metrics(
                    true_source_holdout[name], source_holdout_predictions[name]
                ),
            }
        )

    for method, access, model, calibration_features in method_specs:
        predictions = model.predict(target_features[target_evaluation])
        method_predictions[method] = predictions
        calibration_predictions = model.predict(calibration_features)
        prediction_frames.append(
            _prediction_rows(
                target_metadata,
                true_eval,
                predictions,
                method,
                access,
                target_ood,
                target_percentile,
            )
        )
        for name in TARGET_NAMES:
            metrics = summary_metrics(true_eval[name], predictions[name])
            baseline = method_predictions["source_only"][name]
            baseline_mae = summary_metrics(true_eval[name], baseline)["mae"]
            metric_rows.append(
                {
                    "evaluation": "target_evaluation_cross_survey",
                    "method": method,
                    "target_access": access,
                    "target": name,
                    **metrics,
                    "mae_change_vs_source_percent": 100
                    * (metrics["mae"] / baseline_mae - 1),
                    "wilcoxon_p_vs_source": paired_significance(
                        predictions[name] - true_eval[name],
                        baseline - true_eval[name],
                    ),
                }
            )
            source_calibrator = SplitConformalCalibrator(
                config.conformal_alpha
            ).fit(true_source_holdout[name], calibration_predictions[name])
            lower, upper = source_calibrator.predict_interval(predictions[name])
            calibration_rows.append(
                {
                    "evaluation": "target_evaluation_cross_survey",
                    "method": method,
                    "target_access": access,
                    "calibration_source": "source_holdout_method_specific",
                    "target": name,
                    "alpha": config.conformal_alpha,
                    "nominal_coverage": 1 - config.conformal_alpha,
                    "calibration_size": source_calibrator.calibration_size_,
                    "interval_radius": source_calibrator.radius_,
                    **interval_metrics(true_eval[name], lower, upper),
                }
            )

    # Recalibration uses target labels but no target-evaluation labels.
    for method, access, model, _ in method_specs[:2]:
        adapt_predictions = model.predict(target_features[target_adaptation])
        for name in TARGET_NAMES:
            calibrator = SplitConformalCalibrator(config.conformal_alpha).fit(
                target_targets[name][target_adaptation], adapt_predictions[name]
            )
            lower, upper = calibrator.predict_interval(
                method_predictions[method][name]
            )
            calibration_rows.append(
                {
                    "evaluation": "target_evaluation_cross_survey",
                    "method": method,
                    "target_access": "labeled_target_calibration",
                    "calibration_source": "target_adaptation",
                    "target": name,
                    "alpha": config.conformal_alpha,
                    "nominal_coverage": 1 - config.conformal_alpha,
                    "calibration_size": calibrator.calibration_size_,
                    "interval_radius": calibrator.radius_,
                    **interval_metrics(true_eval[name], lower, upper),
                }
            )

    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    source_holdout_predictions_frame = pd.concat(
        source_holdout_prediction_frames, ignore_index=True
    )
    metrics_frame = pd.DataFrame(metric_rows)
    calibration_frame = pd.DataFrame(calibration_rows)
    prediction_path = output_dir / "per_star_predictions.csv"
    source_holdout_prediction_path = (
        output_dir / "source_holdout_per_star_predictions.csv"
    )
    metrics_path = output_dir / "transfer_metrics.csv"
    calibration_path = output_dir / "calibration_cross_survey.csv"
    predictions_frame.to_csv(prediction_path, index=False)
    source_holdout_predictions_frame.to_csv(
        source_holdout_prediction_path, index=False
    )
    metrics_frame.to_csv(metrics_path, index=False)
    calibration_frame.to_csv(calibration_path, index=False)

    domain_shift = _domain_shift_summary(metrics_frame)
    domain_shift_path = output_dir / "domain_shift_summary.csv"
    domain_shift.to_csv(domain_shift_path, index=False)
    domain_shift_intervals = _domain_shift_intervals(
        true_source_holdout,
        source_holdout_predictions,
        true_eval,
        method_predictions["source_only"],
        config,
    )
    domain_shift_intervals_path = output_dir / "domain_shift_intervals.csv"
    domain_shift_intervals.to_csv(domain_shift_intervals_path, index=False)
    support_overlap = _support_overlap_metrics(
        predictions_frame,
        _slice_targets(source_targets, source_train),
    )
    support_overlap_path = output_dir / "support_overlap_metrics.csv"
    support_overlap.to_csv(support_overlap_path, index=False)

    reference_label_sensitivity_path = output_dir / "reference_label_sensitivity.csv"
    if target.target_errors is not None:
        target_error_frame = pd.DataFrame(
            {
                "object_id": target.object_id[target_rows][target_evaluation],
                **{
                    f"{name}_err": target.target_errors[name][target_rows][
                        target_evaluation
                    ]
                    for name in TARGET_NAMES
                },
            }
        )
        reference_label_sensitivity = formal_label_error_transfer_sensitivity(
            predictions_frame,
            target_error_frame,
            replicates=max(1000, config.bootstrap_replicates),
            confidence=config.bootstrap_confidence,
            random_state=config.random_state + 1901,
        )
    else:
        reference_label_sensitivity = pd.DataFrame(
            [
                {
                    "status": "not_evaluated",
                    "reason": "target survey contract did not include formal label errors",
                }
            ]
        )
    reference_label_sensitivity.to_csv(reference_label_sensitivity_path, index=False)

    bootstrap = _bootstrap_predictions(predictions_frame, config)
    bootstrap_path = output_dir / "star_bootstrap_intervals.csv"
    bootstrap.to_csv(bootstrap_path, index=False)
    adaptation_effects, precision_plan = paired_adaptation_effects(
        predictions_frame,
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
    label_budget = _label_budget_trials(
        config,
        source_features,
        target_features,
        source_targets,
        target_targets,
        source_train,
        target_adaptation,
        target_evaluation,
        method_predictions["source_only"],
    )
    label_budget_path = output_dir / "label_budget_trials.csv"
    label_budget.to_csv(label_budget_path, index=False)
    risk = _risk_curves(predictions_frame, config)
    risk_path = output_dir / "risk_coverage.csv"
    risk.to_csv(risk_path, index=False)
    subgroup = _subgroups(predictions_frame, config)
    subgroup_path = output_dir / "subgroup_metrics.csv"
    subgroup.to_csv(subgroup_path, index=False)
    ablation = _model_ablation(
        config,
        source_features,
        target_features,
        source_targets,
        target_targets,
        source_train,
        source_holdout,
        target_evaluation,
    )
    ablation_path = output_dir / "model_ablation.csv"
    ablation.to_csv(ablation_path, index=False)
    plausibility_path = output_dir / "physical_plausibility.json"
    _physical_plausibility(
        config,
        predictions_frame,
        source_targets,
        source_train,
        plausibility_path,
    )

    transfer_plot = output_dir / "cross_survey_transfer.png"
    calibration_plot = output_dir / "calibration_cross_survey.png"
    label_budget_plot = output_dir / "label_budget.png"
    _plot_transfer(metrics_frame, transfer_plot)
    _plot_calibration(calibration_frame, calibration_plot)
    _plot_label_budget(label_budget, label_budget_plot)

    manifest = {
        "version": "1.2.3",
        "experiment": asdict(config),
        "shift_type": "real_cross_survey_optical_spectra",
        "source_sha256": _sha256(Path(config.source_npz)),
        "target_sha256": _sha256(Path(config.target_npz)),
        "source_rows": int(len(source_rows)),
        "target_rows": int(len(target_rows)),
        "partitions": {
            "source_train": int(len(source_train)),
            "source_holdout": int(len(source_holdout)),
            "target_adaptation": int(len(target_adaptation)),
            "target_evaluation": int(len(target_evaluation)),
        },
        "cross_survey_overlap_removed_from_source": int(overlap_removed),
        "shared_wavelength_pixels": int(len(common_wavelength)),
        "shared_wavelength_range_angstrom": [
            float(common_wavelength[0]),
            float(common_wavelength[-1]),
        ],
        "resolution_matching": (
            "Gaussian line-spread approximation on a constant-log-lambda grid; "
            "the pipeline refuses spectral sharpening."
        ),
        "target_access_taxonomy": {
            "source_only": "no target data during fitting",
            "unlabeled_target_features": "target-adaptation spectra, no labels",
            "labeled_target_calibration": "target-adaptation labels only for intervals",
            "labeled_target": "target-adaptation spectra and labels for fitting",
        },
        "pca_explained_variance": feature_pipeline.explained_variance_ratio,
        "publication_gate": publication_gate,
        "reference_label_errors": {
            "available": target.target_errors is not None,
            "assumption": "independent_unbiased_gaussian_formal_errors",
            "scope": "sensitivity_not_independent_validation",
        },
        "label_semantics": {
            "teff": "APOGEE DR12 calibrated TEFF",
            "logg": "APOGEE DR12 calibrated LOGG",
            "feh": (
                "legacy contract slot containing APOGEE DR12 calibrated "
                "PARAM_M_H global metallicity, not elemental FE_H"
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(
        _summary(config, metrics_frame, calibration_frame, bootstrap, manifest),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "split_manifest": split_manifest_path,
        "predictions": prediction_path,
        "source_holdout_predictions": source_holdout_prediction_path,
        "metrics": metrics_path,
        "domain_shift": domain_shift_path,
        "domain_shift_intervals": domain_shift_intervals_path,
        "support_overlap": support_overlap_path,
        "reference_label_sensitivity": reference_label_sensitivity_path,
        "calibration": calibration_path,
        "bootstrap": bootstrap_path,
        "adaptation_effects": adaptation_effects_path,
        "adaptation_precision_plan": precision_plan_path,
        "publication_gate": publication_gate_path,
        "label_budget": label_budget_path,
        "risk_coverage": risk_path,
        "subgroups": subgroup_path,
        "ablation": ablation_path,
        "physical_plausibility": plausibility_path,
        "transfer_plot": transfer_plot,
        "calibration_plot": calibration_plot,
        "label_budget_plot": label_budget_plot,
    }


def _domain_shift_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    source = metrics.loc[
        (metrics["evaluation"] == "source_holdout_in_domain")
        & (metrics["method"] == "source_only"),
        ["target", "n", "mae", "rmse", "bias", "r2"],
    ].rename(
        columns={
            "n": "source_holdout_n",
            "mae": "source_holdout_mae",
            "rmse": "source_holdout_rmse",
            "bias": "source_holdout_bias",
            "r2": "source_holdout_r2",
        }
    )
    target = metrics.loc[
        (metrics["evaluation"] == "target_evaluation_cross_survey")
        & (metrics["method"] == "source_only"),
        ["target", "n", "mae", "rmse", "bias", "r2"],
    ].rename(
        columns={
            "n": "target_evaluation_n",
            "mae": "target_evaluation_mae",
            "rmse": "target_evaluation_rmse",
            "bias": "target_evaluation_bias",
            "r2": "target_evaluation_r2",
        }
    )
    frame = source.merge(target, on="target", validate="one_to_one")
    frame["cross_survey_mae_change_percent"] = 100.0 * (
        frame["target_evaluation_mae"] / frame["source_holdout_mae"] - 1.0
    )
    frame["comparison"] = "source_holdout_to_disjoint_target_evaluation"
    return frame


def _domain_shift_intervals(
    source_truth: dict[str, np.ndarray],
    source_predictions: dict[str, np.ndarray],
    target_truth: dict[str, np.ndarray],
    target_predictions: dict[str, np.ndarray],
    config: CrossSurveyConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(TARGET_NAMES):
        rows.append(
            {
                "target": name,
                **two_sample_mae_shift_bootstrap(
                    source_truth[name],
                    source_predictions[name],
                    target_truth[name],
                    target_predictions[name],
                    n_bootstrap=max(1000, config.bootstrap_replicates),
                    confidence=config.bootstrap_confidence,
                    random_state=config.random_state + 1301 + index,
                ),
            }
        )
    return pd.DataFrame(rows)


def _support_overlap_metrics(
    predictions: pd.DataFrame,
    source_train_targets: dict[str, np.ndarray],
) -> pd.DataFrame:
    bounds = {
        name: (float(np.min(values)), float(np.max(values)))
        for name, values in source_train_targets.items()
    }
    truth = (
        predictions.loc[predictions["method"] == "source_only"]
        .pivot(index="object_id", columns="target", values="y_true")
        .sort_index()
    )
    supported = np.ones(len(truth), dtype=bool)
    for name, (lower, upper) in bounds.items():
        supported &= truth[name].between(lower, upper).to_numpy()
    supported_ids = set(truth.index[supported])
    rows: list[dict[str, object]] = []
    for (method, target), group in predictions.groupby(["method", "target"]):
        selected = group.loc[group["object_id"].isin(supported_ids)]
        rows.append(
            {
                "method": method,
                "target": target,
                "support_definition": "joint_source_train_rectangular_minmax",
                "supported_stars": len(selected),
                "full_target_stars": group["object_id"].nunique(),
                "retained_fraction": len(selected) / len(group),
                "source_teff_min": bounds["teff"][0],
                "source_teff_max": bounds["teff"][1],
                "source_logg_min": bounds["logg"][0],
                "source_logg_max": bounds["logg"][1],
                "source_feh_min": bounds["feh"][0],
                "source_feh_max": bounds["feh"][1],
                **summary_metrics(selected["y_true"], selected["y_pred"]),
            }
        )
    return pd.DataFrame(rows)


def _new_model(config: CrossSurveyConfig) -> SeparateExtraTreesRegressor:
    return SeparateExtraTreesRegressor(
        target_names=TARGET_NAMES,
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=config.random_state,
    )


def _slice_targets(
    targets: dict[str, np.ndarray], rows: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[rows] for name, values in targets.items()}


def _window_valid_fraction(
    dataset: SurveySpectra, wavelength_min: float, wavelength_max: float
) -> np.ndarray:
    window = (dataset.wavelength >= wavelength_min) & (
        dataset.wavelength <= wavelength_max
    )
    if not np.any(window):
        raise ValueError("analysis window does not overlap the survey wavelength grid")
    return dataset.valid[:, window].mean(axis=1)


def _quality_unique(
    dataset: SurveySpectra,
    minimum_valid: float,
    wavelength_min: float,
    wavelength_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_fraction = _window_valid_fraction(dataset, wavelength_min, wavelength_max)
    quality = (
        (valid_fraction >= minimum_valid)
        & np.isfinite(dataset.snr)
        & (dataset.snr > 0)
    )
    for values in dataset.targets.values():
        quality &= np.isfinite(values)
    reasons = np.full(len(quality), "quality_cut", dtype="U48")
    reasons[quality] = "selected"
    candidates = pd.DataFrame(
        {
            "row": np.arange(len(quality)),
            "object_id": dataset.object_id,
            "snr": dataset.snr,
            "quality": quality,
        }
    )
    kept = (
        candidates.loc[candidates["quality"]]
        .sort_values(["object_id", "snr"], ascending=[True, False])
        .drop_duplicates("object_id")["row"]
        .to_numpy(dtype=int)
    )
    selected = np.zeros(len(quality), dtype=bool)
    selected[kept] = True
    reasons[quality & ~selected] = "duplicate_lower_snr"
    return selected, reasons


def _split(
    rows: np.ndarray,
    teff: np.ndarray,
    test_fraction: float,
    random_state: int,
    population: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    bins = pd.qcut(teff[rows], q=min(5, max(2, len(rows) // 20)), duplicates="drop")
    codes = np.asarray(bins.codes)
    if population is None:
        strata = codes
    else:
        population = np.asarray(population).reshape(-1)
        if len(population) != len(teff):
            raise ValueError("population strata must align with target rows")
        labels = np.asarray(population[rows]).astype(str)
        strata = np.asarray([f"{label}|teff{code}" for label, code in zip(labels, codes)])
    _, counts = np.unique(strata, return_counts=True)
    stratify = strata if len(counts) > 1 and counts.min() >= 2 else None
    return train_test_split(
        rows,
        test_size=test_fraction,
        random_state=random_state,
        stratify=stratify,
    )


def _build_splits(
    source: SurveySpectra,
    target: SurveySpectra,
    config: CrossSurveyConfig,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, int]:
    source_selected, source_reason = _quality_unique(
        source,
        config.valid_fraction_min,
        config.wavelength_min,
        config.wavelength_max,
    )
    target_selected, target_reason = _quality_unique(
        target,
        config.valid_fraction_min,
        config.wavelength_min,
        config.wavelength_max,
    )
    source_valid_fraction = _window_valid_fraction(
        source, config.wavelength_min, config.wavelength_max
    )
    target_valid_fraction = _window_valid_fraction(
        target, config.wavelength_min, config.wavelength_max
    )
    target_ids = set(target.object_id[target_selected].tolist())
    overlap = source_selected & np.isin(source.object_id, list(target_ids))
    source_selected[overlap] = False
    source_reason[overlap] = "cross_survey_object_overlap"
    source_rows = np.flatnonzero(source_selected)
    target_rows = np.flatnonzero(target_selected)
    if len(source_rows) < 40 or len(target_rows) < 30:
        raise ValueError(
            "cross-survey benchmark needs at least 40 disjoint source and 30 target stars"
        )
    source_train_global, source_holdout_global = _split(
        source_rows,
        source.targets["teff"],
        config.source_holdout_fraction,
        config.random_state,
    )
    target_evaluation_global, target_adaptation_global = _split(
        target_rows,
        target.targets["teff"],
        config.target_adaptation_fraction,
        config.random_state + 1,
        population=np.where(
            target.targets["feh"] < -1.5,
            "metal_poor",
            np.where(target.targets["logg"] < 3.5, "giant", "non_giant"),
        ),
    )
    source_lookup = {row: position for position, row in enumerate(source_rows)}
    target_lookup = {row: position for position, row in enumerate(target_rows)}
    selections = {
        "source_selected": source_rows,
        "target_selected": target_rows,
        "source_train_local": np.sort(
            [source_lookup[row] for row in source_train_global]
        ),
        "source_holdout_local": np.sort(
            [source_lookup[row] for row in source_holdout_global]
        ),
        "target_adaptation_local": np.sort(
            [target_lookup[row] for row in target_adaptation_global]
        ),
        "target_evaluation_local": np.sort(
            [target_lookup[row] for row in target_evaluation_global]
        ),
    }
    source_split = np.full(len(source.object_id), "excluded", dtype="U32")
    target_split = np.full(len(target.object_id), "excluded", dtype="U32")
    source_split[source_train_global] = "source_train"
    source_split[source_holdout_global] = "source_holdout"
    target_split[target_adaptation_global] = "target_adaptation"
    target_split[target_evaluation_global] = "target_evaluation"
    manifest = pd.concat(
        [
            pd.DataFrame(
                {
                    "survey": config.source_survey,
                    "row_index": np.arange(len(source.object_id)),
                    "object_id": source.object_id,
                    "snr": source.snr,
                    "valid_fraction": source_valid_fraction,
                    "selected": source_selected,
                    "selection_reason": source_reason,
                    "split": source_split,
                }
            ),
            pd.DataFrame(
                {
                    "survey": config.target_survey,
                    "row_index": np.arange(len(target.object_id)),
                    "object_id": target.object_id,
                    "snr": target.snr,
                    "valid_fraction": target_valid_fraction,
                    "selected": target_selected,
                    "selection_reason": target_reason,
                    "split": target_split,
                }
            ),
        ],
        ignore_index=True,
    )
    return selections, manifest, int(overlap.sum())


def _prediction_rows(
    metadata: pd.DataFrame,
    truth: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    method: str,
    access: str,
    ood_score: np.ndarray,
    ood_percentile: np.ndarray,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name in TARGET_NAMES:
        frame = metadata.copy()
        frame["evaluation"] = "target_evaluation_cross_survey"
        frame["method"] = method
        frame["target_access"] = access
        frame["target"] = name
        frame["y_true"] = truth[name]
        frame["y_pred"] = predictions[name]
        frame["residual"] = frame["y_pred"] - frame["y_true"]
        frame["abs_error"] = frame["residual"].abs()
        frame["ood_score"] = ood_score
        frame["ood_percentile"] = ood_percentile
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _bootstrap_predictions(
    predictions: pd.DataFrame, config: CrossSurveyConfig
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (method, target), group in predictions.groupby(["method", "target"]):
        interval = cluster_bootstrap_metrics(
            group["y_true"],
            group["y_pred"],
            groups=group["object_id"],
            n_bootstrap=config.bootstrap_replicates,
            confidence=config.bootstrap_confidence,
            random_state=config.random_state,
        )
        interval.insert(0, "target", target)
        interval.insert(0, "method", method)
        rows.append(interval)
    return pd.concat(rows, ignore_index=True)


def _label_budget_trials(
    config: CrossSurveyConfig,
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_targets: dict[str, np.ndarray],
    target_targets: dict[str, np.ndarray],
    source_train: np.ndarray,
    target_adaptation: np.ndarray,
    target_evaluation: np.ndarray,
    baseline: dict[str, np.ndarray],
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state + 1701)
    budgets = [0] + sorted(
        {min(int(value), len(target_adaptation)) for value in config.label_budgets}
    )
    rows: list[dict[str, object]] = []
    for budget in budgets:
        repeats = config.label_budget_repeats if budget else 1
        for repeat in range(repeats):
            if budget == 0:
                model = _new_model(config).fit(
                    source_features[source_train],
                    _slice_targets(source_targets, source_train),
                )
            else:
                chosen = rng.choice(target_adaptation, budget, replace=False)
                weight = len(source_train) / budget
                model = _new_model(config).fit(
                    np.vstack([source_features[source_train], target_features[chosen]]),
                    {
                        name: np.concatenate(
                            [source_targets[name][source_train], target_targets[name][chosen]]
                        )
                        for name in TARGET_NAMES
                    },
                    sample_weight=np.concatenate(
                        [np.ones(len(source_train)), np.full(budget, weight)]
                    ),
                )
            current = model.predict(target_features[target_evaluation])
            for name in TARGET_NAMES:
                truth = target_targets[name][target_evaluation]
                metrics = summary_metrics(truth, current[name])
                baseline_mae = summary_metrics(truth, baseline[name])["mae"]
                rows.append(
                    {
                        "budget": budget,
                        "repeat": repeat,
                        "target": name,
                        "target_access": "labeled_target" if budget else "source_only",
                        **metrics,
                        "mae_change_vs_source_percent": 100
                        * (metrics["mae"] / baseline_mae - 1),
                        "wilcoxon_p_vs_source": paired_significance(
                            current[name] - truth, baseline[name] - truth
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _risk_curves(
    predictions: pd.DataFrame, config: CrossSurveyConfig
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (method, target), group in predictions.groupby(["method", "target"]):
        curve = risk_coverage_curve(
            group["y_true"],
            group["y_pred"],
            group["ood_score"],
            config.ood_coverages,
        )
        curve.insert(0, "target", target)
        curve.insert(0, "method", method)
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def _subgroups(
    predictions: pd.DataFrame, config: CrossSurveyConfig
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["snr_regime"] = pd.qcut(frame["snr"], 3, duplicates="drop")
    frame["target_value_regime"] = frame.groupby("target")["y_true"].transform(
        lambda values: pd.qcut(values, 3, labels=False, duplicates="drop")
    )
    rows: list[dict[str, object]] = []
    for (method, target), experiment in frame.groupby(["method", "target"]):
        for dimension in ("snr_regime", "target_value_regime"):
            for subgroup, group in experiment.groupby(dimension, observed=True):
                if len(group) < config.subgroup_minimum:
                    continue
                rows.append(
                    {
                        "method": method,
                        "target": target,
                        "subgroup_dimension": dimension,
                        "subgroup": str(subgroup),
                        **summary_metrics(group["y_true"], group["y_pred"]),
                    }
                )
    return pd.DataFrame(rows)


def _model_ablation(
    config: CrossSurveyConfig,
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_targets: dict[str, np.ndarray],
    target_targets: dict[str, np.ndarray],
    source_train: np.ndarray,
    source_holdout: np.ndarray,
    target_evaluation: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in config.ablation_models:
        model = SeparateFamilyRegressor(
            family,
            target_names=TARGET_NAMES,
            random_state=config.random_state,
            n_estimators=config.ablation_estimators,
        ).fit(source_features[source_train], _slice_targets(source_targets, source_train))
        for evaluation, features, truth, indices in (
            ("source_holdout_in_domain", source_features, source_targets, source_holdout),
            ("target_evaluation_cross_survey", target_features, target_targets, target_evaluation),
        ):
            predictions = model.predict(features[indices])
            for name in TARGET_NAMES:
                rows.append(
                    {
                        "family": family,
                        "evaluation": evaluation,
                        "target": name,
                        **summary_metrics(truth[name][indices], predictions[name]),
                    }
                )
    frame = pd.DataFrame(rows)
    source_mae = frame.loc[
        frame["evaluation"] == "source_holdout_in_domain",
        ["family", "target", "mae"],
    ].rename(columns={"mae": "source_holdout_mae"})
    frame = frame.merge(source_mae, on=["family", "target"], how="left")
    frame["mae_change_vs_source_domain_percent"] = 100 * (
        frame["mae"] / frame["source_holdout_mae"] - 1
    )
    return frame


def _physical_plausibility(
    config: CrossSurveyConfig,
    predictions: pd.DataFrame,
    source_targets: dict[str, np.ndarray],
    source_train: np.ndarray,
    output_path: Path,
) -> None:
    pivot = predictions.pivot(
        index=["method", "object_id"], columns="target", values="y_pred"
    ).reset_index()
    hard_bounds = {
        "teff": (2500.0, 50000.0),
        "logg": (-1.0, 6.5),
        "feh": (-6.0, 1.5),
    }
    violations = np.zeros(len(pivot), dtype=bool)
    for name, (lower, upper) in hard_bounds.items():
        violations |= ~pivot[name].between(lower, upper).to_numpy()
    payload: dict[str, object] = {
        "hard_physical_bounds": hard_bounds,
        "method_summary": [
            {
                "method": str(method),
                "n": int(len(group)),
                "hard_bound_violation_fraction": float(
                    violations[group.index.to_numpy(dtype=int)].mean()
                ),
            }
            for method, group in pivot.groupby("method")
        ],
    }
    if config.isochrone_grid_csv is None:
        payload.update(
            {
                "status": "bounds_evaluated_isochrone_not_evaluated",
                "isochrone_reason": "No cited isochrone grid was configured.",
                "isochrone_required_columns": list(TARGET_NAMES),
            }
        )
        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return
    grid = pd.read_csv(config.isochrone_grid_csv)
    missing = set(TARGET_NAMES) - set(grid)
    if missing:
        raise ValueError(f"isochrone grid is missing columns: {sorted(missing)}")
    reference = np.column_stack(
        [source_targets[name][source_train] for name in TARGET_NAMES]
    )
    checker = IsochroneManifold(config.isochrone_threshold_quantile).fit(
        grid[list(TARGET_NAMES)].to_numpy(), reference
    )
    distances, flagged = checker.score(pivot[list(TARGET_NAMES)].to_numpy())
    summaries = []
    for method, group in pivot.groupby("method"):
        positions = group.index.to_numpy(dtype=int)
        summaries.append(
            {
                "method": method,
                "n": int(len(group)),
                "flagged_fraction": float(flagged[positions].mean()),
                "mean_distance": float(distances[positions].mean()),
            }
        )
    payload.update(
        {
            "status": "bounds_and_isochrone_evaluated",
            "isochrone_method_summary": summaries,
        }
    )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_transfer(frame: pd.DataFrame, output_path: Path) -> None:
    selected = frame.loc[frame["evaluation"] == "target_evaluation_cross_survey"]
    pivot = selected.pivot(index="target", columns="method", values="mae")
    axis = pivot.plot(kind="bar", figsize=(9, 5))
    axis.set_ylabel("MAE")
    axis.set_title("Real cross-survey transfer and adaptation")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(axis.figure)


def _plot_calibration(frame: pd.DataFrame, output_path: Path) -> None:
    pivot = frame.pivot_table(
        index="target", columns=["method", "calibration_source"], values="coverage"
    )
    axis = pivot.plot(kind="bar", figsize=(11, 5))
    axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Empirical coverage")
    axis.set_title("Nominal 90% interval coverage across surveys")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(axis.figure)


def _plot_label_budget(frame: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, target in zip(axes, TARGET_NAMES):
        summary = (
            frame.loc[frame["target"] == target]
            .groupby("budget")["mae"]
            .agg(["mean", "std"])
        )
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
    figure.suptitle("Cross-survey target-label budget")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(
    config: CrossSurveyConfig,
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    bootstrap: pd.DataFrame,
    manifest: dict[str, object],
) -> str:
    transfer = metrics.loc[
        metrics["evaluation"] == "target_evaluation_cross_survey",
        ["method", "target", "mae", "r2", "mae_change_vs_source_percent"],
    ]
    coverage = calibration[
        ["method", "calibration_source", "target", "coverage", "mean_width"]
    ]
    mae_intervals = bootstrap.loc[
        bootstrap["metric"] == "mae",
        ["method", "target", "estimate", "ci_lower", "ci_upper"],
    ]
    return (
        f"=== {config.source_survey} to {config.target_survey} cross-survey audit ===\n"
        f"Partitions: {manifest['partitions']}\n"
        f"Shared optical grid: {manifest['shared_wavelength_pixels']} pixels; "
        f"R={config.common_resolving_power:.0f}.\n\n"
        "Transfer metrics:\n"
        + transfer.to_string(index=False)
        + "\n\nCalibration:\n"
        + coverage.to_string(index=False)
        + "\n\nStar-bootstrap MAE intervals:\n"
        + mae_intervals.to_string(index=False)
        + "\n"
    )
