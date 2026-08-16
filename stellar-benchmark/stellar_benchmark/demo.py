"""Deterministic end-to-end synthetic benchmark used as a smoke experiment."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptation import retrain_with_target_data
from .config import SyntheticDemoConfig
from .data.preprocessing import add_snr_bins, train_test_split_by_group
from .eval.harness import EvalHarness
from .models.baseline import BaselineRegressor


TARGET_COLUMNS = ["teff", "logg", "feh"]
FEATURE_COLUMNS = [f"feat_{index}" for index in range(20)]


def make_synthetic_survey(
    n: int,
    feature_shift: float,
    label_offsets: tuple[float, float, float],
    seed: int,
    survey_name: str,
) -> pd.DataFrame:
    """Create two surveys sharing physics but differing in measurement domain."""

    rng = np.random.default_rng(seed)
    teff_true = rng.uniform(4000, 7000, n)
    logg_true = rng.uniform(0.5, 5.0, n)
    feh_true = rng.uniform(-2.5, 0.5, n)
    snr = rng.uniform(15, 250, n)

    t = (teff_true - 5500) / 1500
    g = (logg_true - 2.75) / 2.25
    m = (feh_true + 1.0) / 1.5
    noise_scale = 0.02 + 1.5 / snr

    features = []
    for index in range(len(FEATURE_COLUMNS)):
        a = np.sin(index + 1) * 0.7
        b = np.cos((index + 1) * 0.5) * 0.6
        c = np.sin((index + 1) * 0.3) * 0.8
        nonlinear = 0.20 * np.sin((index + 1) * t) + 0.10 * g * m
        domain_pattern = feature_shift * np.cos(index * 0.4)
        features.append(
            a * t + b * g + c * m + nonlinear + domain_pattern
            + rng.normal(0, noise_scale, n)
        )

    df = pd.DataFrame(np.column_stack(features), columns=FEATURE_COLUMNS)
    df["source_id"] = [f"{survey_name}-{index:06d}" for index in range(n)]
    df["teff"] = teff_true + label_offsets[0] + rng.normal(0, 25, n)
    df["logg"] = logg_true + label_offsets[1] + rng.normal(0, 0.04, n)
    df["feh"] = feh_true + label_offsets[2] + rng.normal(0, 0.04, n)
    df["snr"] = snr
    df["stellar_type"] = pd.cut(
        teff_true,
        bins=[0, 4500, 5500, 6500, 10000],
        labels=["K", "G", "F", "A"],
    )
    return add_snr_bins(df)


def run_synthetic_demo(config: SyntheticDemoConfig) -> dict[str, Path]:
    """Run and persist in-domain, zero-shot, and retrained evaluations."""

    source = make_synthetic_survey(
        config.source_rows,
        feature_shift=0.0,
        label_offsets=(0.0, 0.0, 0.0),
        seed=config.random_state,
        survey_name="source",
    )
    target = make_synthetic_survey(
        config.target_rows,
        feature_shift=0.18,
        label_offsets=(80.0, 0.08, 0.10),
        seed=config.random_state + 1,
        survey_name="target",
    )

    source_train, source_eval = train_test_split_by_group(
        source, test_fraction=0.20, random_state=config.random_state
    )
    target_adapt, target_eval = train_test_split_by_group(
        target,
        test_fraction=1 - config.target_adaptation_fraction,
        random_state=config.random_state + 2,
    )

    X_source = source_train[FEATURE_COLUMNS].to_numpy()
    Y_source = source_train[TARGET_COLUMNS].to_numpy()
    model = BaselineRegressor(
        model_type=config.model_type,
        n_bootstrap=config.n_bootstrap,
        random_state=config.random_state,
    ).fit(X_source, Y_source)

    harness = EvalHarness(TARGET_COLUMNS, strata=["snr_bin", "stellar_type"])
    source_metrics, _, _ = _evaluate_partition(
        model, harness, source_eval, label="source_holdout"
    )
    zero_shot_metrics, zero_shot_pred, _ = _evaluate_partition(
        model, harness, target_eval, label="target_zero_shot"
    )

    adapted = retrain_with_target_data(
        model,
        target_adapt[FEATURE_COLUMNS].to_numpy(),
        target_adapt[TARGET_COLUMNS].to_numpy(),
        strategy="source_plus_target",
        X_source=X_source,
        Y_source=Y_source,
    )
    adapted_metrics, adapted_pred, _ = _evaluate_partition(
        adapted, harness, target_eval, label="target_retrained"
    )
    y_true_target = {
        target_name: target_eval[target_name].to_numpy()
        for target_name in TARGET_COLUMNS
    }
    comparison = harness.compare_matched(
        y_true_target, zero_shot_pred, adapted_pred
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "source_metrics": output_dir / "source_holdout_metrics.csv",
        "zero_shot_metrics": output_dir / "target_zero_shot_metrics.csv",
        "adapted_metrics": output_dir / "target_retrained_metrics.csv",
        "comparison": output_dir / "matched_comparison.csv",
        "manifest": output_dir / "manifest.json",
        "summary": output_dir / "summary.txt",
    }
    source_metrics.to_csv(artifacts["source_metrics"], index=False)
    zero_shot_metrics.to_csv(artifacts["zero_shot_metrics"], index=False)
    adapted_metrics.to_csv(artifacts["adapted_metrics"], index=False)
    comparison.to_csv(artifacts["comparison"], index=False)

    manifest = {
        "experiment": asdict(config),
        "targets": TARGET_COLUMNS,
        "features": FEATURE_COLUMNS,
        "split_unit": "source_id",
        "partition_rows": {
            "source_train": len(source_train),
            "source_eval": len(source_eval),
            "target_adaptation": len(target_adapt),
            "target_eval": len(target_eval),
        },
        "uncertainty_note": (
            "Reported coverage uses bootstrap ensemble spread as an uncertainty "
            "proxy, not a complete predictive uncertainty estimate."
        ),
    }
    artifacts["manifest"].write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    summary = "\n\n".join(
        [
            "=== Source-domain held-out performance ===\n"
            + harness.report(source_metrics),
            "=== Target-domain zero-shot performance ===\n"
            + harness.report(zero_shot_metrics),
            "=== Target-domain performance after source+target retraining ===\n"
            + harness.report(adapted_metrics),
            "=== Matched zero-shot vs. retrained comparison ===\n"
            + comparison.to_string(index=False),
        ]
    )
    artifacts["summary"].write_text(summary + "\n", encoding="utf-8")
    return artifacts


def _evaluate_partition(
    model: BaselineRegressor,
    harness: EvalHarness,
    df: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    predictions, spread = model.predict_with_uncertainty(
        df[FEATURE_COLUMNS].to_numpy()
    )
    y_true = {name: df[name].to_numpy() for name in TARGET_COLUMNS}
    y_pred = {
        name: predictions[:, index] for index, name in enumerate(TARGET_COLUMNS)
    }
    y_spread = {
        name: spread[:, index] for index, name in enumerate(TARGET_COLUMNS)
    }
    metrics = harness.evaluate(df, y_true, y_pred, y_spread, label=label)
    return metrics, y_pred, y_spread
