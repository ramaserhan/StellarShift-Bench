import numpy as np
import pandas as pd

from stellar_benchmark.eval.reliability import (
    IsochroneManifold,
    MahalanobisOODScorer,
    SplitConformalCalibrator,
    cluster_bootstrap_metrics,
    interval_metrics,
    risk_coverage_curve,
    two_sample_mae_shift_bootstrap,
    two_axis_mae_bootstrap,
    two_axis_metric_bootstrap,
)


def test_split_conformal_intervals_and_coverage_metrics():
    truth = np.arange(20, dtype=float)
    prediction = truth + np.linspace(-1, 1, 20)
    calibrator = SplitConformalCalibrator(alpha=0.1).fit(truth, prediction)
    lower, upper = calibrator.predict_interval(prediction)
    result = interval_metrics(truth, lower, upper)
    assert result["coverage"] >= 0.9
    assert result["mean_width"] > 0


def test_ood_score_and_selective_risk_curve():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(80, 4))
    test = np.vstack([rng.normal(size=(20, 4)), np.full((3, 4), 8.0)])
    scorer = MahalanobisOODScorer().fit(train)
    scores = scorer.score(test)
    assert scores[-3:].min() > scores[:-3].max()
    truth = np.zeros(len(test))
    prediction = scores.copy()
    curve = risk_coverage_curve(truth, prediction, scores, coverages=(1.0, 0.8))
    assert curve.loc[curve.requested_coverage == 0.8, "mae"].iloc[0] < curve.loc[
        curve.requested_coverage == 1.0, "mae"
    ].iloc[0]


def test_cluster_and_two_axis_bootstrap_are_deterministic():
    truth = np.arange(12, dtype=float)
    prediction = truth + np.tile([-1.0, 1.0], 6)
    first = cluster_bootstrap_metrics(
        truth, prediction, groups=np.arange(12), n_bootstrap=30, random_state=8
    )
    second = cluster_bootstrap_metrics(
        truth, prediction, groups=np.arange(12), n_bootstrap=30, random_state=8
    )
    pd.testing.assert_frame_equal(first, second)

    rows = pd.DataFrame(
        [
            {"seed": seed, "object_id": star, "abs_error": 1 + 0.1 * seed}
            for seed in range(3)
            for star in range(8)
        ]
    )
    clean = pd.Series(0.8, index=np.arange(8))
    effect = two_axis_mae_bootstrap(rows, clean, n_bootstrap=30, random_state=5)
    assert effect["mae_change_percent"] > 0
    assert effect["object_count"] == 8


def test_isochrone_plausibility_requires_explicit_reference_grid():
    grid = np.column_stack(
        [np.linspace(3500, 6500, 30), np.linspace(1, 5, 30), np.linspace(-2, 0, 30)]
    )
    reference = grid + np.array([10.0, 0.01, 0.01])
    checker = IsochroneManifold(threshold_quantile=0.95).fit(grid, reference)
    distances, flagged = checker.score(
        np.array([[4500, 2.3, -1.3], [20000, -8, 5]], dtype=float)
    )
    assert distances[1] > distances[0]
    assert flagged[1]


def test_two_axis_all_metric_bootstrap_is_deterministic():
    rows = pd.DataFrame(
        {
            "seed": np.repeat([1, 2], 4),
            "object_id": np.tile(np.arange(4), 2),
            "y_true": np.tile([0.0, 1.0, 2.0, 3.0], 2),
            "y_pred": [0.1, 0.9, 2.2, 2.8, 0.2, 1.0, 2.1, 2.7],
        }
    )
    first = two_axis_metric_bootstrap(rows, n_bootstrap=20, random_state=5)
    second = two_axis_metric_bootstrap(rows, n_bootstrap=20, random_state=5)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == {"mae", "rmse", "bias", "scatter", "r2"}


def test_two_sample_mae_shift_bootstrap_resamples_both_domains():
    source_truth = np.zeros(80)
    source_prediction = np.linspace(0.5, 1.5, 80)
    target_truth = np.zeros(120)
    target_prediction = np.linspace(1.5, 2.5, 120)
    first = two_sample_mae_shift_bootstrap(
        source_truth,
        source_prediction,
        target_truth,
        target_prediction,
        n_bootstrap=300,
        random_state=17,
    )
    second = two_sample_mae_shift_bootstrap(
        source_truth,
        source_prediction,
        target_truth,
        target_prediction,
        n_bootstrap=300,
        random_state=17,
    )
    assert first == second
    assert first["source_holdout_n"] == 80
    assert first["target_evaluation_n"] == 120
    assert first["mae_difference_ci_lower"] > 0
    assert first["cross_survey_change_ci_lower_percent"] > 0
