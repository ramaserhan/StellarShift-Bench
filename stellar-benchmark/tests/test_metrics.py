import numpy as np

from stellar_benchmark.eval import metrics as M


def test_bias_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert M.bias(y, y) == 0.0


def test_bias_sign():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = y_true + 1.0
    assert M.bias(y_true, y_pred) == 1.0


def test_scatter_zero_for_constant_offset():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true + 5.0  # constant offset, no spread in residuals
    assert M.scatter(y_true, y_pred) == 0.0


def test_r_squared_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.r_squared(y, y) == 1.0


def test_mae_and_rmse():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert M.mean_absolute_error(y_true, y_pred) == 3.5
    assert np.isclose(M.root_mean_squared_error(y_true, y_pred), np.sqrt(12.5))


def test_calibration_coverage_range():
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 1, 5000)
    y_pred = np.zeros_like(y_true)
    y_pred_std = np.ones_like(y_true)
    cov = M.calibration_coverage(y_true, y_pred, y_pred_std, z=1.0)
    # For a well-calibrated Gaussian, ~68% should fall within 1 sigma.
    assert 0.6 < cov < 0.75


def test_paired_significance_requires_matched_length():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    try:
        M.paired_significance(a, b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_paired_significance_identical_errors():
    residuals = np.array([1.0, -2.0, 3.0])
    assert M.paired_significance(residuals, residuals) == 1.0
