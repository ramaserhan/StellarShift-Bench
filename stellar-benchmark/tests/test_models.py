import numpy as np

from stellar_benchmark.models import BaselineRegressor


def test_baseline_fit_predict_and_uncertainty_shapes():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 4))
    Y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]])
    model = BaselineRegressor(model_type="gbm", n_bootstrap=2, random_state=3)
    model.fit(X, Y)

    predictions, spread = model.predict_with_uncertainty(X[:7])
    assert predictions.shape == (7, 2)
    assert spread.shape == (7, 2)
    assert np.all(spread >= 0)


def test_predict_before_fit_has_clear_error():
    model = BaselineRegressor(n_bootstrap=2)
    try:
        model.predict(np.ones((2, 3)))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
