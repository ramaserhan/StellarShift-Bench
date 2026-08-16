import numpy as np

from stellar_benchmark.adaptation import retrain_with_target_data
from stellar_benchmark.models import BaselineRegressor


def test_retraining_returns_new_model_and_preserves_source_model():
    rng = np.random.default_rng(5)
    X_source = rng.normal(size=(50, 3))
    Y_source = np.column_stack([X_source[:, 0], X_source[:, 1]])
    X_target = rng.normal(loc=0.5, size=(20, 3))
    Y_target = np.column_stack([X_target[:, 0] + 0.2, X_target[:, 1]])

    source_model = BaselineRegressor(n_bootstrap=2, random_state=5).fit(
        X_source, Y_source
    )
    before = source_model.predict(X_target[:5])
    adapted = retrain_with_target_data(
        source_model,
        X_target,
        Y_target,
        strategy="source_plus_target",
        X_source=X_source,
        Y_source=Y_source,
    )

    assert adapted is not source_model
    assert np.allclose(source_model.predict(X_target[:5]), before)
    assert adapted.predict(X_target[:5]).shape == before.shape
