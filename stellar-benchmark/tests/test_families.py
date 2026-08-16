import numpy as np

from stellar_benchmark.models.families import SeparateFamilyRegressor


def test_all_model_families_fit_and_predict_separate_targets():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 8))
    targets = {
        "teff": 5000 + 100 * X[:, 0],
        "logg": 4 + 0.2 * X[:, 1],
        "feh": -0.5 + 0.1 * X[:, 2],
    }
    for family in ("ridge", "extra_trees", "mlp"):
        model = SeparateFamilyRegressor(
            family, random_state=4, n_estimators=10
        ).fit(X, targets)
        predictions = model.predict(X[:7])
        assert set(predictions) == set(targets)
        assert all(values.shape == (7,) for values in predictions.values())
