"""Baseline stellar-parameter regressors with a shared interface.

The literature this project is based on found that simpler models (e.g.
pretrained MLPs) are often competitive with much more complex architectures
in cross-survey transfer settings. So the baselines here are intentionally
simple gradient-boosted trees and small MLPs -- the point of the benchmark is
a fair, controlled comparison, not maximizing in-domain accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _make_base_model(model_type: str, random_state: int):
    if model_type == "gbm":
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state
        )
    if model_type == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=2000,
                random_state=random_state,
            ),
        )
    raise ValueError(f"Unknown model_type: {model_type!r} (expected 'gbm' or 'mlp')")


class BaselineRegressor:
    """Multi-target regressor with a bootstrap uncertainty proxy.

    fit(X, Y) / predict(X) -> point estimates
    ``predict_with_uncertainty`` returns ensemble mean and spread. The spread
    measures model instability across bootstrap resamples; it is not a complete
    estimate of aleatoric plus epistemic predictive uncertainty.
    """

    def __init__(self, model_type: str = "gbm", n_bootstrap: int = 10, random_state: int = 42):
        if model_type not in {"gbm", "mlp"}:
            raise ValueError("model_type must be 'gbm' or 'mlp'")
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be positive")
        self.model_type = model_type
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self._models: list[MultiOutputRegressor] = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "BaselineRegressor":
        X, Y = _validate_training_arrays(X, Y)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        self._models = []
        for i in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)  # bootstrap resample
            base = _make_base_model(self.model_type, self.random_state + i)
            model = MultiOutputRegressor(base)
            model.fit(X[idx], Y[idx])
            self._models.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_prediction_input(X)
        preds = np.stack([m.predict(X) for m in self._models], axis=0)
        return preds.mean(axis=0)

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = self._validate_prediction_input(X)
        preds = np.stack([m.predict(X) for m in self._models], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def clone_untrained(self) -> "BaselineRegressor":
        return BaselineRegressor(self.model_type, self.n_bootstrap, self.random_state)

    def _validate_prediction_input(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("model must be fitted before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        expected_features = self._models[0].n_features_in_
        if X.shape[1] != expected_features:
            raise ValueError(f"X has {X.shape[1]} features; expected {expected_features}")
        return X


def _validate_training_arrays(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2 or Y.shape[1] == 0:
        raise ValueError("Y must be a one- or two-dimensional target array")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of rows")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("X and Y must contain only finite values")
    return X, Y
