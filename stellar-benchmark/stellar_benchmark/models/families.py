"""Small, defensible model-family ablations for spectral regression."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from .spectral import SeparateExtraTreesRegressor


class SeparateFamilyRegressor:
    """One estimator per target for ridge, ExtraTrees, or a small MLP."""

    def __init__(
        self,
        family: str,
        target_names: tuple[str, ...] = ("teff", "logg", "feh"),
        random_state: int = 42,
        n_estimators: int = 400,
    ) -> None:
        if family not in {"ridge", "extra_trees", "mlp"}:
            raise ValueError("family must be 'ridge', 'extra_trees', or 'mlp'")
        self.family = family
        self.target_names = target_names
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.models_: dict[str, object] = {}
        self.extra_trees_: SeparateExtraTreesRegressor | None = None

    def fit(
        self, X: np.ndarray, targets: dict[str, np.ndarray]
    ) -> "SeparateFamilyRegressor":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a non-empty finite 2D array")
        if set(targets) != set(self.target_names):
            raise ValueError("targets must exactly match target_names")
        if self.family == "extra_trees":
            self.extra_trees_ = SeparateExtraTreesRegressor(
                target_names=self.target_names,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            ).fit(X, targets)
            return self
        self.models_ = {}
        for offset, name in enumerate(self.target_names):
            values = np.asarray(targets[name], dtype=float).reshape(-1)
            if len(values) != len(X) or not np.isfinite(values).all():
                raise ValueError(f"target {name!r} must be finite and row-aligned")
            if self.family == "ridge":
                estimator = TransformedTargetRegressor(
                    regressor=make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                    transformer=StandardScaler(),
                )
            else:
                estimator = TransformedTargetRegressor(
                    regressor=make_pipeline(
                        StandardScaler(),
                        MLPRegressor(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            early_stopping=True,
                            validation_fraction=0.15,
                            max_iter=800,
                            random_state=self.random_state + offset,
                        ),
                    ),
                    transformer=StandardScaler(),
                )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                estimator.fit(X, values)
            self.models_[name] = estimator
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D array")
        if self.family == "extra_trees":
            if self.extra_trees_ is None:
                raise RuntimeError("model must be fitted before prediction")
            return self.extra_trees_.predict(X)
        if not self.models_:
            raise RuntimeError("model must be fitted before prediction")
        return {
            name: np.asarray(self.models_[name].predict(X), dtype=float)
            for name in self.target_names
        }
