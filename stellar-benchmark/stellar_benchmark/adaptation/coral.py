"""Correlation Alignment (CORAL) for label-free target-domain adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _matrix_power_symmetric(matrix: np.ndarray, power: float, floor: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return (vectors * np.power(values, power)) @ vectors.T


@dataclass
class CORALAdapter:
    """Align source feature covariance to an unlabeled target sample.

    The adapter learns means and regularized covariance matrices from source
    and target-adaptation features. It transforms labeled source features into
    target style; a fresh predictor is then trained on the transformed source
    features and evaluated on model-held-out target features.
    """

    regularization: float = 1e-3
    source_mean_: np.ndarray | None = None
    target_mean_: np.ndarray | None = None
    transform_: np.ndarray | None = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "CORALAdapter":
        X_source = self._validate(X_source, "X_source")
        X_target = self._validate(X_target, "X_target")
        if X_source.shape[1] != X_target.shape[1]:
            raise ValueError("source and target must have the same feature width")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        source_mean = X_source.mean(axis=0)
        target_mean = X_target.mean(axis=0)
        source_cov = np.cov(X_source - source_mean, rowvar=False)
        target_cov = np.cov(X_target - target_mean, rowvar=False)
        dimension = X_source.shape[1]
        source_cov = source_cov + self.regularization * np.eye(dimension)
        target_cov = target_cov + self.regularization * np.eye(dimension)
        whitening = _matrix_power_symmetric(
            source_cov, -0.5, self.regularization
        )
        coloring = _matrix_power_symmetric(
            target_cov, 0.5, self.regularization
        )
        self.source_mean_ = source_mean
        self.target_mean_ = target_mean
        self.transform_ = whitening @ coloring
        return self

    def transform_source(self, X_source: np.ndarray) -> np.ndarray:
        if (
            self.source_mean_ is None
            or self.target_mean_ is None
            or self.transform_ is None
        ):
            raise RuntimeError("CORAL adapter must be fitted before transformation")
        X_source = self._validate(X_source, "X_source")
        if X_source.shape[1] != len(self.source_mean_):
            raise ValueError("source features do not match fitted width")
        aligned = (X_source - self.source_mean_) @ self.transform_ + self.target_mean_
        return aligned.astype(np.float32)

    @staticmethod
    def _validate(X: np.ndarray, name: str) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all():
            raise ValueError(f"{name} must be a finite 2D array with at least two rows")
        return X
