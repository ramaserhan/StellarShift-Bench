"""Leakage-safe feature and model pipelines for continuum-normalized spectra."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer


class SpectralFeaturePipeline:
    """Trim, mask, impute, and compress spectra using source training only."""

    def __init__(
        self,
        wavelength_min: float = 5800.0,
        wavelength_max: float = 7580.0,
        outlier_percentiles: tuple[float, float] = (0.01, 99.99),
        pca_components: int = 64,
        random_state: int = 42,
    ) -> None:
        if wavelength_min >= wavelength_max:
            raise ValueError("wavelength_min must be smaller than wavelength_max")
        if (
            len(outlier_percentiles) != 2
            or not 0 <= outlier_percentiles[0] < outlier_percentiles[1] <= 100
        ):
            raise ValueError("outlier_percentiles must be increasing")
        if pca_components < 1:
            raise ValueError("pca_components must be positive")
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.outlier_percentiles = outlier_percentiles
        self.pca_components = pca_components
        self.random_state = random_state
        self.wavelength_keep_: np.ndarray | None = None
        self.outlier_bounds_: tuple[float, float] | None = None
        self.imputer_: SimpleImputer | None = None
        self.pca_: PCA | None = None

    def fit(
        self, normalized_flux: np.ndarray, valid: np.ndarray, wavelength: np.ndarray
    ) -> "SpectralFeaturePipeline":
        flux, valid, wavelength = _validate_spectral_arrays(
            normalized_flux, valid, wavelength
        )
        wavelength_keep = (
            (wavelength >= self.wavelength_min)
            & (wavelength <= self.wavelength_max)
        )
        if not wavelength_keep.any():
            raise ValueError("configured wavelength interval retains no pixels")
        features = flux[:, wavelength_keep].copy()
        feature_valid = valid[:, wavelength_keep] & np.isfinite(features)
        train_values = features[feature_valid]
        if len(train_values) == 0:
            raise ValueError("training spectra contain no valid pixels")
        lower, upper = np.percentile(train_values, self.outlier_percentiles)
        features[~feature_valid] = np.nan
        features[(features < lower) | (features > upper)] = np.nan

        imputer = SimpleImputer(strategy="median")
        imputed = imputer.fit_transform(features)
        max_components = min(imputed.shape[0] - 1, imputed.shape[1])
        if max_components < 1:
            raise ValueError("at least two training spectra are required")
        n_components = min(self.pca_components, max_components)
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized" if n_components < max_components else "auto",
            random_state=self.random_state,
        )
        pca.fit(imputed)

        self.wavelength_keep_ = wavelength_keep
        self.outlier_bounds_ = (float(lower), float(upper))
        self.imputer_ = imputer
        self.pca_ = pca
        return self

    def transform(
        self, normalized_flux: np.ndarray, valid: np.ndarray
    ) -> tuple[np.ndarray, int]:
        self._require_fitted()
        flux = np.asarray(normalized_flux, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        if flux.ndim != 2 or flux.shape != valid.shape:
            raise ValueError("normalized_flux and valid must be same-shaped 2D arrays")
        assert self.wavelength_keep_ is not None
        if flux.shape[1] != len(self.wavelength_keep_):
            raise ValueError("spectrum width does not match the fitted wavelength grid")
        assert self.outlier_bounds_ is not None
        lower, upper = self.outlier_bounds_
        features = flux[:, self.wavelength_keep_].copy()
        feature_valid = valid[:, self.wavelength_keep_] & np.isfinite(features)
        features[~feature_valid] = np.nan
        extreme = np.isfinite(features) & (
            (features < lower) | (features > upper)
        )
        features[extreme] = np.nan
        assert self.imputer_ is not None and self.pca_ is not None
        transformed = self.pca_.transform(self.imputer_.transform(features))
        return transformed.astype(np.float32), int(extreme.sum())

    def fit_transform(
        self, normalized_flux: np.ndarray, valid: np.ndarray, wavelength: np.ndarray
    ) -> np.ndarray:
        self.fit(normalized_flux, valid, wavelength)
        transformed, _ = self.transform(normalized_flux, valid)
        return transformed

    @property
    def explained_variance_ratio(self) -> float:
        self._require_fitted()
        assert self.pca_ is not None
        return float(self.pca_.explained_variance_ratio_.sum())

    def _require_fitted(self) -> None:
        if self.pca_ is None or self.imputer_ is None:
            raise RuntimeError("feature pipeline must be fitted first")


class SeparateExtraTreesRegressor:
    """One ExtraTrees model per physical target to avoid unit-scale coupling."""

    def __init__(
        self,
        target_names: tuple[str, ...] = ("teff", "logg", "feh"),
        n_estimators: int = 400,
        min_samples_leaf: int = 2,
        max_features: float = 0.7,
        random_state: int = 42,
    ) -> None:
        if not target_names:
            raise ValueError("target_names must not be empty")
        self.target_names = target_names
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.models_: dict[str, ExtraTreesRegressor] = {}

    def fit(
        self,
        X: np.ndarray,
        targets: dict[str, np.ndarray],
        sample_weight: np.ndarray | None = None,
    ) -> "SeparateExtraTreesRegressor":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a non-empty finite 2D array")
        if set(targets) != set(self.target_names):
            raise ValueError("targets must exactly match target_names")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(sample_weight) != len(X) or (sample_weight < 0).any():
                raise ValueError("sample_weight must be non-negative and row-aligned")

        self.models_ = {}
        for name in self.target_names:
            values = np.asarray(targets[name], dtype=float).reshape(-1)
            if len(values) != len(X) or not np.isfinite(values).all():
                raise ValueError(f"target {name!r} must be finite and row-aligned")
            model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_jobs=-1,
                random_state=self.random_state,
            )
            model.fit(X, values, sample_weight=sample_weight)
            self.models_[name] = model
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if not self.models_:
            raise RuntimeError("model must be fitted before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D array")
        return {name: self.models_[name].predict(X) for name in self.target_names}


def _validate_spectral_arrays(
    flux: np.ndarray, valid: np.ndarray, wavelength: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux = np.asarray(flux, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    wavelength = np.asarray(wavelength, dtype=float).reshape(-1)
    if flux.ndim != 2 or flux.shape != valid.shape:
        raise ValueError("flux and valid must be same-shaped 2D arrays")
    if flux.shape[1] != len(wavelength):
        raise ValueError("wavelength length must match spectral columns")
    if not np.isfinite(wavelength).all() or not np.all(np.diff(wavelength) > 0):
        raise ValueError("wavelength must be finite and strictly increasing")
    return flux, valid, wavelength
