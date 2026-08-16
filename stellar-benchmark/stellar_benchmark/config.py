"""Configuration objects for surveys and benchmark experiments.

Keeping these as plain dataclasses (rather than scattering magic strings through
the code) makes every experiment reproducible from a single YAML file, which is
exactly the kind of thing reviewers of cross-survey ML work say is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SurveyConfig:
    """Describes one survey's data source and preprocessing choices."""

    name: str
    # Columns in the loaded dataframe that serve as input features (flux array
    # column name, or a list of scalar feature columns).
    feature_columns: list[str]
    # Target stellar parameters to predict, e.g. ["teff", "logg", "feh"].
    target_columns: list[str]
    # Minimum signal-to-noise ratio to keep a spectrum.
    snr_min: float = 50.0
    # Optional stellar-type column used for stratified evaluation.
    stellar_type_column: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("survey name must not be empty")
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if not self.target_columns:
            raise ValueError("target_columns must not be empty")
        if self.snr_min < 0:
            raise ValueError("snr_min must be non-negative")


@dataclass
class ExperimentConfig:
    """Describes a single train-on-A / evaluate-on-B experiment."""

    source: SurveyConfig
    target: SurveyConfig
    model_type: str = "gbm"  # "gbm" or "mlp"
    random_state: int = 42
    # Fraction of the target-domain crossmatch sample reserved for supervised
    # adaptation. The rest is held out for evaluation.
    target_adaptation_fraction: float = 0.3
    results_dir: str = "results"
    strata: list[str] = field(default_factory=lambda: ["snr_bin"])

    def __post_init__(self) -> None:
        if self.model_type not in {"gbm", "mlp"}:
            raise ValueError("model_type must be 'gbm' or 'mlp'")
        if not 0 < self.target_adaptation_fraction < 1:
            raise ValueError("target_adaptation_fraction must be between 0 and 1")


@dataclass
class SyntheticDemoConfig:
    """Configuration for the deterministic end-to-end smoke experiment."""

    name: str = "synthetic_smoke"
    random_state: int = 42
    model_type: str = "gbm"
    n_bootstrap: int = 3
    source_rows: int = 600
    target_rows: int = 300
    target_adaptation_fraction: float = 0.30
    output_dir: str = "results/synthetic_smoke"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if self.model_type not in {"gbm", "mlp"}:
            raise ValueError("model_type must be 'gbm' or 'mlp'")
        if self.n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2 for an uncertainty proxy")
        if self.source_rows < 50 or self.target_rows < 50:
            raise ValueError("source_rows and target_rows must each be at least 50")
        if not 0 < self.target_adaptation_fraction < 1:
            raise ValueError("target_adaptation_fraction must be between 0 and 1")


def load_synthetic_demo_config(path: str | Path) -> SyntheticDemoConfig:
    """Load and validate a synthetic-demo YAML configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    unknown = set(raw) - set(SyntheticDemoConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return SyntheticDemoConfig(**raw)


@dataclass
class DesiSNRConfig:
    """Configuration for the real-DESI controlled S/N case study."""

    name: str = "desi_snr_case_study"
    input_npz: str = "data/processed/desi_r_raw.npz"
    output_dir: str = "results/desi_case_study"
    random_state: int = 42

    valid_fraction_min: float = 0.95
    snr_min: float = 10.0
    teff_range: list[float] = field(default_factory=lambda: [3200.0, 8000.0])
    logg_range: list[float] = field(default_factory=lambda: [2.0, 5.5])
    feh_range: list[float] = field(default_factory=lambda: [-3.0, 0.5])
    teff_error_max: float = 150.0
    logg_error_max: float = 0.20
    feh_error_max: float = 0.30

    source_train_fraction: float = 0.60
    source_holdout_fraction: float = 0.20
    target_adaptation_fraction: float = 0.10
    target_evaluation_fraction: float = 0.10

    wavelength_min: float = 5800.0
    wavelength_max: float = 7580.0
    continuum_window: int = 301
    continuum_polyorder: int = 3
    outlier_percentiles: list[float] = field(default_factory=lambda: [0.01, 99.99])
    pca_components: int = 64

    n_estimators: int = 400
    min_samples_leaf: int = 2
    max_features: float = 0.70

    noise_factors: list[float] = field(
        default_factory=lambda: [1.25, 1.50, 2.0, 2.5, 3.0]
    )
    noise_seed_start: int = 2026
    noise_seed_count: int = 10
    adaptation_noise_factor: float = 2.0
    adaptation_noise_seed: int = 2026

    augmentation_views: int = 2
    augmentation_seed_start: int = 4042

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not 0 < self.valid_fraction_min <= 1:
            raise ValueError("valid_fraction_min must be in (0, 1]")
        if self.snr_min < 0:
            raise ValueError("snr_min must be non-negative")
        for name in ("teff_range", "logg_range", "feh_range"):
            values = getattr(self, name)
            if len(values) != 2 or values[0] >= values[1]:
                raise ValueError(f"{name} must contain two increasing values")
        if min(self.teff_error_max, self.logg_error_max, self.feh_error_max) <= 0:
            raise ValueError("label-error maxima must be positive")

        fractions = [
            self.source_train_fraction,
            self.source_holdout_fraction,
            self.target_adaptation_fraction,
            self.target_evaluation_fraction,
        ]
        if any(value <= 0 for value in fractions) or not abs(sum(fractions) - 1) < 1e-9:
            raise ValueError("split fractions must be positive and sum to one")
        if self.wavelength_min >= self.wavelength_max:
            raise ValueError("wavelength_min must be smaller than wavelength_max")
        if self.continuum_window < 3 or self.continuum_window % 2 == 0:
            raise ValueError("continuum_window must be an odd integer of at least 3")
        if not 0 <= self.continuum_polyorder < self.continuum_window:
            raise ValueError("continuum_polyorder must be smaller than the window")
        if (
            len(self.outlier_percentiles) != 2
            or not 0 <= self.outlier_percentiles[0] < self.outlier_percentiles[1] <= 100
        ):
            raise ValueError("outlier_percentiles must be two increasing percentiles")
        if self.pca_components < 1:
            raise ValueError("pca_components must be positive")
        if self.n_estimators < 1 or self.min_samples_leaf < 1:
            raise ValueError("tree counts and leaf sizes must be positive")
        if not 0 < self.max_features <= 1:
            raise ValueError("max_features must be in (0, 1]")
        if not self.noise_factors or any(value <= 1 for value in self.noise_factors):
            raise ValueError("noise_factors must all be greater than one")
        if self.noise_seed_count < 1:
            raise ValueError("noise_seed_count must be positive")
        if self.adaptation_noise_factor <= 1:
            raise ValueError("adaptation_noise_factor must be greater than one")
        if self.augmentation_views < 0:
            raise ValueError("augmentation_views must not be negative")


def load_desi_snr_config(path: str | Path) -> DesiSNRConfig:
    """Load and validate a DESI S/N experiment YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    unknown = set(raw) - set(DesiSNRConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return DesiSNRConfig(**raw)


@dataclass
class DesiReliabilityConfig(DesiSNRConfig):
    """Extended real-DESI experiment with reliability and adaptation audits."""

    output_dir: str = "results/desi_reliability"
    bootstrap_replicates: int = 500
    bootstrap_confidence: float = 0.95
    conformal_alphas: list[float] = field(default_factory=lambda: [0.32, 0.10, 0.05])
    ood_coverages: list[float] = field(
        default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.5]
    )
    coral_regularization: float = 1e-3
    label_budgets: list[int] = field(default_factory=lambda: [5, 10, 25, 50, 90])
    label_budget_repeats: int = 5
    ablation_models: list[str] = field(
        default_factory=lambda: ["ridge", "extra_trees", "mlp"]
    )
    representation_components: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128]
    )
    ablation_estimators: int = 150
    ablation_noise_factor: float = 2.0
    subgroup_minimum: int = 8
    isochrone_grid_csv: str | None = None
    isochrone_threshold_quantile: float = 0.99

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bootstrap_replicates < 20:
            raise ValueError("bootstrap_replicates must be at least 20")
        if not 0 < self.bootstrap_confidence < 1:
            raise ValueError("bootstrap_confidence must be in (0, 1)")
        if not self.conformal_alphas or any(
            not 0 < value < 1 for value in self.conformal_alphas
        ):
            raise ValueError("conformal_alphas must contain values in (0, 1)")
        if not self.ood_coverages or any(
            not 0 < value <= 1 for value in self.ood_coverages
        ):
            raise ValueError("ood_coverages must contain values in (0, 1]")
        if self.coral_regularization <= 0:
            raise ValueError("coral_regularization must be positive")
        if not self.label_budgets or any(value < 1 for value in self.label_budgets):
            raise ValueError("label_budgets must contain positive integers")
        if self.label_budget_repeats < 1:
            raise ValueError("label_budget_repeats must be positive")
        if any(
            value not in {"ridge", "extra_trees", "mlp"}
            for value in self.ablation_models
        ):
            raise ValueError("ablation_models contains an unsupported family")
        if not self.representation_components or any(
            value < 1 for value in self.representation_components
        ):
            raise ValueError("representation_components must be positive")
        if self.ablation_estimators < 1:
            raise ValueError("ablation_estimators must be positive")
        if self.ablation_noise_factor <= 1:
            raise ValueError("ablation_noise_factor must be greater than one")
        if self.subgroup_minimum < 5:
            raise ValueError("subgroup_minimum must be at least five")
        if not 0 < self.isochrone_threshold_quantile < 1:
            raise ValueError("isochrone_threshold_quantile must be in (0, 1)")


def load_desi_reliability_config(path: str | Path) -> DesiReliabilityConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    unknown = set(raw) - set(DesiReliabilityConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return DesiReliabilityConfig(**raw)


@dataclass
class CrossSurveyConfig:
    """LAMOST-to-DESI or other survey-to-survey transfer experiment."""

    name: str = "lamost_to_desi"
    source_npz: str = "data/processed/lamost_shared_blue.npz"
    target_npz: str = "data/processed/desi_shared_blue.npz"
    output_dir: str = "results/lamost_to_desi"
    source_survey: str = "LAMOST"
    target_survey: str = "DESI"
    source_label_provenance: str = "declared in dataset manifest"
    target_label_provenance: str = "declared in dataset manifest"
    source_label_scale: str = "unspecified"
    target_label_scale: str = "unspecified"
    source_resolving_power: float = 1800.0
    target_resolving_power: float = 2500.0
    common_resolving_power: float = 1800.0
    wavelength_min: float = 4000.0
    wavelength_max: float = 5500.0
    log_wavelength_step: float = 2e-4
    valid_fraction_min: float = 0.95
    source_holdout_fraction: float = 0.20
    target_adaptation_fraction: float = 0.30
    random_state: int = 42
    continuum_window: int = 101
    continuum_polyorder: int = 3
    outlier_percentiles: list[float] = field(default_factory=lambda: [0.01, 99.99])
    pca_components: int = 64
    n_estimators: int = 400
    min_samples_leaf: int = 2
    max_features: float = 0.70
    conformal_alpha: float = 0.10
    coral_regularization: float = 1e-3
    label_budgets: list[int] = field(default_factory=lambda: [5, 10, 25, 50, 100])
    label_budget_repeats: int = 5
    bootstrap_replicates: int = 500
    bootstrap_confidence: float = 0.95
    ood_coverages: list[float] = field(
        default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.5]
    )
    subgroup_minimum: int = 20
    ablation_models: list[str] = field(
        default_factory=lambda: ["ridge", "extra_trees", "mlp"]
    )
    ablation_estimators: int = 150
    isochrone_grid_csv: str | None = None
    isochrone_threshold_quantile: float = 0.99
    enforce_publication_gate: bool = False
    publication_min_source_train: int = 1000
    publication_min_source_holdout: int = 200
    publication_min_target_adaptation: int = 100
    publication_min_target_evaluation: int = 350
    publication_min_giants: int = 50
    publication_min_metal_poor: int = 50

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.source_survey or not self.target_survey:
            raise ValueError("experiment and survey names must not be empty")
        if not 0 < self.source_holdout_fraction < 1:
            raise ValueError("source_holdout_fraction must be in (0, 1)")
        if not 0 < self.target_adaptation_fraction < 1:
            raise ValueError("target_adaptation_fraction must be in (0, 1)")
        if self.wavelength_min <= 0 or self.wavelength_min >= self.wavelength_max:
            raise ValueError("wavelength bounds must be positive and increasing")
        if self.log_wavelength_step <= 0:
            raise ValueError("log_wavelength_step must be positive")
        if not 0 < self.valid_fraction_min <= 1:
            raise ValueError("valid_fraction_min must be in (0, 1]")
        if min(
            self.source_resolving_power,
            self.target_resolving_power,
            self.common_resolving_power,
        ) <= 0:
            raise ValueError("resolving powers must be positive")
        if self.common_resolving_power > min(
            self.source_resolving_power, self.target_resolving_power
        ):
            raise ValueError("common resolution cannot exceed either survey resolution")
        if self.continuum_window < 3 or self.continuum_window % 2 == 0:
            raise ValueError("continuum_window must be an odd integer of at least 3")
        if not 0 <= self.continuum_polyorder < self.continuum_window:
            raise ValueError("continuum_polyorder must be smaller than the window")
        if self.pca_components < 1 or self.n_estimators < 1:
            raise ValueError("PCA and tree counts must be positive")
        if not 0 < self.conformal_alpha < 1:
            raise ValueError("conformal_alpha must be in (0, 1)")
        if self.coral_regularization <= 0:
            raise ValueError("coral_regularization must be positive")
        if not self.label_budgets or any(value < 1 for value in self.label_budgets):
            raise ValueError("label_budgets must contain positive integers")
        if self.label_budget_repeats < 1 or self.bootstrap_replicates < 20:
            raise ValueError("repeat and bootstrap counts are too small")
        if not 0 < self.bootstrap_confidence < 1:
            raise ValueError("bootstrap_confidence must be in (0, 1)")
        if not self.ood_coverages or any(
            not 0 < value <= 1 for value in self.ood_coverages
        ):
            raise ValueError("ood_coverages must contain values in (0, 1]")
        if self.subgroup_minimum < 5:
            raise ValueError("subgroup_minimum must be at least five")
        if any(
            value not in {"ridge", "extra_trees", "mlp"}
            for value in self.ablation_models
        ):
            raise ValueError("ablation_models contains an unsupported family")
        if self.ablation_estimators < 1:
            raise ValueError("ablation_estimators must be positive")
        if not 0 < self.isochrone_threshold_quantile < 1:
            raise ValueError("isochrone_threshold_quantile must be in (0, 1)")
        publication_counts = (
            self.publication_min_source_train,
            self.publication_min_source_holdout,
            self.publication_min_target_adaptation,
            self.publication_min_target_evaluation,
            self.publication_min_giants,
            self.publication_min_metal_poor,
        )
        if any(value < 1 for value in publication_counts):
            raise ValueError("quality-gate counts must be positive")


def load_cross_survey_config(path: str | Path) -> CrossSurveyConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    unknown = set(raw) - set(CrossSurveyConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return CrossSurveyConfig(**raw)
