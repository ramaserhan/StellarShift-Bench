"""Survey-neutral spectral schema and wavelength/resolution harmonization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


TARGET_NAMES = ("teff", "logg", "feh")


@dataclass(frozen=True)
class SurveySpectra:
    survey: str
    wavelength: np.ndarray
    flux: np.ndarray
    valid: np.ndarray
    object_id: np.ndarray
    snr: np.ndarray
    targets: dict[str, np.ndarray]
    target_errors: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if not self.survey.strip():
            raise ValueError("survey must not be empty")
        wavelength = np.asarray(self.wavelength, dtype=float).reshape(-1)
        flux = np.asarray(self.flux, dtype=float)
        valid = np.asarray(self.valid, dtype=bool)
        object_id = np.asarray(self.object_id).reshape(-1)
        snr = np.asarray(self.snr, dtype=float).reshape(-1)
        if not np.isfinite(wavelength).all() or not np.all(np.diff(wavelength) > 0):
            raise ValueError("wavelength must be finite and strictly increasing")
        if flux.ndim != 2 or flux.shape != valid.shape:
            raise ValueError("flux and valid must be same-shaped 2D arrays")
        if flux.shape[1] != len(wavelength):
            raise ValueError("wavelength must match the flux width")
        if len(object_id) != len(flux) or len(snr) != len(flux):
            raise ValueError("object_id and snr must be row-aligned")
        if not np.isfinite(snr).all():
            raise ValueError("snr must be finite")
        if set(self.targets) != set(TARGET_NAMES):
            raise ValueError(f"targets must be exactly {TARGET_NAMES}")
        for name, values in self.targets.items():
            values = np.asarray(values, dtype=float).reshape(-1)
            if len(values) != len(flux) or not np.isfinite(values).all():
                raise ValueError(f"target {name!r} must be finite and row-aligned")
        if self.target_errors is not None:
            if set(self.target_errors) != set(TARGET_NAMES):
                raise ValueError(f"target_errors must be exactly {TARGET_NAMES}")
            for name, values in self.target_errors.items():
                values = np.asarray(values, dtype=float).reshape(-1)
                if (
                    len(values) != len(flux)
                    or not np.isfinite(values).all()
                    or np.any(values <= 0)
                ):
                    raise ValueError(
                        f"target error {name!r} must be positive, finite, and row-aligned"
                    )


def load_survey_npz(path: str | Path, survey: str | None = None) -> SurveySpectra:
    """Load the public cross-survey NPZ contract."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"survey NPZ not found: {source_path}")
    required = {"wavelength", "flux", "valid", "object_id", "snr", *TARGET_NAMES}
    with np.load(source_path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"survey NPZ is missing arrays: {missing}")
        inferred_survey = survey
        if inferred_survey is None and "survey" in data.files:
            inferred_survey = str(np.asarray(data["survey"]).item())
        if inferred_survey is None:
            inferred_survey = source_path.stem
        error_keys = {f"{name}_err" for name in TARGET_NAMES}
        present_error_keys = error_keys & set(data.files)
        if present_error_keys and present_error_keys != error_keys:
            missing_errors = sorted(error_keys - present_error_keys)
            raise ValueError(
                "survey NPZ has an incomplete target-error contract; "
                f"missing arrays: {missing_errors}"
            )
        target_errors = None
        if present_error_keys:
            target_errors = {
                name: np.array(data[f"{name}_err"], copy=True)
                for name in TARGET_NAMES
            }
        return SurveySpectra(
            survey=inferred_survey,
            wavelength=np.array(data["wavelength"], copy=True),
            flux=np.array(data["flux"], copy=True),
            valid=np.array(data["valid"], copy=True),
            object_id=np.array(data["object_id"], copy=True),
            snr=np.array(data["snr"], copy=True),
            targets={name: np.array(data[name], copy=True) for name in TARGET_NAMES},
            target_errors=target_errors,
        )


def shared_log_wavelength_grid(
    wavelength_min: float,
    wavelength_max: float,
    log_step: float,
) -> np.ndarray:
    """Create a common constant-velocity wavelength grid."""

    if wavelength_min <= 0 or wavelength_min >= wavelength_max:
        raise ValueError("wavelength bounds must be positive and increasing")
    if log_step <= 0:
        raise ValueError("log_step must be positive")
    count = int(np.floor((np.log(wavelength_max) - np.log(wavelength_min)) / log_step)) + 1
    if count < 10:
        raise ValueError("shared grid must contain at least ten pixels")
    return np.exp(np.log(wavelength_min) + np.arange(count) * log_step)


def harmonize_spectra(
    wavelength: np.ndarray,
    flux: np.ndarray,
    valid: np.ndarray,
    target_wavelength: np.ndarray,
    input_resolving_power: float,
    target_resolving_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample spectra and, when needed, degrade them to a common resolution.

    Resolution matching assumes approximately Gaussian line-spread functions
    on the constant-log-wavelength target grid. This declared approximation is
    not a replacement for survey-specific line-spread-function vectors.
    """

    wavelength = np.asarray(wavelength, dtype=float).reshape(-1)
    flux = np.asarray(flux, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    target_wavelength = np.asarray(target_wavelength, dtype=float).reshape(-1)
    if flux.ndim != 2 or flux.shape != valid.shape or flux.shape[1] != len(wavelength):
        raise ValueError("flux, valid, and wavelength shapes do not align")
    if not np.all(np.diff(wavelength) > 0) or not np.all(np.diff(target_wavelength) > 0):
        raise ValueError("wavelength grids must be strictly increasing")
    if min(input_resolving_power, target_resolving_power) <= 0:
        raise ValueError("resolving powers must be positive")
    if input_resolving_power < target_resolving_power - 1e-9:
        raise ValueError("cannot sharpen spectra to a higher resolving power")

    output = np.full((len(flux), len(target_wavelength)), np.nan, dtype=np.float32)
    output_valid = np.zeros_like(output, dtype=bool)
    for row in range(len(flux)):
        good = valid[row] & np.isfinite(flux[row])
        if good.sum() < 2:
            continue
        inside = (
            (target_wavelength >= wavelength[good].min())
            & (target_wavelength <= wavelength[good].max())
        )
        output[row, inside] = np.interp(
            target_wavelength[inside], wavelength[good], flux[row, good]
        ).astype(np.float32)
        output_valid[row, inside] = True

    if input_resolving_power > target_resolving_power:
        log_step = float(np.median(np.diff(np.log(target_wavelength))))
        sigma_log = np.sqrt(
            (1 / target_resolving_power) ** 2 - (1 / input_resolving_power) ** 2
        ) / 2.354820045
        sigma_pixels = sigma_log / log_step
        if sigma_pixels > 0.05:
            weights = output_valid.astype(np.float32)
            filled = np.where(output_valid, output, 0.0)
            smoothed_flux = gaussian_filter1d(filled, sigma_pixels, axis=1, mode="nearest")
            smoothed_weight = gaussian_filter1d(weights, sigma_pixels, axis=1, mode="nearest")
            reliable = smoothed_weight > 0.8
            output[:] = np.nan
            np.divide(smoothed_flux, smoothed_weight, out=output, where=reliable)
            output_valid = reliable
    return output, output_valid


def save_survey_npz(dataset: SurveySpectra, path: str | Path) -> Path:
    """Write a validated survey dataset in the cross-survey contract."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    target_errors = {}
    if dataset.target_errors is not None:
        target_errors = {
            f"{name}_err": np.asarray(values)
            for name, values in dataset.target_errors.items()
        }
    np.savez_compressed(
        destination,
        survey=np.asarray(dataset.survey),
        wavelength=dataset.wavelength,
        flux=dataset.flux,
        valid=dataset.valid,
        object_id=dataset.object_id,
        snr=dataset.snr,
        **dataset.targets,
        **target_errors,
    )
    return destination
