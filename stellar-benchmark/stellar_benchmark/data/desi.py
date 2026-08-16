"""DESI DR1 stellar-spectrum ingestion, quality control, and perturbations.

The real-data case study uses RVSpecFit labels as reference labels. They are
not independent ground truth, so every generated manifest records that label
provenance explicitly.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

from ..config import DesiSNRConfig


RAW_KEYS = {
    "wavelength",
    "flux",
    "ivar",
    "valid",
    "valid_fraction",
    "targetid",
    "teff",
    "logg",
    "feh",
    "teff_err",
    "logg_err",
    "feh_err",
    "sn_r",
    "source_file",
}


def extract_desi_arm_spectra(
    files: Iterable[str | Path], output_path: str | Path, arm: str = "R"
) -> dict[str, object]:
    """Extract aligned, quality-flagged spectra from one DESI B/R/Z arm.

    The historical output key ``sn_r`` is retained for v0.3 compatibility but
    stores the selected arm's S/N value.
    """

    arm = arm.upper()
    if arm not in {"B", "R", "Z"}:
        raise ValueError("arm must be one of B, R, or Z")

    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise ImportError(
            "DESI FITS extraction requires the 'desi' extra: "
            "python -m pip install -e '.[desi]'"
        ) from exc

    paths = sorted((Path(path) for path in files), key=_healpix_sort_key)
    if not paths:
        raise ValueError("at least one DESI FITS file is required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DESI FITS files not found: {missing}")

    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in RAW_KEYS if key != "wavelength"
    }
    wavelength_reference: np.ndarray | None = None
    input_rows = 0

    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            required_hdus = {
                "FIBERMAP",
                "RVTAB",
                f"{arm}_WAVELENGTH",
                f"{arm}_FLUX",
                f"{arm}_IVAR",
                f"{arm}_MASK",
            }
            available_hdus = {hdu.name for hdu in hdul}
            absent = sorted(required_hdus - available_hdus)
            if absent:
                raise ValueError(f"{path.name} is missing HDUs: {absent}")

            rvt = hdul["RVTAB"].data
            fibermap = hdul["FIBERMAP"].data
            if len(rvt) != len(fibermap) or not np.array_equal(
                fibermap["TARGETID"], rvt["TARGETID"]
            ):
                raise ValueError(f"TARGETID alignment failed in {path.name}")

            wavelength = np.asarray(
                hdul[f"{arm}_WAVELENGTH"].data, dtype=np.float64
            ).squeeze()
            flux = np.asarray(hdul[f"{arm}_FLUX"].data, dtype=np.float32)
            ivar = np.asarray(hdul[f"{arm}_IVAR"].data, dtype=np.float32)
            mask = np.asarray(hdul[f"{arm}_MASK"].data)
            if flux.shape != ivar.shape or flux.shape != mask.shape:
                raise ValueError(f"{arm}-arm array shapes disagree in {path.name}")
            if flux.ndim != 2 or flux.shape[1] != len(wavelength):
                raise ValueError(f"invalid {arm}-arm wavelength grid in {path.name}")

            if wavelength_reference is None:
                wavelength_reference = wavelength.copy()
            elif not np.allclose(wavelength_reference, wavelength, rtol=0, atol=1e-8):
                raise ValueError(f"wavelength grid mismatch in {path.name}")

            good_star = rvt["SUCCESS"].astype(bool) & (rvt["RVS_WARN"] == 0)
            for column in ("TEFF", "LOGG", "FEH"):
                good_star &= np.isfinite(rvt[column])

            valid = (
                np.isfinite(flux)
                & np.isfinite(ivar)
                & (ivar > 0)
                & (mask == 0)
            )
            input_rows += len(rvt)

            parts["flux"].append(flux[good_star])
            parts["ivar"].append(ivar[good_star])
            parts["valid"].append(valid[good_star])
            parts["valid_fraction"].append(valid[good_star].mean(axis=1))
            parts["targetid"].append(
                np.asarray(rvt["TARGETID"][good_star], dtype=np.int64)
            )
            for output_name, column in (
                ("teff", "TEFF"),
                ("logg", "LOGG"),
                ("feh", "FEH"),
                ("teff_err", "TEFF_ERR"),
                ("logg_err", "LOGG_ERR"),
                ("feh_err", "FEH_ERR"),
                ("sn_r", f"SN_{arm}"),
            ):
                parts[output_name].append(
                    np.asarray(rvt[column][good_star], dtype=np.float32)
                )
            parts["source_file"].append(
                np.full(int(good_star.sum()), path.name, dtype="U128")
            )

    assert wavelength_reference is not None
    arrays = {
        key: np.concatenate(value, axis=0) for key, value in parts.items()
    }
    arrays["wavelength"] = wavelength_reference

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return {
        "arm": arm,
        "files": len(paths),
        "input_rows": input_rows,
        "quality_flagged_rows": int(len(arrays["targetid"])),
        "unique_targetids": int(np.unique(arrays["targetid"]).size),
        "shape": tuple(arrays["flux"].shape),
        "output_path": destination,
    }


def extract_desi_r_spectra(
    files: Iterable[str | Path], output_path: str | Path
) -> dict[str, object]:
    """Backward-compatible wrapper for the DESI R arm."""

    return extract_desi_arm_spectra(files, output_path, arm="R")


def load_desi_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load and strictly validate an extracted DESI NPZ dataset."""

    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"DESI NPZ dataset not found: {data_path}")
    with np.load(data_path, allow_pickle=False) as source:
        missing = sorted(RAW_KEYS - set(source.files))
        if missing:
            raise ValueError(f"DESI NPZ dataset is missing arrays: {missing}")
        data = {key: np.array(source[key], copy=True) for key in RAW_KEYS}

    n_rows = len(data["targetid"])
    flux_shape = data["flux"].shape
    if len(flux_shape) != 2 or flux_shape[0] != n_rows:
        raise ValueError("flux must be a two-dimensional row-aligned array")
    for key in ("ivar", "valid"):
        if data[key].shape != flux_shape:
            raise ValueError(f"{key} must have the same shape as flux")
    if len(data["wavelength"]) != flux_shape[1]:
        raise ValueError("wavelength length must match the flux columns")
    for key in RAW_KEYS - {"wavelength", "flux", "ivar", "valid"}:
        if len(data[key]) != n_rows:
            raise ValueError(f"{key} is not row-aligned")
    if not np.isfinite(data["wavelength"]).all():
        raise ValueError("wavelength must be finite")
    if not np.all(np.diff(data["wavelength"]) > 0):
        raise ValueError("wavelength must be strictly increasing")
    return data


def create_split_manifest(
    data: dict[str, np.ndarray], config: DesiSNRConfig
) -> pd.DataFrame:
    """Apply declared quality cuts and build four disjoint TARGETID splits."""

    n = len(data["targetid"])
    pass_pixels = data["valid_fraction"] >= config.valid_fraction_min
    pass_snr = data["sn_r"] >= config.snr_min
    pass_range = (
        _inside(data["teff"], config.teff_range)
        & _inside(data["logg"], config.logg_range)
        & _inside(data["feh"], config.feh_range)
    )
    pass_uncertainty = (
        _positive_finite_below(data["teff_err"], config.teff_error_max)
        & _positive_finite_below(data["logg_err"], config.logg_error_max)
        & _positive_finite_below(data["feh_err"], config.feh_error_max)
    )
    selected = pass_pixels & pass_snr & pass_range & pass_uncertainty

    # If repeated observations are present, retain the highest-S/N spectrum.
    candidate = pd.DataFrame(
        {
            "row_index": np.arange(n),
            "targetid": data["targetid"],
            "sn_r": data["sn_r"],
            "selected": selected,
        }
    )
    keep_rows = (
        candidate.loc[candidate["selected"]]
        .sort_values(["targetid", "sn_r"], ascending=[True, False])
        .drop_duplicates("targetid", keep="first")["row_index"]
        .to_numpy(dtype=int)
    )
    unique_selected = np.zeros(n, dtype=bool)
    unique_selected[keep_rows] = True
    selected &= unique_selected
    selected_indices = np.flatnonzero(selected)
    if len(selected_indices) < 40:
        raise ValueError("at least 40 quality-selected stars are required")

    strata = _joint_strata(
        data["teff"][selected_indices], data["sn_r"][selected_indices]
    )
    strata_full = np.full(n, -1, dtype=int)
    strata_full[selected_indices] = strata

    target_total = (
        config.target_adaptation_fraction + config.target_evaluation_fraction
    )
    source_pool, target_pool = train_test_split(
        selected_indices,
        test_size=target_total,
        random_state=config.random_state,
        stratify=strata_full[selected_indices],
    )
    source_holdout_within_source = config.source_holdout_fraction / (
        config.source_train_fraction + config.source_holdout_fraction
    )
    source_train, source_holdout = train_test_split(
        source_pool,
        test_size=source_holdout_within_source,
        random_state=config.random_state,
        stratify=strata_full[source_pool],
    )
    target_evaluation_within_target = config.target_evaluation_fraction / target_total
    target_adaptation, target_evaluation = train_test_split(
        target_pool,
        test_size=target_evaluation_within_target,
        random_state=config.random_state,
        stratify=strata_full[target_pool],
    )

    split = np.full(n, "excluded", dtype="U24")
    split[source_train] = "source_train"
    split[source_holdout] = "source_holdout"
    split[target_adaptation] = "target_adaptation"
    split[target_evaluation] = "target_evaluation"

    manifest = pd.DataFrame(
        {
            "row_index": np.arange(n),
            "targetid": data["targetid"],
            "source_file": data["source_file"],
            "teff": data["teff"],
            "logg": data["logg"],
            "feh": data["feh"],
            "teff_err": data["teff_err"],
            "logg_err": data["logg_err"],
            "feh_err": data["feh_err"],
            "sn_r": data["sn_r"],
            "valid_fraction": data["valid_fraction"],
            "pass_pixels": pass_pixels,
            "pass_snr": pass_snr,
            "pass_range": pass_range,
            "pass_uncertainty": pass_uncertainty,
            "selected": selected,
            "split": split,
        }
    )
    assert_no_targetid_leakage(manifest)
    return manifest


def assert_no_targetid_leakage(manifest: pd.DataFrame) -> None:
    """Raise if any selected TARGETID appears in more than one partition."""

    selected = manifest.loc[manifest["split"] != "excluded", ["targetid", "split"]]
    partitions_per_target = selected.groupby("targetid")["split"].nunique()
    if (partitions_per_target > 1).any():
        raise ValueError("TARGETID leakage detected across partitions")


def continuum_normalize(
    flux: np.ndarray,
    valid: np.ndarray,
    window_length: int,
    polyorder: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Median-scale and Savitzky-Golay continuum-normalize each spectrum."""

    flux = np.asarray(flux, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if flux.ndim != 2 or flux.shape != valid.shape:
        raise ValueError("flux and valid must be same-shaped 2D arrays")
    if window_length > flux.shape[1] or window_length < 3 or window_length % 2 == 0:
        raise ValueError("window_length must be odd and fit within the spectrum")
    if not 0 <= polyorder < window_length:
        raise ValueError("polyorder must be smaller than window_length")
    if (valid.sum(axis=1) < 2).any():
        raise ValueError("every spectrum needs at least two valid pixels")

    masked_flux = np.where(valid, flux, np.nan)
    scale = np.nanmedian(masked_flux, axis=1)
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("every spectrum must have a positive finite median flux")
    scaled = flux / scale[:, None]

    pixel_index = np.arange(flux.shape[1])
    filled = np.empty_like(scaled)
    for row in range(len(scaled)):
        good = valid[row] & np.isfinite(scaled[row])
        filled[row] = np.interp(pixel_index, pixel_index[good], scaled[row, good])

    continuum = savgol_filter(
        filled,
        window_length=window_length,
        polyorder=polyorder,
        axis=1,
        mode="interp",
    )
    normalized_valid = valid & np.isfinite(continuum) & (continuum > 0.05)
    normalized = np.full_like(filled, np.nan, dtype=np.float32)
    np.divide(filled, continuum, out=normalized, where=normalized_valid)
    normalized[normalized_valid] -= 1.0
    return normalized, normalized_valid


def inject_noise(
    flux: np.ndarray,
    ivar: np.ndarray,
    valid: np.ndarray,
    targetids: np.ndarray,
    factor: float | np.ndarray,
    seed: int,
) -> np.ndarray:
    """Increase noise standard deviation by a declared factor.

    Independent Gaussian noise with standard deviation
    ``sqrt(factor**2 - 1) / sqrt(ivar)`` is added, making the final standard
    deviation approximately ``factor`` times the original. Per-star random
    streams are derived from TARGETID, so results do not depend on row order.
    """

    flux = np.asarray(flux, dtype=np.float32)
    ivar = np.asarray(ivar, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    targetids = np.asarray(targetids, dtype=np.int64).reshape(-1)
    if flux.ndim != 2 or flux.shape != ivar.shape or flux.shape != valid.shape:
        raise ValueError("flux, ivar, and valid must be same-shaped 2D arrays")
    if len(targetids) != len(flux):
        raise ValueError("targetids must be row-aligned")

    factors = np.asarray(factor, dtype=float)
    if factors.ndim == 0:
        factors = np.full(len(flux), float(factors))
    factors = factors.reshape(-1)
    if len(factors) != len(flux) or not np.isfinite(factors).all():
        raise ValueError("factor must be finite and scalar or row-aligned")
    if (factors < 1).any():
        raise ValueError("noise factors must be at least one")
    if not np.isfinite(ivar[valid]).all() or (ivar[valid] <= 0).any():
        raise ValueError("valid pixels must have positive finite inverse variance")

    standard_noise = np.empty_like(flux)
    for row, targetid in enumerate(targetids):
        value = int(targetid)
        sequence = np.random.SeedSequence(
            [seed, value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF]
        )
        rng = np.random.default_rng(sequence)
        standard_noise[row] = rng.standard_normal(flux.shape[1])

    additional_sigma = np.zeros_like(flux)
    multiplier = np.sqrt(factors**2 - 1.0)[:, None]
    base_sigma = np.zeros_like(flux)
    base_sigma[valid] = 1.0 / np.sqrt(ivar[valid])
    additional_sigma = base_sigma * multiplier
    shifted = flux + standard_noise * additional_sigma
    shifted[~np.isfinite(flux)] = flux[~np.isfinite(flux)]
    return shifted.astype(np.float32)


def config_manifest(config: DesiSNRConfig) -> dict[str, object]:
    """Serializable experiment configuration plus scientific caveats."""

    return {
        "experiment": asdict(config),
        "split_unit": "TARGETID",
        "reference_labels": "DESI DR1 Stellar Reddening VAC RVSpecFit RVTAB",
        "label_caveat": (
            "RVSpecFit labels and their formal errors are pipeline-derived reference "
            "labels, not independent astrophysical ground truth."
        ),
        "shift_caveat": (
            "The S/N domain shift is a controlled Gaussian perturbation of real DESI "
            "spectra, not a naturally observed cross-survey shift."
        ),
    }


def _joint_strata(teff: np.ndarray, snr: np.ndarray) -> np.ndarray:
    teff_bin = np.asarray(
        pd.qcut(teff, q=3, labels=False, duplicates="drop"), dtype=int
    )
    snr_bin = np.asarray(
        pd.qcut(snr, q=5, labels=False, duplicates="drop"), dtype=int
    )
    snr_bins = int(snr_bin.max()) + 1
    joint = teff_bin * snr_bins + snr_bin
    _, counts = np.unique(joint, return_counts=True)
    if counts.min() >= 10:
        return joint
    fallback = np.asarray(
        pd.qcut(teff, q=min(5, len(teff) // 8), labels=False, duplicates="drop"),
        dtype=int,
    )
    _, fallback_counts = np.unique(fallback, return_counts=True)
    if fallback_counts.min() >= 4:
        return fallback
    return np.zeros(len(teff), dtype=int)


def _inside(values: np.ndarray, interval: list[float]) -> np.ndarray:
    return np.isfinite(values) & (values >= interval[0]) & (values <= interval[1])


def _positive_finite_below(values: np.ndarray, maximum: float) -> np.ndarray:
    return np.isfinite(values) & (values > 0) & (values <= maximum)


def _healpix_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem.rsplit("-", maxsplit=1)[-1]), path.name
    except ValueError:
        return 0, path.name
