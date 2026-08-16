"""Build real LAMOST and DESI contracts on the APOGEE DR12 label scale.

The public Ho et al. (2017) tutorial supplies high-S/N LAMOST DR2 spectra
with direct APOGEE DR12 ASPCAP labels.  DESI DR1 spectra are retrieved from
SPARCL for a disjoint APOGEE DR12 crossmatch.  Both domains therefore use the
same version-pinned reference labels while the optical input survey changes.

The legacy ``feh`` slot in the benchmark contract carries APOGEE DR12's
calibrated ``PARAM_M_H`` value.  This is global metallicity [M/H], not an
element-by-element ``FE_H`` abundance, and every public result must say so.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .cross_survey import SurveySpectra, save_survey_npz


LABEL_SCALE = "APOGEE_DR12_ASPCAP_v603_calibrated_TEFF_LOGG_PARAM_M_H"


def rest_frame_grid(
    minimum: float = 3900.0,
    maximum: float = 5600.0,
    step: float = 0.8,
) -> np.ndarray:
    """Return the shared linear grid used before benchmark harmonization."""

    if minimum <= 0 or maximum <= minimum or step <= 0:
        raise ValueError("rest-frame grid bounds and step must be positive")
    return np.arange(minimum, maximum + step / 2.0, step, dtype=np.float64)


def build_apogee_dr12_contracts(
    *,
    lamost_labels_fits: str | Path,
    lamost_spectra_dir: str | Path,
    apogee_allstar_fits: str | Path,
    desi_selected_csv: str | Path,
    desi_sparcl_pickles: Iterable[str | Path],
    desi_redshift_json: str | Path,
    source_output: str | Path,
    target_output: str | Path,
    source_manifest_output: str | Path,
    target_manifest_output: str | Path,
    grid: np.ndarray | None = None,
) -> dict[str, object]:
    """Create validated, disjoint survey contracts from public products."""

    grid = rest_frame_grid() if grid is None else np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not np.all(np.diff(grid) > 0):
        raise ValueError("grid must be one-dimensional and strictly increasing")
    source, source_manifest = _build_lamost_source(
        Path(lamost_labels_fits),
        Path(lamost_spectra_dir),
        Path(apogee_allstar_fits),
        grid,
    )
    target, target_manifest = _build_desi_target(
        Path(desi_selected_csv),
        [Path(path) for path in desi_sparcl_pickles],
        Path(desi_redshift_json),
        grid,
    )
    overlap = set(source.object_id.tolist()) & set(target.object_id.tolist())
    if overlap:
        raise ValueError(
            f"source and target contain {len(overlap)} overlapping APOGEE objects"
        )

    source_path = save_survey_npz(source, source_output)
    target_path = save_survey_npz(target, target_output)
    source_manifest_path = Path(source_manifest_output)
    target_manifest_path = Path(target_manifest_output)
    source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.to_csv(source_manifest_path, index=False)
    target_manifest.to_csv(target_manifest_path, index=False)
    return {
        "source_npz": source_path,
        "target_npz": target_path,
        "source_manifest": source_manifest_path,
        "target_manifest": target_manifest_path,
        "source_rows": len(source.object_id),
        "target_rows": len(target.object_id),
        "target_metal_poor": int(np.sum(target.targets["feh"] < -1.5)),
        "target_giants": int(np.sum(target.targets["logg"] < 3.5)),
        "label_scale": LABEL_SCALE,
    }


def _require_astropy():
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - optional data dependency
        raise ImportError(
            "APOGEE/LAMOST contract building requires astropy; "
            "install stellar-benchmark[desi]"
        ) from exc
    return fits


def _build_lamost_source(
    labels_path: Path,
    spectra_dir: Path,
    allstar_path: Path,
    grid: np.ndarray,
) -> tuple[SurveySpectra, pd.DataFrame]:
    fits = _require_astropy()
    labels = fits.getdata(labels_path, 1)
    label_frame = pd.DataFrame(
        {
            "lamost_id": np.char.strip(np.asarray(labels["LAMOST_ID"]).astype(str)),
            "apogee_id": np.char.strip(np.asarray(labels["APOGEE_ID"]).astype(str)),
            "ra": _native_float(labels["RA"]),
            "dec": _native_float(labels["Dec"]),
            "teff": _native_float(labels["TEFF"]),
            "logg": _native_float(labels["LOGG"]),
            "feh": _native_float(labels["PARAM_M_H"]),
        }
    )
    quality = _apogee_quality_rows(allstar_path, set(label_frame["apogee_id"]))
    label_frame = label_frame.merge(
        quality,
        on="apogee_id",
        how="inner",
        validate="one_to_one",
    )
    label_frame = label_frame.loc[~label_frame["ho2017_badstar"]].copy()
    label_frame = label_frame.sort_values("apogee_id").reset_index(drop=True)

    flux_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    snr_rows: list[float] = []
    source_files: list[str] = []
    for row in label_frame.itertuples(index=False):
        spectrum_path = spectra_dir / row.lamost_id
        if not spectrum_path.is_file():
            matches = list(spectra_dir.rglob(row.lamost_id))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"expected one LAMOST spectrum for {row.lamost_id}, found {len(matches)}"
                )
            spectrum_path = matches[0]
        with fits.open(spectrum_path, memmap=False) as hdus:
            data = np.asarray(hdus[0].data, dtype=float)
            wavelength = data[2]
            flux = data[0]
            ivar = data[1]
            redshift = float(hdus[0].header.get("Z", 0.0))
        resampled_flux, resampled_valid, resampled_ivar = _resample_rest_frame(
            wavelength,
            flux,
            ivar,
            np.asarray(ivar > 0, dtype=bool),
            redshift,
            grid,
        )
        flux_rows.append(resampled_flux)
        valid_rows.append(resampled_valid)
        snr_rows.append(_median_snr(resampled_flux, resampled_ivar, resampled_valid, grid))
        source_files.append(str(spectrum_path))

    source = SurveySpectra(
        survey="LAMOST_DR2_HO2017",
        wavelength=grid,
        flux=np.asarray(flux_rows, dtype=np.float32),
        valid=np.asarray(valid_rows, dtype=bool),
        object_id=label_frame["apogee_id"].to_numpy(dtype=str),
        snr=np.asarray(snr_rows, dtype=float),
        targets={
            name: label_frame[name].to_numpy(dtype=float)
            for name in ("teff", "logg", "feh")
        },
        target_errors={
            "teff": label_frame["teff_err"].to_numpy(dtype=float),
            "logg": label_frame["logg_err"].to_numpy(dtype=float),
            "feh": label_frame["feh_err"].to_numpy(dtype=float),
        },
    )
    manifest = label_frame.copy()
    manifest["source_file"] = source_files
    manifest["contract_snr"] = source.snr
    manifest["valid_fraction"] = source.valid.mean(axis=1)
    manifest["label_scale"] = LABEL_SCALE
    manifest["feh_slot_semantics"] = "APOGEE_DR12_PARAM_M_H"
    return source, manifest


def _apogee_quality_rows(allstar_path: Path, selected_ids: set[str]) -> pd.DataFrame:
    fits = _require_astropy()
    with fits.open(allstar_path, memmap=True) as hdus:
        data = hdus[1].data
        apogee_id = np.char.strip(np.asarray(data["APOGEE_ID"]).astype(str))
        selected = np.isin(apogee_id, list(selected_ids))
        paramflag = np.asarray(data["PARAMFLAG"])[selected].astype(np.int64)
        frame = pd.DataFrame(
            {
                "apogee_id": apogee_id[selected],
                "teff_catalog": _native_float(data["TEFF"][selected]),
                "logg_catalog": _native_float(data["LOGG"][selected]),
                "feh_catalog": _native_float(data["PARAM_M_H"][selected]),
                "teff_err": _native_float(data["TEFF_ERR"][selected]),
                "logg_err": _native_float(data["LOGG_ERR"][selected]),
                "feh_err": _native_float(data["PARAM_M_H_ERR"][selected]),
                "apogee_snr": _native_float(data["SNR"][selected]),
                "aspcapflag": np.asarray(data["ASPCAPFLAG"])[selected].astype(np.int64),
                "starflag": np.asarray(data["STARFLAG"])[selected].astype(np.int64),
            }
        )
    flags = frame["aspcapflag"].to_numpy(dtype=np.int64)
    relevant_aspcap_flag = (
        ((flags & (2**7)) != 0)
        | ((flags & (2**23)) != 0)
        | ((flags & (2**3)) != 0)
        | ((flags & (2**4)) != 0)
    )
    parameter_flag = (
        (paramflag[:, 0] != 0)
        | (paramflag[:, 1] != 0)
        | (paramflag[:, 3] != 0)
        | (paramflag[:, 4] != 0)
    )
    frame["ho2017_badstar"] = (
        ~frame["teff_catalog"].between(4000.0, 6000.0)
        | (frame["logg_catalog"] < 0)
        | relevant_aspcap_flag
        | parameter_flag
    )
    frame["quality_score"] = (
        frame["teff_err"] / 150.0
        + frame["logg_err"] / 0.2
        + frame["feh_err"] / 0.2
        - np.log1p(np.maximum(frame["apogee_snr"], 0)) / 10.0
    )
    return (
        frame.sort_values(
            ["ho2017_badstar", "quality_score", "apogee_snr"],
            ascending=[True, True, False],
        )
        .drop_duplicates("apogee_id")
        .reset_index(drop=True)
    )


def _build_desi_target(
    selected_csv: Path,
    pickle_paths: list[Path],
    redshift_json: Path,
    grid: np.ndarray,
) -> tuple[SurveySpectra, pd.DataFrame]:
    selected = pd.read_csv(selected_csv)
    required = {
        "APOGEE_ID",
        "TEFF",
        "LOGG",
        "PARAM_M_H",
        "TEFF_ERR",
        "LOGG_ERR",
        "PARAM_M_H_ERR",
        "desi_targetid",
    }
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"DESI selection table is missing columns: {missing}")
    records: dict[int, dict[str, object]] = {}
    for path in sorted(pickle_paths):
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if not payload or not payload[0]["status"]["success"]:
            raise ValueError(f"SPARCL retrieval failed in {path}")
        for record in payload[1:]:
            targetid = int(record["targetid"])
            if targetid in records:
                raise ValueError(f"duplicate SPARCL targetid {targetid}")
            records[targetid] = record

    redshift_payload = json.loads(redshift_json.read_text(encoding="utf-8"))
    redshifts = {
        int(record["targetid"]): (
            float(record["redshift"]),
            int(record["redshift_warning"]),
            str(record["sparcl_id"]),
        )
        for record in redshift_payload[1:]
    }
    selected = selected.copy()
    selected["desi_targetid"] = selected["desi_targetid"].astype(np.int64)
    selected = selected.loc[selected["desi_targetid"].isin(records)].copy()
    selected["redshift"] = selected["desi_targetid"].map(
        lambda value: redshifts[int(value)][0]
    )
    selected["redshift_warning"] = selected["desi_targetid"].map(
        lambda value: redshifts[int(value)][1]
    )
    selected["sparcl_id"] = selected["desi_targetid"].map(
        lambda value: redshifts[int(value)][2]
    )
    selected = selected.loc[selected["redshift_warning"] == 0].copy()
    selected = selected.sort_values("APOGEE_ID").reset_index(drop=True)

    wavelength: np.ndarray | None = None
    flux_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    snr_rows: list[float] = []
    for row in selected.itertuples(index=False):
        record = records[int(row.desi_targetid)]
        if wavelength is None:
            wavelength = _desi_wavelength(len(np.asarray(record["flux"])))
        flux = np.asarray(record["flux"], dtype=float)
        ivar = np.asarray(record["ivar"], dtype=float)
        mask = np.asarray(record["mask"])
        valid = (mask == 0) & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
        resampled_flux, resampled_valid, resampled_ivar = _resample_rest_frame(
            wavelength,
            flux,
            ivar,
            valid,
            float(row.redshift),
            grid,
        )
        flux_rows.append(resampled_flux)
        valid_rows.append(resampled_valid)
        snr_rows.append(_median_snr(resampled_flux, resampled_ivar, resampled_valid, grid))

    target = SurveySpectra(
        survey="DESI_DR1_SPARCL",
        wavelength=grid,
        flux=np.asarray(flux_rows, dtype=np.float32),
        valid=np.asarray(valid_rows, dtype=bool),
        object_id=selected["APOGEE_ID"].to_numpy(dtype=str),
        snr=np.asarray(snr_rows, dtype=float),
        targets={
            "teff": selected["TEFF"].to_numpy(dtype=float),
            "logg": selected["LOGG"].to_numpy(dtype=float),
            "feh": selected["PARAM_M_H"].to_numpy(dtype=float),
        },
        target_errors={
            "teff": selected["TEFF_ERR"].to_numpy(dtype=float),
            "logg": selected["LOGG_ERR"].to_numpy(dtype=float),
            "feh": selected["PARAM_M_H_ERR"].to_numpy(dtype=float),
        },
    )
    manifest = selected.copy()
    manifest["contract_snr"] = target.snr
    manifest["valid_fraction"] = target.valid.mean(axis=1)
    manifest["label_scale"] = LABEL_SCALE
    manifest["feh_slot_semantics"] = "APOGEE_DR12_PARAM_M_H"
    return target, manifest


def _desi_wavelength(pixel_count: int) -> np.ndarray:
    if pixel_count != 7781:
        raise ValueError(
            "DESI DR1 SPARCL wavelength reconstruction expects 7,781 BRZ pixels; "
            f"found {pixel_count}"
        )
    return 3600.0 + 0.8 * np.arange(pixel_count, dtype=float)


def _resample_rest_frame(
    wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    valid: np.ndarray,
    redshift: float,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelength = np.asarray(wavelength, dtype=float) / (1.0 + float(redshift))
    flux = np.asarray(flux, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    finite = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(ivar)
    if finite.sum() < 2:
        raise ValueError("spectrum has fewer than two finite pixels")
    wavelength = wavelength[finite]
    flux = flux[finite]
    ivar = ivar[finite]
    valid = valid[finite]
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    flux = flux[order]
    ivar = ivar[order]
    valid = valid[order]
    inside = (grid >= wavelength[0]) & (grid <= wavelength[-1])
    output_flux = np.full(len(grid), np.nan, dtype=np.float32)
    output_ivar = np.zeros(len(grid), dtype=np.float32)
    output_valid = np.zeros(len(grid), dtype=bool)
    output_flux[inside] = np.interp(grid[inside], wavelength, flux).astype(np.float32)
    output_ivar[inside] = np.maximum(
        np.interp(grid[inside], wavelength, ivar), 0.0
    ).astype(np.float32)
    valid_weight = np.interp(grid[inside], wavelength, valid.astype(float))
    output_valid[inside] = (
        (valid_weight > 0.999)
        & np.isfinite(output_flux[inside])
        & (output_ivar[inside] > 0)
    )
    return output_flux, output_valid, output_ivar


def _median_snr(
    flux: np.ndarray,
    ivar: np.ndarray,
    valid: np.ndarray,
    wavelength: np.ndarray,
) -> float:
    window = valid & (wavelength >= 4000.0) & (wavelength <= 5500.0)
    values = np.asarray(flux[window], dtype=float) * np.sqrt(
        np.asarray(ivar[window], dtype=float)
    )
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return 0.0
    return float(np.median(values))


def _native_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(np.float64)
