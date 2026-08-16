"""Rebuild the frozen APOGEE DR12 to DESI DR1 positional selection.

The script consumes the two CSV files produced by the SQL printed by
``acquire_real_cross_survey.py catalog-queries``, the APOGEE DR12 allStar-v603
catalog, and the Ho et al. LAMOST tutorial label table. It writes both the full
nearest-neighbour crossmatch and the source-disjoint target selection used for
SPARCL retrieval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


APOGEE_COLUMNS = (
    "APOGEE_ID",
    "RA",
    "DEC",
    "TEFF",
    "LOGG",
    "PARAM_M_H",
    "TEFF_ERR",
    "LOGG_ERR",
    "PARAM_M_H_ERR",
    "SNR",
    "ASPCAPFLAG",
    "STARFLAG",
)
DESI_COLUMNS = (
    "targetid",
    "mean_fiber_ra",
    "mean_fiber_dec",
    "survey",
    "program",
    "healpix",
    "tsnr2_gpbbright",
    "tsnr2_gpbbackup",
    "coadd_exptime",
)


def _unit_vectors(ra_degrees: np.ndarray, dec_degrees: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_degrees, dtype=float))
    dec = np.deg2rad(np.asarray(dec_degrees, dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack(
        [cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)]
    )


def _quality_score(frame: pd.DataFrame) -> np.ndarray:
    """Deterministic ranking for duplicate APOGEE catalog entries."""

    return (
        frame["TEFF_ERR"].to_numpy(dtype=float) / 150.0
        + frame["LOGG_ERR"].to_numpy(dtype=float) / 0.2
        + frame["PARAM_M_H_ERR"].to_numpy(dtype=float) / 0.2
        - np.log1p(np.maximum(frame["SNR"].to_numpy(dtype=float), 0.0)) / 10.0
        + np.log1p(np.abs(frame["ASPCAPFLAG"].to_numpy(dtype=np.int64)))
        + np.log1p(np.abs(frame["STARFLAG"].to_numpy(dtype=np.int64)))
    )


def prepare_apogee_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(APOGEE_COLUMNS) - set(frame))
    if missing:
        raise ValueError(f"APOGEE catalog is missing columns: {missing}")
    result = frame.loc[:, APOGEE_COLUMNS].copy()
    result["APOGEE_ID"] = result["APOGEE_ID"].astype(str).str.strip()
    numeric = [name for name in APOGEE_COLUMNS if name != "APOGEE_ID"]
    for name in numeric:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    finite_columns = [
        "RA",
        "DEC",
        "TEFF",
        "LOGG",
        "PARAM_M_H",
        "TEFF_ERR",
        "LOGG_ERR",
        "PARAM_M_H_ERR",
        "SNR",
    ]
    finite = np.isfinite(result[finite_columns].to_numpy(dtype=float)).all(axis=1)
    positive_errors = (result[["TEFF_ERR", "LOGG_ERR", "PARAM_M_H_ERR"]] > 0).all(
        axis=1
    )
    result = result.loc[
        finite & positive_errors & result["APOGEE_ID"].ne("")
    ].copy()
    result["ASPCAPFLAG"] = result["ASPCAPFLAG"].astype(np.int64)
    result["STARFLAG"] = result["STARFLAG"].astype(np.int64)
    result["quality_score"] = _quality_score(result)
    return (
        result.sort_values(
            ["APOGEE_ID", "quality_score", "SNR"],
            ascending=[True, True, False],
        )
        .drop_duplicates("APOGEE_ID", keep="first")
        .reset_index(drop=True)
    )


def prepare_desi_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(DESI_COLUMNS) - set(frame))
    if missing:
        raise ValueError(f"DESI catalog is missing columns: {missing}")
    result = frame.loc[:, DESI_COLUMNS].copy()
    result["targetid"] = pd.to_numeric(result["targetid"], errors="raise").astype(
        np.int64
    )
    for name in ("mean_fiber_ra", "mean_fiber_dec"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    finite = np.isfinite(
        result[["mean_fiber_ra", "mean_fiber_dec"]].to_numpy(dtype=float)
    ).all(axis=1)
    return (
        result.loc[finite]
        .drop_duplicates("targetid", keep="first")
        .reset_index(drop=True)
    )


def crossmatch_apogee_to_desi(
    apogee: pd.DataFrame,
    desi: pd.DataFrame,
    radius_arcsec: float = 1.5,
) -> pd.DataFrame:
    """Return the nearest DESI primary spectrum for each APOGEE object."""

    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    apogee = prepare_apogee_catalog(apogee)
    desi = prepare_desi_catalog(desi)
    tree = cKDTree(_unit_vectors(desi["mean_fiber_ra"], desi["mean_fiber_dec"]))
    angular_radius = np.deg2rad(radius_arcsec / 3600.0)
    chord_radius = 2.0 * np.sin(angular_radius / 2.0)
    chord, index = tree.query(
        _unit_vectors(apogee["RA"], apogee["DEC"]),
        k=1,
        distance_upper_bound=chord_radius,
    )
    matched = np.isfinite(chord) & (index < len(desi))
    left = apogee.loc[matched].reset_index(drop=True)
    right = desi.iloc[index[matched]].reset_index(drop=True)
    separation = 2.0 * np.arcsin(np.clip(chord[matched] / 2.0, 0.0, 1.0))
    output = left.copy()
    rename = {
        "targetid": "desi_targetid",
        "mean_fiber_ra": "desi_mean_fiber_ra",
        "mean_fiber_dec": "desi_mean_fiber_dec",
        "survey": "desi_survey",
        "program": "desi_program",
        "healpix": "desi_healpix",
        "tsnr2_gpbbright": "desi_tsnr2_gpbbright",
        "tsnr2_gpbbackup": "desi_tsnr2_gpbbackup",
        "coadd_exptime": "desi_coadd_exptime",
    }
    for source, destination in rename.items():
        output[destination] = right[source].to_numpy()
    output["separation_arcsec"] = np.rad2deg(separation) * 3600.0
    return (
        output.sort_values(
            ["desi_targetid", "separation_arcsec", "quality_score", "SNR"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates("desi_targetid", keep="first")
        .sort_values("separation_arcsec")
        .reset_index(drop=True)
    )


def select_source_disjoint_targets(
    crossmatch: pd.DataFrame,
    source_apogee_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen target-label and cross-survey identity contract."""

    source_ids = {str(value).strip() for value in source_apogee_ids}
    clean_labels = crossmatch.loc[crossmatch["ASPCAPFLAG"] == 0].copy()
    selected = clean_labels.loc[
        ~clean_labels["APOGEE_ID"].astype(str).str.strip().isin(source_ids)
    ].copy()
    selected = selected.sort_values(
        ["PARAM_M_H", "APOGEE_ID"], ascending=[True, True]
    ).reset_index(drop=True)
    flow = pd.DataFrame(
        [
            {
                "stage": "apogee_desi_matches_within_1p5_arcsec",
                "rows": len(crossmatch),
                "removed_at_stage": 0,
            },
            {
                "stage": "apogee_aspcapflag_equals_zero",
                "rows": len(clean_labels),
                "removed_at_stage": len(crossmatch) - len(clean_labels),
            },
            {
                "stage": "source_apogee_id_overlap_removed",
                "rows": len(selected),
                "removed_at_stage": len(clean_labels) - len(selected),
            },
        ]
    )
    return selected, flow


def _native_endian(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype.byteorder not in {"=", "|"}:
        result = result.byteswap().view(result.dtype.newbyteorder("="))
    return result


def _load_apogee(path: Path) -> pd.DataFrame:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("install stellar-benchmark[desi] to read APOGEE FITS") from exc
    with fits.open(path, memmap=True) as hdus:
        data = hdus[1].data
        return pd.DataFrame(
            {
                name: np.char.strip(np.asarray(data[name]).astype(str))
                if name == "APOGEE_ID"
                else _native_endian(data[name])
                for name in APOGEE_COLUMNS
            }
        )


def _load_source_ids(path: Path) -> set[str]:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("install stellar-benchmark[desi] to read LAMOST labels") from exc
    data = fits.getdata(path, 1)
    return set(np.char.strip(np.asarray(data["APOGEE_ID"]).astype(str)).tolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apogee-allstar", required=True, type=Path)
    parser.add_argument("--desi-catalog", required=True, nargs="+", type=Path)
    parser.add_argument("--lamost-labels", required=True, type=Path)
    parser.add_argument("--crossmatch-output", required=True, type=Path)
    parser.add_argument("--selection-output", required=True, type=Path)
    parser.add_argument("--flow-output", required=True, type=Path)
    parser.add_argument("--radius-arcsec", type=float, default=1.5)
    args = parser.parse_args()

    apogee = _load_apogee(args.apogee_allstar)
    desi = pd.concat(
        [pd.read_csv(path, usecols=DESI_COLUMNS) for path in args.desi_catalog],
        ignore_index=True,
    )
    crossmatch = crossmatch_apogee_to_desi(
        apogee, desi, radius_arcsec=args.radius_arcsec
    )
    selected, flow = select_source_disjoint_targets(
        crossmatch, _load_source_ids(args.lamost_labels)
    )
    for path in (args.crossmatch_output, args.selection_output, args.flow_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    crossmatch.to_csv(args.crossmatch_output, index=False)
    selected.to_csv(args.selection_output, index=False)
    flow.to_csv(args.flow_output, index=False)
    print(flow.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
