import numpy as np
import pandas as pd

from scripts.build_apogee_desi_crossmatch import (
    crossmatch_apogee_to_desi,
    select_source_disjoint_targets,
)


def _apogee_row(object_id, ra, flag=0, snr=100.0):
    return {
        "APOGEE_ID": object_id,
        "RA": ra,
        "DEC": 0.0,
        "TEFF": 4800.0,
        "LOGG": 2.5,
        "PARAM_M_H": -0.5,
        "TEFF_ERR": 50.0,
        "LOGG_ERR": 0.05,
        "PARAM_M_H_ERR": 0.03,
        "SNR": snr,
        "ASPCAPFLAG": flag,
        "STARFLAG": 0,
    }


def _desi_row(targetid, ra):
    return {
        "targetid": targetid,
        "mean_fiber_ra": ra,
        "mean_fiber_dec": 0.0,
        "survey": "main",
        "program": "bright",
        "healpix": 1,
        "tsnr2_gpbbright": 10.0,
        "tsnr2_gpbbackup": 20.0,
        "coadd_exptime": 100.0,
    }


def test_crossmatch_and_selection_flow_are_deterministic():
    apogee = pd.DataFrame(
        [
            _apogee_row("A", 10.0, snr=40.0),
            _apogee_row("A", 10.0, snr=120.0),
            _apogee_row("B", 20.0, flag=8),
            _apogee_row("C", 30.0),
        ]
    )
    desi = pd.DataFrame(
        [
            _desi_row(101, 10.0 + 0.2 / 3600.0),
            _desi_row(102, 20.0 + 0.3 / 3600.0),
            _desi_row(103, 30.0 + 0.4 / 3600.0),
            _desi_row(104, 80.0),
        ]
    )
    crossmatch = crossmatch_apogee_to_desi(apogee, desi)
    assert list(crossmatch["APOGEE_ID"]) == ["A", "B", "C"]
    assert np.allclose(
        crossmatch["separation_arcsec"].to_numpy(), [0.2, 0.3, 0.4], atol=1e-5
    )
    assert crossmatch.loc[crossmatch["APOGEE_ID"] == "A", "SNR"].iloc[0] == 120.0

    selected, flow = select_source_disjoint_targets(crossmatch, {"C"})
    assert list(selected["APOGEE_ID"]) == ["A"]
    assert list(flow["rows"]) == [3, 2, 1]
    assert list(flow["removed_at_stage"]) == [0, 1, 1]
