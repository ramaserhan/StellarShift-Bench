import numpy as np
import pandas as pd

from stellar_benchmark.config import DesiSNRConfig
from stellar_benchmark.data.desi import (
    assert_no_targetid_leakage,
    continuum_normalize,
    create_split_manifest,
    inject_noise,
)


def _dataset(n=120, pixels=101):
    rng = np.random.default_rng(3)
    wavelength = np.linspace(5800, 5880, pixels)
    flux = 100 + rng.normal(0, 0.5, size=(n, pixels))
    valid = np.ones_like(flux, dtype=bool)
    ivar = np.full_like(flux, 4.0)
    teff = np.linspace(3400, 7000, n)
    return {
        "wavelength": wavelength,
        "flux": flux.astype(np.float32),
        "ivar": ivar.astype(np.float32),
        "valid": valid,
        "valid_fraction": valid.mean(axis=1),
        "targetid": np.arange(1000, 1000 + n, dtype=np.int64),
        "teff": teff.astype(np.float32),
        "logg": rng.uniform(2.5, 5.0, n).astype(np.float32),
        "feh": rng.uniform(-2.0, 0.3, n).astype(np.float32),
        "teff_err": np.full(n, 20, dtype=np.float32),
        "logg_err": np.full(n, 0.05, dtype=np.float32),
        "feh_err": np.full(n, 0.05, dtype=np.float32),
        "sn_r": np.linspace(11, 60, n).astype(np.float32),
        "source_file": np.full(n, "fixture.fits", dtype="U32"),
    }


def test_manifest_is_deterministic_and_targetid_disjoint():
    data = _dataset()
    config = DesiSNRConfig(continuum_window=21, pca_components=10)
    first = create_split_manifest(data, config)
    second = create_split_manifest(data, config)
    pd.testing.assert_frame_equal(first, second)
    assert_no_targetid_leakage(first)
    assert set(first["split"]) == {
        "source_train",
        "source_holdout",
        "target_adaptation",
        "target_evaluation",
    }


def test_continuum_normalization_preserves_mask():
    data = _dataset(n=4)
    data["valid"][0, 20] = False
    normalized, valid = continuum_normalize(
        data["flux"], data["valid"], window_length=21, polyorder=3
    )
    assert normalized.shape == data["flux"].shape
    assert not valid[0, 20]
    assert np.isnan(normalized[0, 20])
    assert np.isfinite(normalized[valid]).all()


def test_noise_injection_is_order_independent_and_factor_one_is_identity():
    data = _dataset(n=5)
    identity = inject_noise(
        data["flux"],
        data["ivar"],
        data["valid"],
        data["targetid"],
        factor=1.0,
        seed=7,
    )
    np.testing.assert_allclose(identity, data["flux"])

    shifted = inject_noise(
        data["flux"],
        data["ivar"],
        data["valid"],
        data["targetid"],
        factor=2.0,
        seed=7,
    )
    order = np.array([3, 0, 4, 1, 2])
    reordered = inject_noise(
        data["flux"][order],
        data["ivar"][order],
        data["valid"][order],
        data["targetid"][order],
        factor=2.0,
        seed=7,
    )
    inverse = np.argsort(order)
    np.testing.assert_allclose(shifted, reordered[inverse])
