import json

import numpy as np
import pandas as pd

from stellar_benchmark.config import DesiSNRConfig
from stellar_benchmark.desi_experiment import run_desi_snr_experiment


def test_desi_experiment_saves_auditable_artifacts(tmp_path):
    rng = np.random.default_rng(11)
    n, pixels = 120, 101
    wavelength = np.linspace(5800, 5880, pixels)
    teff = np.linspace(3400, 7000, n)
    logg = rng.uniform(2.5, 5.0, n)
    feh = rng.uniform(-2.0, 0.3, n)
    x = np.linspace(-1, 1, pixels)
    flux = (
        100
        + 4 * x[None, :]
        - ((teff - 5000) / 1000)[:, None]
        * np.exp(-((x[None, :] + 0.2) / 0.08) ** 2)
        - logg[:, None] * np.exp(-((x[None, :] - 0.3) / 0.06) ** 2)
        + rng.normal(0, 0.4, size=(n, pixels))
    ).astype(np.float32)
    valid = np.ones_like(flux, dtype=bool)
    input_path = tmp_path / "fixture.npz"
    np.savez_compressed(
        input_path,
        wavelength=wavelength,
        flux=flux,
        ivar=np.full_like(flux, 6.25),
        valid=valid,
        valid_fraction=valid.mean(axis=1),
        targetid=np.arange(2000, 2000 + n, dtype=np.int64),
        teff=teff.astype(np.float32),
        logg=logg.astype(np.float32),
        feh=feh.astype(np.float32),
        teff_err=np.full(n, 20, dtype=np.float32),
        logg_err=np.full(n, 0.05, dtype=np.float32),
        feh_err=np.full(n, 0.05, dtype=np.float32),
        sn_r=np.linspace(11, 60, n).astype(np.float32),
        source_file=np.full(n, "fixture.fits", dtype="U32"),
    )
    output_dir = tmp_path / "results"
    config = DesiSNRConfig(
        input_npz=str(input_path),
        output_dir=str(output_dir),
        continuum_window=21,
        continuum_polyorder=3,
        wavelength_min=5800,
        wavelength_max=5880,
        pca_components=10,
        n_estimators=5,
        noise_factors=[1.5],
        noise_seed_count=2,
        adaptation_noise_factor=1.5,
        augmentation_views=1,
    )
    artifacts = run_desi_snr_experiment(config)
    assert all(path.is_file() for path in artifacts.values())
    summary = pd.read_csv(artifacts["snr_summary"])
    assert set(summary["model"]) == {"source_only", "noise_augmented"}
    manifest = json.loads(artifacts["manifest"].read_text())
    assert manifest["split_unit"] == "TARGETID"
    assert manifest["partition_rows"]["target_evaluation"] > 0
