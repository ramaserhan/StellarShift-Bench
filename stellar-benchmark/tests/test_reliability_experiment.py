from pathlib import Path

import numpy as np
import pandas as pd

from stellar_benchmark.config import DesiReliabilityConfig
from stellar_benchmark.reliability_experiment import run_desi_reliability_experiment


def test_reliability_experiment_writes_per_star_and_access_audits(tmp_path):
    rng = np.random.default_rng(23)
    n, pixels = 120, 101
    wavelength = np.linspace(5800, 5880, pixels)
    teff = np.linspace(3400, 7000, n)
    logg = rng.uniform(2.5, 5.0, n)
    feh = rng.uniform(-2.0, 0.3, n)
    x = np.linspace(-1, 1, pixels)
    flux = (
        100
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
        targetid=np.arange(5000, 5000 + n, dtype=np.int64),
        teff=teff.astype(np.float32),
        logg=logg.astype(np.float32),
        feh=feh.astype(np.float32),
        teff_err=np.full(n, 20, dtype=np.float32),
        logg_err=np.full(n, 0.05, dtype=np.float32),
        feh_err=np.full(n, 0.05, dtype=np.float32),
        sn_r=np.linspace(11, 60, n).astype(np.float32),
        source_file=np.full(n, "fixture.fits", dtype="U32"),
    )
    config = DesiReliabilityConfig(
        input_npz=str(input_path),
        output_dir=str(tmp_path / "results"),
        continuum_window=21,
        wavelength_min=5800,
        wavelength_max=5880,
        pca_components=10,
        n_estimators=5,
        noise_factors=[2.0],
        noise_seed_count=1,
        augmentation_views=1,
        bootstrap_replicates=20,
        label_budgets=[5],
        label_budget_repeats=1,
        ablation_models=["ridge"],
        representation_components=[5, 10],
        ablation_estimators=5,
        subgroup_minimum=5,
    )
    artifacts = run_desi_reliability_experiment(config)
    assert all(Path(path).is_file() for path in artifacts.values())
    predictions = pd.read_csv(artifacts["shift_predictions"])
    assert {"object_id", "seed", "pi90_lower", "pi90_upper"} <= set(predictions)
    adaptation = pd.read_csv(artifacts["adaptation"])
    assert set(adaptation["target_access"]) >= {
        "source_only",
        "unlabeled_target_features",
        "labeled_target",
    }

