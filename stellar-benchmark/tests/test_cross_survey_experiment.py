from pathlib import Path

import numpy as np
import pandas as pd

from stellar_benchmark.config import CrossSurveyConfig
from stellar_benchmark.cross_survey_experiment import run_cross_survey_experiment
from stellar_benchmark.data.cross_survey import SurveySpectra, save_survey_npz


def _fixture(rng, n, wavelength, object_ids, survey, offset):
    teff = rng.uniform(3500, 7000, n)
    logg = rng.uniform(2.2, 5.0, n)
    feh = rng.uniform(-2.2, 0.3, n)
    flux = np.ones((n, len(wavelength)))
    for center, width, amplitude in (
        (4300, 10, (teff - 5000) / 5000),
        (4861, 8, logg / 20),
        (5170, 12, (feh + 2.5) / 8),
    ):
        flux -= amplitude[:, None] * np.exp(
            -0.5 * ((wavelength[None, :] - center - offset) / width) ** 2
        )
    flux += rng.normal(0, 0.01, flux.shape)
    return SurveySpectra(
        survey,
        wavelength,
        flux.astype(np.float32),
        np.ones_like(flux, dtype=bool),
        object_ids,
        np.full(n, 40.0),
        {"teff": teff, "logg": logg, "feh": feh},
    )


def test_cross_survey_runner_keeps_access_levels_and_evaluation_disjoint(tmp_path):
    rng = np.random.default_rng(14)
    source = _fixture(
        rng, 90, np.linspace(3900, 5600, 500), np.arange(1000, 1090), "A", 0
    )
    target = _fixture(
        rng, 60, np.linspace(3950, 5550, 550), np.arange(3000, 3060), "B", 1.0
    )
    source_path = save_survey_npz(source, tmp_path / "source.npz")
    target_path = save_survey_npz(target, tmp_path / "target.npz")
    config = CrossSurveyConfig(
        source_npz=str(source_path),
        target_npz=str(target_path),
        output_dir=str(tmp_path / "results"),
        source_survey="A",
        target_survey="B",
        source_resolving_power=2000,
        target_resolving_power=2500,
        common_resolving_power=1800,
        wavelength_min=4000,
        wavelength_max=5500,
        log_wavelength_step=0.001,
        continuum_window=21,
        pca_components=10,
        n_estimators=5,
        label_budgets=[5],
        label_budget_repeats=1,
        bootstrap_replicates=20,
        subgroup_minimum=5,
        ablation_models=["ridge"],
        ablation_estimators=5,
    )
    artifacts = run_cross_survey_experiment(config)
    assert all(Path(path).is_file() for path in artifacts.values())
    metrics = pd.read_csv(artifacts["metrics"])
    assert set(metrics["target_access"].dropna()) >= {
        "source_only",
        "unlabeled_target_features",
        "labeled_target",
    }
    split = pd.read_csv(artifacts["split_manifest"])
    source_ids = set(split.loc[split["split"].str.startswith("source"), "object_id"])
    target_ids = set(split.loc[split["split"].str.startswith("target"), "object_id"])
    assert source_ids.isdisjoint(target_ids)
