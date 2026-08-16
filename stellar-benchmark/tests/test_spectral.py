import numpy as np

from stellar_benchmark.models.spectral import (
    SeparateExtraTreesRegressor,
    SpectralFeaturePipeline,
)


def test_spectral_pipeline_and_separate_models():
    rng = np.random.default_rng(8)
    wavelength = np.linspace(5800, 6000, 81)
    flux = rng.normal(0, 0.1, size=(40, 81)).astype(np.float32)
    valid = np.ones_like(flux, dtype=bool)
    valid[0, 10] = False
    flux[0, 10] = np.nan

    pipeline = SpectralFeaturePipeline(
        wavelength_min=5820,
        wavelength_max=5980,
        pca_components=8,
        random_state=9,
    )
    features = pipeline.fit_transform(flux[:30], valid[:30], wavelength)
    held_out, _ = pipeline.transform(flux[30:], valid[30:])
    assert features.shape == (30, 8)
    assert held_out.shape == (10, 8)
    assert 0 < pipeline.explained_variance_ratio <= 1

    targets = {
        "teff": np.linspace(4000, 6000, 30),
        "logg": np.linspace(2, 5, 30),
        "feh": np.linspace(-2, 0.3, 30),
    }
    model = SeparateExtraTreesRegressor(n_estimators=5).fit(features, targets)
    predictions = model.predict(held_out)
    assert set(predictions) == set(targets)
    assert all(len(values) == 10 for values in predictions.values())
