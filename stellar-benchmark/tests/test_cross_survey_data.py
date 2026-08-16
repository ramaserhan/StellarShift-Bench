import numpy as np

from stellar_benchmark.data.cross_survey import (
    SurveySpectra,
    harmonize_spectra,
    load_survey_npz,
    save_survey_npz,
    shared_log_wavelength_grid,
)


def test_cross_survey_round_trip_and_harmonization(tmp_path):
    wavelength = np.linspace(4000, 5500, 300)
    flux = np.vstack(
        [
            1 - 0.2 * np.exp(-0.5 * ((wavelength - center) / 8) ** 2)
            for center in (4500, 4800, 5200)
        ]
    )
    dataset = SurveySpectra(
        survey="fixture",
        wavelength=wavelength,
        flux=flux,
        valid=np.ones_like(flux, dtype=bool),
        object_id=np.arange(3),
        snr=np.full(3, 50.0),
        targets={
            "teff": np.array([4500, 5000, 5500]),
            "logg": np.array([2.5, 4.0, 4.5]),
            "feh": np.array([-1.0, -0.5, 0.0]),
        },
        target_errors={
            "teff": np.full(3, 50.0),
            "logg": np.full(3, 0.1),
            "feh": np.full(3, 0.05),
        },
    )
    path = save_survey_npz(dataset, tmp_path / "survey.npz")
    restored = load_survey_npz(path)
    assert restored.survey == "fixture"
    assert restored.target_errors is not None
    assert np.allclose(restored.target_errors["teff"], 50.0)
    grid = shared_log_wavelength_grid(4100, 5400, 5e-4)
    aligned, valid = harmonize_spectra(
        restored.wavelength,
        restored.flux,
        restored.valid,
        grid,
        input_resolving_power=3000,
        target_resolving_power=1800,
    )
    assert aligned.shape == valid.shape == (3, len(grid))
    assert valid.mean() > 0.9


def test_harmonization_refuses_spectral_sharpening():
    wavelength = np.linspace(4000, 5000, 30)
    flux = np.ones((2, 30))
    try:
        harmonize_spectra(
            wavelength,
            flux,
            np.ones_like(flux, dtype=bool),
            wavelength,
            input_resolving_power=1800,
            target_resolving_power=3000,
        )
    except ValueError as error:
        assert "cannot sharpen" in str(error)
    else:
        raise AssertionError("expected a resolution error")
