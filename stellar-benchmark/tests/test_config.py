import tempfile
from pathlib import Path

from stellar_benchmark.config import (
    load_cross_survey_config,
    load_desi_reliability_config,
    load_desi_snr_config,
    load_synthetic_demo_config,
)


def test_yaml_config_loads_and_rejects_unknown_keys():
    with tempfile.TemporaryDirectory() as temporary_directory:
        valid_path = Path(temporary_directory) / "valid.yaml"
        valid_path.write_text("name: test_run\nsource_rows: 80\ntarget_rows: 60\n")
        config = load_synthetic_demo_config(valid_path)
        assert config.name == "test_run"
        assert config.source_rows == 80

        invalid_path = Path(temporary_directory) / "invalid.yaml"
        invalid_path.write_text("unexpected_option: true\n")
        try:
            load_synthetic_demo_config(invalid_path)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_desi_yaml_config_loads_and_validates(tmp_path):
    path = tmp_path / "desi.yaml"
    path.write_text(
        "input_npz: sample.npz\ncontinuum_window: 101\nnoise_seed_count: 3\n"
    )
    config = load_desi_snr_config(path)
    assert config.input_npz == "sample.npz"
    assert config.continuum_window == 101
    assert config.noise_seed_count == 3


def test_v1_configs_load():
    reliability = load_desi_reliability_config("configs/desi_reliability.yaml")
    cross = load_cross_survey_config("configs/lamost_to_desi.yaml")
    assert reliability.bootstrap_replicates == 1000
    assert cross.source_survey == "LAMOST_DR2_HO2017"
    assert cross.target_survey == "DESI_DR1_SPARCL"
    assert cross.source_label_scale == cross.target_label_scale
