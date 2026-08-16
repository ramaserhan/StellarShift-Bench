import json
import tempfile
from pathlib import Path

import pandas as pd

from stellar_benchmark.config import SyntheticDemoConfig
from stellar_benchmark.demo import run_synthetic_demo


def test_end_to_end_demo_saves_reproducible_artifacts():
    with tempfile.TemporaryDirectory() as temporary_directory:
        config = SyntheticDemoConfig(
            source_rows=80,
            target_rows=60,
            n_bootstrap=2,
            output_dir=temporary_directory,
        )
        artifacts = run_synthetic_demo(config)

        assert all(path.is_file() for path in artifacts.values())
        comparison = pd.read_csv(artifacts["comparison"])
        assert set(comparison["target"]) == {"teff", "logg", "feh"}
        manifest = json.loads(Path(artifacts["manifest"]).read_text())
        assert manifest["split_unit"] == "source_id"
        assert manifest["partition_rows"]["target_adaptation"] > 0
