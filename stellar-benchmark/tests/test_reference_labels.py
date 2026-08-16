import numpy as np
import pandas as pd

from stellar_benchmark.eval.reference_labels import (
    formal_label_error_sensitivity,
    formal_label_error_transfer_sensitivity,
)


def test_formal_label_error_sensitivity_keeps_large_shift_positive():
    n = 40
    object_id = np.arange(n)
    truth = np.linspace(0.0, 1.0, n)
    clean = pd.DataFrame(
        {
            "object_id": object_id,
            "model": "source_only",
            "target": "teff",
            "y_true": truth,
            "y_pred": truth + 0.05,
            "noise_factor": 1.0,
            "seed": -1,
        }
    )
    shifted = pd.concat(
        [
            pd.DataFrame(
                {
                    "object_id": object_id,
                    "model": "source_only",
                    "target": "teff",
                    "y_true": truth,
                    "y_pred": truth + 0.20 + seed * 0.005,
                    "noise_factor": 2.0,
                    "seed": seed,
                }
            )
            for seed in range(4)
        ],
        ignore_index=True,
    )
    errors = pd.DataFrame({"object_id": object_id, "teff_err": 0.01})
    result = formal_label_error_sensitivity(
        clean, shifted, errors, replicates=300, random_state=5
    )
    assert result.loc[0, "sensitivity_change_ci_lower"] > 0
    assert result.loc[0, "scope"].startswith("formal_error_sensitivity")


def test_formal_label_error_transfer_sensitivity_preserves_pairing():
    n = 50
    object_id = np.arange(n)
    truth = np.linspace(0.0, 1.0, n)
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "object_id": object_id,
                    "method": method,
                    "target": "teff",
                    "y_true": truth,
                    "y_pred": truth + error,
                }
            )
            for method, error in (("source_only", 0.2), ("adapted", 0.05))
        ],
        ignore_index=True,
    )
    errors = pd.DataFrame({"object_id": object_id, "teff_err": 0.01})
    result = formal_label_error_transfer_sensitivity(
        predictions, errors, replicates=300, random_state=8
    )
    adapted = result.loc[result["method"] == "adapted"].iloc[0]
    assert adapted["sensitivity_difference_ci_upper"] < 0
    assert adapted["fraction_difference_below_zero"] == 1.0
