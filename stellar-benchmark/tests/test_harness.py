import numpy as np
import pandas as pd

from stellar_benchmark.eval import EvalHarness


def test_stratification_uses_positions_with_non_range_index():
    df = pd.DataFrame(
        {"group": ["cool"] * 5 + ["hot"] * 5},
        index=[100, 101, 102, 103, 104, 200, 201, 202, 203, 204],
    )
    y_true = {"teff": np.arange(10, dtype=float)}
    y_pred = {"teff": np.arange(10, dtype=float) + 1}
    result = EvalHarness(["teff"], strata=["group"]).evaluate(
        df, y_true, y_pred, label="test"
    )

    grouped = result[result["stratum"] == "group"]
    assert set(grouped["group"]) == {"cool", "hot"}
    assert set(grouped["n"]) == {5}


def test_harness_rejects_misaligned_rows():
    df = pd.DataFrame({"group": ["a", "b"]})
    try:
        EvalHarness(["teff"]).evaluate(
            df,
            {"teff": np.array([1.0])},
            {"teff": np.array([1.0])},
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
