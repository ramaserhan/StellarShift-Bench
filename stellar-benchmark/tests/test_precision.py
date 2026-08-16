import numpy as np
import pandas as pd

from stellar_benchmark.eval.precision import paired_adaptation_effects


def test_paired_effects_preserve_pairing_and_reject_false_equivalence():
    rng = np.random.default_rng(9)
    n = 80
    baseline = np.abs(rng.normal(10.0, 2.0, n))
    better = baseline - rng.normal(1.5, 0.4, n)
    uncertain = baseline + rng.normal(0.0, 5.0, n)
    frames = []
    for method, errors in {
        "source_only": baseline,
        "better": better,
        "uncertain": uncertain,
    }.items():
        frames.append(
            pd.DataFrame(
                {
                    "object_id": np.arange(n),
                    "method": method,
                    "target": "teff",
                    "abs_error": errors,
                }
            )
        )
    effects, planning = paired_adaptation_effects(
        pd.concat(frames, ignore_index=True),
        bootstrap_replicates=500,
        random_state=2,
    )

    better_row = effects.loc[effects.method == "better"].iloc[0]
    uncertain_row = effects.loc[effects.method == "uncertain"].iloc[0]
    assert better_row.paired_difference_ci_upper < 0
    assert uncertain_row.conclusion == "inconclusive_not_equivalent"
    assert (planning.required_n_for_two_sided_detection >= 2).all()


def test_paired_effects_requires_expected_columns():
    frame = pd.DataFrame({"method": ["source_only"]})
    try:
        paired_adaptation_effects(frame)
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("missing columns should fail")
