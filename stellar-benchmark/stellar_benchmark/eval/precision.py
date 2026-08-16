"""Paired adaptation effects and prospective precision planning.

The functions in this module deliberately separate two questions:

1. What range of paired MAE differences is compatible with the held-out data?
2. How large should a future evaluation set be to resolve a prespecified effect?

The second calculation is a design aid based on the observed paired-error
standard deviation.  It is not retrospective evidence that a null result is
true and is never labelled as post-hoc "power" for the completed test.
"""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
from scipy.stats import norm


def paired_adaptation_effects(
    predictions: pd.DataFrame,
    *,
    baseline: str = "source_only",
    bootstrap_replicates: int = 5_000,
    confidence: float = 0.95,
    planning_margin_fraction: float = 0.05,
    planning_power: float = 0.80,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired effect intervals and a prospective sample-size plan.

    Effects are defined as ``method absolute error - baseline absolute error``;
    negative values favour the method.  The bootstrap resamples held-out stars,
    preserving the paired comparison.  Planning calculations use a two-sided
    normal approximation and the observed standard deviation of paired errors.
    """

    required = {"object_id", "method", "target", "abs_error"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"adaptation predictions are missing columns: {missing}")
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if not 0 < planning_margin_fraction < 1:
        raise ValueError("planning_margin_fraction must lie between zero and one")
    if not 0 < planning_power < 1:
        raise ValueError("planning_power must lie strictly between zero and one")

    rng = np.random.default_rng(random_state)
    alpha = 1.0 - confidence
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_power = float(norm.ppf(planning_power))
    effect_rows: list[dict[str, object]] = []
    planning_rows: list[dict[str, object]] = []

    for target, target_rows in predictions.groupby("target", sort=True):
        wide = target_rows.pivot(
            index="object_id", columns="method", values="abs_error"
        )
        if baseline not in wide:
            raise ValueError(f"baseline method {baseline!r} is absent for {target}")
        baseline_errors = wide[baseline].to_numpy(dtype=float)
        baseline_mae = float(np.mean(baseline_errors))
        planning_margin = planning_margin_fraction * baseline_mae

        for method in sorted(column for column in wide.columns if column != baseline):
            paired = wide[[baseline, method]].dropna()
            if len(paired) < 2:
                raise ValueError(
                    f"at least two paired stars are required for {method}/{target}"
                )
            baseline_current = paired[baseline].to_numpy(dtype=float)
            method_current = paired[method].to_numpy(dtype=float)
            differences = method_current - baseline_current
            n = len(differences)
            estimate = float(np.mean(differences))
            observed_sd = float(np.std(differences, ddof=1))
            indices = rng.integers(0, n, size=(bootstrap_replicates, n))
            bootstrap_means = np.mean(differences[indices], axis=1)
            ci_lower, ci_upper = np.quantile(
                bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0]
            )
            baseline_mean_boot = np.mean(baseline_current[indices], axis=1)
            relative_boot = 100.0 * bootstrap_means / baseline_mean_boot
            relative_lower, relative_upper = np.quantile(
                relative_boot, [alpha / 2.0, 1.0 - alpha / 2.0]
            )

            if ci_upper < 0:
                conclusion = "detectable_improvement"
            elif ci_lower > 0:
                conclusion = "detectable_harm"
            else:
                conclusion = "inconclusive_not_equivalent"

            if ci_upper < -planning_margin:
                margin_relation = "ci_beyond_better_planning_margin"
            elif ci_lower > planning_margin:
                margin_relation = "ci_beyond_worse_planning_margin"
            elif ci_lower >= -planning_margin and ci_upper <= planning_margin:
                margin_relation = "ci_inside_margin_not_confirmatory_equivalence"
            else:
                margin_relation = "ci_crosses_planning_margin_boundary"

            method_mae = float(np.mean(method_current))
            effect_rows.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "target": target,
                    "n_paired_stars": n,
                    "baseline_mae": baseline_mae,
                    "method_mae": method_mae,
                    "paired_mae_difference": estimate,
                    "paired_difference_ci_lower": float(ci_lower),
                    "paired_difference_ci_upper": float(ci_upper),
                    "relative_difference_percent": 100.0 * estimate / baseline_mae,
                    "relative_ci_lower_percent": float(relative_lower),
                    "relative_ci_upper_percent": float(relative_upper),
                    "bootstrap_fraction_below_zero": float(
                        np.mean(bootstrap_means < 0)
                    ),
                    "planning_margin_fraction": planning_margin_fraction,
                    "planning_margin_absolute": planning_margin,
                    "conclusion": conclusion,
                    "planning_margin_relation": margin_relation,
                    "confidence": confidence,
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_unit": "object_id_paired",
                }
            )

            mde = (z_alpha + z_power) * observed_sd / np.sqrt(n)
            required_n = max(
                2,
                ceil(
                    ((z_alpha + z_power) * observed_sd / planning_margin) ** 2
                ),
            )
            ci_half_width = z_alpha * observed_sd / np.sqrt(n)
            required_n_precision = max(
                2, ceil((z_alpha * observed_sd / planning_margin) ** 2)
            )
            planning_rows.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "target": target,
                    "current_n": n,
                    "paired_difference_sd": observed_sd,
                    "current_approx_ci_half_width": ci_half_width,
                    "current_minimum_detectable_difference": mde,
                    "current_mde_percent_of_baseline_mae": 100.0 * mde / baseline_mae,
                    "planning_effect_fraction": planning_margin_fraction,
                    "planning_effect_absolute": planning_margin,
                    "required_n_for_two_sided_detection": required_n,
                    "required_n_for_ci_half_width": required_n_precision,
                    "alpha_two_sided": alpha,
                    "planning_power": planning_power,
                    "planning_basis": "observed_paired_error_sd_normal_approximation",
                    "interpretation": "prospective_design_aid_not_retrospective_evidence",
                }
            )

    return pd.DataFrame(effect_rows), pd.DataFrame(planning_rows)
