"""Build a deterministic, runnable notebook with verified v1.2.3 outputs."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / "results" / "lamost_dr2_to_desi_dr1_apogee_dr12"
DESI = ROOT / "results" / "desi_reliability_v1"
FIGURES = ROOT / "output" / "figures"


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def executed_code(source: str, text: str = "", image_path: Path | None = None) -> dict:
    outputs: list[dict] = []
    if text:
        outputs.append(
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in text.rstrip().splitlines()],
            }
        )
    if image_path is not None:
        outputs.append(
            {
                "data": {
                    "image/png": base64.b64encode(image_path.read_bytes()).decode(),
                    "text/plain": [f"<Figure: {image_path.name}>"],
                },
                "metadata": {},
                "output_type": "display_data",
            }
        )
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


gate = json.loads((CROSS / "publication_gate.json").read_text())
gate_table = pd.DataFrame(gate["criteria"])[
    ["criterion", "observed", "required", "passed"]
]
domain = pd.read_csv(CROSS / "domain_shift_summary.csv")
domain_display = domain[
    [
        "target",
        "source_holdout_mae",
        "target_evaluation_mae",
        "cross_survey_mae_change_percent",
        "target_evaluation_r2",
    ]
].round(3)
metrics = pd.read_csv(CROSS / "transfer_metrics.csv")
methods = metrics.loc[
    metrics["evaluation"] == "target_evaluation_cross_survey",
    ["method", "target_access", "target", "mae", "r2", "mae_change_vs_source_percent"],
].round(3)
calibration = pd.read_csv(CROSS / "calibration_cross_survey.csv")
calibration_display = calibration[
    ["method", "calibration_source", "target", "coverage", "mean_width"]
].round(3)
support = pd.read_csv(CROSS / "support_overlap_metrics.csv")
support_display = support.loc[
    support["method"] == "source_only",
    ["target", "supported_stars", "retained_fraction", "mae", "r2"],
].round(3)
effects = pd.read_csv(CROSS / "adaptation_effect_intervals.csv")
effects_display = effects[
    [
        "method",
        "target",
        "paired_mae_difference",
        "paired_difference_ci_lower",
        "paired_difference_ci_upper",
        "conclusion",
    ]
].round(3)
budgets = pd.read_csv(CROSS / "label_budget_trials.csv")
budget_display = (
    budgets.groupby(["budget", "target"], as_index=False)["mae_change_vs_source_percent"]
    .median()
    .pivot(index="budget", columns="target", values="mae_change_vs_source_percent")
    .round(2)
)
reference = pd.read_csv(CROSS / "reference_label_sensitivity.csv")
reference_display = reference[
    [
        "method",
        "target",
        "sensitivity_difference_median",
        "sensitivity_difference_ci_lower",
        "sensitivity_difference_ci_upper",
        "fraction_difference_below_zero",
    ]
].round(3)
ablation = pd.read_csv(CROSS / "model_ablation.csv")
ablation_display = ablation.loc[
    ablation["evaluation"] == "target_evaluation_cross_survey",
    ["family", "target", "mae", "r2", "mae_change_vs_source_domain_percent"],
].round(3)
noise = pd.read_csv(DESI / "nested_bootstrap_intervals.csv")
noise_display = noise.loc[
    (noise["model"] == "source_only") & (noise["noise_factor"] == 2.0),
    ["target", "clean_mae", "mae", "mae_change_percent", "change_ci_lower", "change_ci_upper"],
].round(3)

NOTEBOOK_SETUP = r'''
from pathlib import Path
import json

import pandas as pd
from IPython.display import Image, display

working_directory = Path.cwd().resolve()
candidates = [working_directory, working_directory / "stellar-benchmark", *working_directory.parents]
ROOT = next(
    (
        candidate
        for candidate in candidates
        if (candidate / "results" / "lamost_dr2_to_desi_dr1_apogee_dr12").is_dir()
    ),
    None,
)
if ROOT is None:
    raise FileNotFoundError(
        "Run this notebook from the extracted stellar-benchmark source release."
    )

CROSS = ROOT / "results" / "lamost_dr2_to_desi_dr1_apogee_dr12"
DESI = ROOT / "results" / "desi_reliability_v1"
FIGURES = ROOT / "output" / "figures"

gate = json.loads((CROSS / "publication_gate.json").read_text())
gate_table = pd.DataFrame(gate["criteria"])[
    ["criterion", "observed", "required", "passed"]
]
domain_shift = pd.read_csv(CROSS / "domain_shift_summary.csv")[[
    "target", "source_holdout_mae", "target_evaluation_mae",
    "cross_survey_mae_change_percent", "target_evaluation_r2",
]].round(3)
transfer_metrics = pd.read_csv(CROSS / "transfer_metrics.csv")
method_metrics = transfer_metrics.loc[
    transfer_metrics["evaluation"] == "target_evaluation_cross_survey",
    ["method", "target_access", "target", "mae", "r2", "mae_change_vs_source_percent"],
].round(3)
calibration_rows = pd.read_csv(CROSS / "calibration_cross_survey.csv")
calibration = calibration_rows[[
    "method", "calibration_source", "target", "coverage", "mean_width"
]].round(3)
support_rows = pd.read_csv(CROSS / "support_overlap_metrics.csv")
within_support = support_rows.loc[
    support_rows["method"] == "source_only",
    ["target", "supported_stars", "retained_fraction", "mae", "r2"],
].round(3)
effect_rows = pd.read_csv(CROSS / "adaptation_effect_intervals.csv")
paired_effects = effect_rows[[
    "method", "target", "paired_mae_difference", "paired_difference_ci_lower",
    "paired_difference_ci_upper", "conclusion",
]].round(3)
budget_rows = pd.read_csv(CROSS / "label_budget_trials.csv")
label_budget_medians = (
    budget_rows.groupby(["budget", "target"], as_index=False)["mae_change_vs_source_percent"]
    .median()
    .pivot(index="budget", columns="target", values="mae_change_vs_source_percent")
    .round(2)
)
reference_rows = pd.read_csv(CROSS / "reference_label_sensitivity.csv")
reference_sensitivity = reference_rows[[
    "method", "target", "sensitivity_difference_median",
    "sensitivity_difference_ci_lower", "sensitivity_difference_ci_upper",
    "fraction_difference_below_zero",
]].round(3)
ablation_rows = pd.read_csv(CROSS / "model_ablation.csv")
target_ablation = ablation_rows.loc[
    ablation_rows["evaluation"] == "target_evaluation_cross_survey",
    ["family", "target", "mae", "r2", "mae_change_vs_source_domain_percent"],
].round(3)
noise_rows = pd.read_csv(DESI / "nested_bootstrap_intervals.csv")
noise_2x = noise_rows.loc[
    (noise_rows["model"] == "source_only") & (noise_rows["noise_factor"] == 2.0),
    ["target", "clean_mae", "mae", "mae_change_percent", "change_ci_lower", "change_ci_upper"],
].round(3)

print(f"Loaded verified v1.2.3 evidence from {ROOT}")
'''


cells = [
    markdown(
        """
# StellarShift-Bench v1.2.3 - instant verified results

This notebook is useful **before any cell is run** because verified outputs are
embedded, and it is also fully runnable from the extracted source release. Its
data-loading cells read the archived LAMOST DR2 to DESI DR1 evidence. The same
harness also provides the controlled DESI S/N comparison.

Scientific status: real cross-survey optical transfer; 1,088 model-held-out DESI
evaluation stars; exact APOGEE DR12 ASPCAP v603 reference-label scale; object-
disjoint splits. The internal `feh` field contains APOGEE `PARAM_M_H` ([M/H]),
not elemental `FE_H`.
"""
    ),
    markdown("## Load the archived evidence"),
    executed_code(NOTEBOOK_SETUP, "Loaded verified v1.2.3 evidence"),
    markdown("## Prespecified quality gate - passed before fitting"),
    executed_code(
        "gate_table",
        gate_table.to_string(index=False),
    ),
    markdown(
        """
## Headline: real cross-survey shift is much larger than controlled noise

Source-only MAE roughly doubles for temperature and gravity and more than
triples for [M/H]. The earlier 2x-noise DESI result is retained as a descriptive
measurement-shift comparison, not a causal decomposition.
"""
    ),
    executed_code(
        "domain_shift",
        domain_display.to_string(index=False),
    ),
    executed_code(
        "display(Image(filename=str(FIGURES / 'real_vs_controlled_shift.png')))",
        image_path=FIGURES / "real_vs_controlled_shift.png",
    ),
    markdown("### Controlled DESI 2x-noise reference"),
    executed_code(
        "noise_2x",
        noise_display.to_string(index=False),
    ),
    markdown(
        """
## Three-way adaptation comparison

CORAL uses target spectra without labels. Retraining uses only the disjoint
target-adaptation labels. The same 1,088 evaluation stars are used for every
method. CORAL is inconclusive for Teff/log g and detectably harms [M/H]; labeled
retraining improves all three targets.
"""
    ),
    executed_code(
        "method_metrics",
        methods.to_string(index=False),
    ),
    executed_code(
        "display(Image(filename=str(FIGURES / 'adaptation_relative_effects.png')))",
        image_path=FIGURES / "adaptation_relative_effects.png",
    ),
    markdown("### Paired same-star adaptation effects"),
    executed_code(
        "paired_effects",
        effects_display.to_string(index=False),
    ),
    markdown(
        """
## Calibration fails under real shift

Nominal 90% intervals calibrated on LAMOST under-cover on DESI. Recalibration
using the 467-star target-adaptation partition restores approximately nominal
coverage without touching target evaluation labels.
"""
    ),
    executed_code(
        "display(Image(filename=str(CROSS / 'calibration_cross_survey.png')))",
        image_path=CROSS / "calibration_cross_survey.png",
    ),
    executed_code(
        "calibration",
        calibration_display.to_string(index=False),
    ),
    markdown(
        """
## Population support explains part, not all, of the loss

The full effect combines survey and population shift. A second analysis keeps
the 918 target stars jointly within source-training Teff/log g/[M/H] minima and
maxima. Degradation remains substantial.
"""
    ),
    executed_code(
        "display(Image(filename=str(FIGURES / 'support_sensitivity.png')))",
        image_path=FIGURES / "support_sensitivity.png",
    ),
    executed_code(
        "within_support",
        support_display.to_string(index=False),
    ),
    markdown(
        """
## Target-label budgets: small samples can hurt

Ten repeated draws at each budget show that five labels raise median error for
all three targets. At 100 labels, median improvements appear across all targets.
"""
    ),
    executed_code(
        "display(Image(filename=str(CROSS / 'label_budget.png')))",
        image_path=CROSS / "label_budget.png",
    ),
    executed_code(
        "# Median MAE change (%) across ten random draws\nlabel_budget_medians",
        budget_display.to_string(),
    ),
    markdown(
        """
## Formal APOGEE label-error sensitivity

The simulation perturbs labels by reported independent Gaussian formal errors
while preserving the same perturbed truth across methods. It does not validate
APOGEE or address shared pipeline systematics. Retraining remains better in all
replicates; CORAL [M/H] remains worse in all replicates.
"""
    ),
    executed_code(
        "reference_sensitivity",
        reference_display.to_string(index=False),
    ),
    markdown("## Model-family ablation"),
    executed_code(
        "target_ablation",
        ablation_display.to_string(index=False),
    ),
    markdown(
        """
## Reproduce the benchmark

```bash
python -m pip install -e ".[dev,desi]"
python -m pytest -q

# Public acquisition and contract commands are documented in data/README.md
stellar-benchmark cross-survey-run --config configs/lamost_to_desi.yaml
```

The evidence bundle includes the exact configuration, acquisition manifest,
input hashes, prespecified quality gate, split manifest, per-star predictions,
calibration, paired effects, support sensitivity, label budgets, OOD/subgroup
tables, ablations, physical checks, report, and this notebook. Raw survey spectra
are not redistributed.

### Claim boundary

Supported: the frozen LAMOST DR2 to DESI DR1 transfer and reliability findings
on the common APOGEE DR12 scale. Not supported: independent APOGEE validation,
pure instrument causality, universal survey/population generalization, exact
line-spread modeling, B+R+Z fusion, or isochrone-manifold consistency.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = ROOT / "examples" / "StellarShift_v1.2.3_Instant_Results.ipynb"
destination.write_text(
    json.dumps(notebook, indent=1, sort_keys=True) + "\n", encoding="utf-8"
)
print(destination)
