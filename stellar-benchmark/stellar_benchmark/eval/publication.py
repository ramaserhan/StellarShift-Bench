"""Evidence gates that separate executable demos from publishable survey runs."""

from __future__ import annotations

import numpy as np


def cross_survey_publication_gate(
    *,
    source_train_targets: dict[str, np.ndarray],
    source_holdout_count: int,
    target_adaptation_count: int,
    target_evaluation_targets: dict[str, np.ndarray],
    source_label_scale: str,
    target_label_scale: str,
    minimum_source_train: int,
    minimum_source_holdout: int,
    minimum_target_adaptation: int,
    minimum_target_evaluation: int,
    minimum_giants: int,
    minimum_metal_poor: int,
) -> dict[str, object]:
    """Evaluate predeclared adequacy criteria for a real cross-survey claim."""

    placeholders = ("unspecified", "unknown", "tbd", "to_be_pinned")
    source_scale = source_label_scale.strip()
    target_scale = target_label_scale.strip()
    label_scale_declared = bool(source_scale and target_scale) and not any(
        token in source_scale.lower() or token in target_scale.lower()
        for token in placeholders
    )
    label_scale_matched = label_scale_declared and source_scale == target_scale

    source_train_count = len(np.asarray(source_train_targets["teff"]))
    target_teff = np.asarray(target_evaluation_targets["teff"], dtype=float)
    target_logg = np.asarray(target_evaluation_targets["logg"], dtype=float)
    target_feh = np.asarray(target_evaluation_targets["feh"], dtype=float)
    target_evaluation_count = len(target_teff)
    giant_count = int(np.sum(target_logg < 3.5))
    metal_poor_count = int(np.sum(target_feh < -1.5))

    criteria = [
        {
            "criterion": "shared_reference_label_scale_declared",
            "observed": f"{source_scale} | {target_scale}",
            "required": "same exact, version-pinned scale for both surveys",
            "passed": label_scale_matched,
        },
        {
            "criterion": "source_train_stars",
            "observed": source_train_count,
            "required": minimum_source_train,
            "passed": source_train_count >= minimum_source_train,
        },
        {
            "criterion": "source_holdout_stars",
            "observed": source_holdout_count,
            "required": minimum_source_holdout,
            "passed": source_holdout_count >= minimum_source_holdout,
        },
        {
            "criterion": "target_adaptation_stars",
            "observed": target_adaptation_count,
            "required": minimum_target_adaptation,
            "passed": target_adaptation_count >= minimum_target_adaptation,
        },
        {
            "criterion": "target_evaluation_stars",
            "observed": target_evaluation_count,
            "required": minimum_target_evaluation,
            "passed": target_evaluation_count >= minimum_target_evaluation,
        },
        {
            "criterion": "target_evaluation_giants_logg_lt_3p5",
            "observed": giant_count,
            "required": minimum_giants,
            "passed": giant_count >= minimum_giants,
        },
        {
            "criterion": "target_evaluation_metal_poor_feh_lt_minus_1p5",
            "observed": metal_poor_count,
            "required": minimum_metal_poor,
            "passed": metal_poor_count >= minimum_metal_poor,
        },
    ]
    blockers = [
        str(item["criterion"]) for item in criteria if not bool(item["passed"])
    ]
    return {
        "passed": not blockers,
        "criteria": criteria,
        "blockers": blockers,
        "status_if_failed": "engineering_run_only_no_publishable_cross_survey_claim",
        "target_evaluation_minimum_basis": (
            "rounded above the 344-star prospective requirement estimated for "
            "detecting a 5% Teff MAE effect for CORAL at 80% power in the DESI "
            "case study"
        ),
    }
