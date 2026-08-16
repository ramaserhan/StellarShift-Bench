import numpy as np

from stellar_benchmark.eval.publication import cross_survey_publication_gate


def _targets(n, giants=0, metal_poor=0):
    logg = np.full(n, 4.5)
    feh = np.full(n, -0.3)
    logg[:giants] = 2.5
    feh[:metal_poor] = -2.0
    return {"teff": np.full(n, 5000.0), "logg": logg, "feh": feh}


def test_publication_gate_blocks_placeholder_labels_and_thin_subgroups():
    result = cross_survey_publication_gate(
        source_train_targets=_targets(1200),
        source_holdout_count=250,
        target_adaptation_count=120,
        target_evaluation_targets=_targets(400, giants=10, metal_poor=5),
        source_label_scale="TO_BE_PINNED",
        target_label_scale="TO_BE_PINNED",
        minimum_source_train=1000,
        minimum_source_holdout=200,
        minimum_target_adaptation=100,
        minimum_target_evaluation=350,
        minimum_giants=50,
        minimum_metal_poor=50,
    )
    assert not result["passed"]
    assert "shared_reference_label_scale_declared" in result["blockers"]
    assert "target_evaluation_giants_logg_lt_3p5" in result["blockers"]


def test_publication_gate_passes_declared_adequate_design():
    result = cross_survey_publication_gate(
        source_train_targets=_targets(1200),
        source_holdout_count=250,
        target_adaptation_count=120,
        target_evaluation_targets=_targets(400, giants=60, metal_poor=55),
        source_label_scale="APOGEE_DR17_ASPCAP_vac",
        target_label_scale="APOGEE_DR17_ASPCAP_vac",
        minimum_source_train=1000,
        minimum_source_holdout=200,
        minimum_target_adaptation=100,
        minimum_target_evaluation=350,
        minimum_giants=50,
        minimum_metal_poor=50,
    )
    assert result["passed"]
    assert not result["blockers"]
