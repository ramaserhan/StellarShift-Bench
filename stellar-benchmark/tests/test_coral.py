import numpy as np

from stellar_benchmark.adaptation.coral import CORALAdapter


def test_coral_aligns_source_mean_and_covariance():
    rng = np.random.default_rng(12)
    source = rng.normal(size=(800, 5))
    target = rng.normal(size=(900, 5)) @ np.diag([2.0, 0.5, 1.5, 0.8, 1.2]) + 3
    aligned = CORALAdapter(regularization=1e-6).fit(source, target).transform_source(source)
    np.testing.assert_allclose(aligned.mean(axis=0), target.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(
        np.cov(aligned, rowvar=False), np.cov(target, rowvar=False), atol=3e-2
    )


def test_coral_rejects_mismatched_feature_width():
    source = np.ones((5, 3))
    target = np.ones((5, 4))
    try:
        CORALAdapter().fit(source, target)
    except ValueError as error:
        assert "same feature width" in str(error)
    else:
        raise AssertionError("expected a feature-width error")
