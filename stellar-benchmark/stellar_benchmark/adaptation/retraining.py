"""Supervised target-domain retraining baselines.

These strategies fit a new model from scratch. They are intentionally not
called fine-tuning: the current scikit-learn baselines do not update an
existing model's learned weights.
"""

from __future__ import annotations

import numpy as np

from ..models.baseline import BaselineRegressor


def retrain_with_target_data(
    source_trained_model: BaselineRegressor,
    X_target_train: np.ndarray,
    Y_target_train: np.ndarray,
    strategy: str = "source_plus_target",
    X_source: np.ndarray | None = None,
    Y_source: np.ndarray | None = None,
) -> BaselineRegressor:
    """Fit a new baseline using labeled target-domain data.

    ``source_plus_target`` trains on the original source sample plus a disjoint
    target adaptation sample. ``target_only`` uses only the labeled target
    sample. The source-trained model is never mutated, preserving a valid
    matched pre-adaptation comparator.
    """

    adapted = source_trained_model.clone_untrained()

    if strategy == "source_plus_target":
        if X_source is None or Y_source is None:
            raise ValueError("source_plus_target requires X_source and Y_source")
        X_combined = np.concatenate([X_source, X_target_train], axis=0)
        Y_combined = np.concatenate([Y_source, Y_target_train], axis=0)
        adapted.fit(X_combined, Y_combined)
    elif strategy == "target_only":
        adapted.fit(X_target_train, Y_target_train)
    else:
        raise ValueError(f"unknown retraining strategy: {strategy!r}")

    return adapted
