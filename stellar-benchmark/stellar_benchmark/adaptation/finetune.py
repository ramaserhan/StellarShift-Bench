"""Backward-compatible wrapper for the renamed retraining baseline."""

from __future__ import annotations

import warnings

import numpy as np

from ..models.baseline import BaselineRegressor
from .retraining import retrain_with_target_data

def finetune_on_target(
    source_trained_model: BaselineRegressor,
    X_finetune: np.ndarray,
    Y_finetune: np.ndarray,
    strategy: str = "retrain_on_combined",
    X_source: np.ndarray | None = None,
    Y_source: np.ndarray | None = None,
) -> BaselineRegressor:
    """Deprecated alias; use :func:`retrain_with_target_data`."""

    warnings.warn(
        "finetune_on_target retrains a new model and is deprecated; use "
        "retrain_with_target_data",
        DeprecationWarning,
        stacklevel=2,
    )
    strategy_map = {
        "retrain_on_combined": "source_plus_target",
        "target_only": "target_only",
    }
    if strategy not in strategy_map:
        raise ValueError(f"unknown legacy strategy: {strategy!r}")
    return retrain_with_target_data(
        source_trained_model,
        X_finetune,
        Y_finetune,
        strategy=strategy_map[strategy],
        X_source=X_source,
        Y_source=Y_source,
    )
