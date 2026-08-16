"""The core evaluation harness.

This is the main deliverable of the project: a single, reusable object that
takes predictions for one or more stellar parameters and produces a
consistent report -- in-domain metrics, cross-domain metrics, stratified
breakdowns, and (optionally) a matched baseline-vs-adaptation comparison with
a significance test. Every experiment in examples/ routes through this class
so results are always comparable to each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M


@dataclass
class EvalHarness:
    target_columns: list[str]
    # Column names in the dataframe passed to `stratify` used for breakdowns,
    # e.g. ["snr_bin", "stellar_type"]. Must already exist in the dataframe.
    strata: list[str] = field(default_factory=list)

    def evaluate(
        self,
        df: pd.DataFrame,
        y_true: dict[str, np.ndarray],
        y_pred: dict[str, np.ndarray],
        y_pred_std: dict[str, np.ndarray] | None = None,
        label: str = "eval",
    ) -> pd.DataFrame:
        """Evaluate predictions for every target parameter, overall and
        stratified by each column in `self.strata`. Returns a tidy dataframe:
        one row per (target, stratum_name, stratum_value).
        """
        rows = []
        for target in self.target_columns:
            if target not in y_true or target not in y_pred:
                raise KeyError(f"missing predictions or labels for target {target!r}")
            yt = np.asarray(y_true[target]).reshape(-1)
            yp = np.asarray(y_pred[target]).reshape(-1)
            if len(df) != len(yt) or len(yt) != len(yp):
                raise ValueError(
                    f"dataframe, labels, and predictions must align for target {target!r}"
                )
            std = None
            if y_pred_std is not None and target in y_pred_std:
                std = np.asarray(y_pred_std[target]).reshape(-1)

            overall = M.summary_metrics(yt, yp, std)
            rows.append({"label": label, "target": target, "stratum": "overall", "group": "all", **overall})

            for stratum_col in self.strata:
                if stratum_col not in df.columns:
                    continue
                # ``indices`` returns row positions. ``groups`` returns index
                # labels, which breaks whenever df has a non-RangeIndex.
                grouped_positions = df.groupby(
                    stratum_col, observed=True, dropna=False
                ).indices
                for group_val, idx in grouped_positions.items():
                    idx = np.asarray(idx, dtype=int)
                    if len(idx) < 5:
                        continue  # too few stars in this bin to trust the metric
                    m = M.summary_metrics(
                        yt[idx], yp[idx], None if std is None else std[idx]
                    )
                    rows.append(
                        {
                            "label": label,
                            "target": target,
                            "stratum": stratum_col,
                            "group": str(group_val),
                            **m,
                        }
                    )
        return pd.DataFrame(rows)

    def compare_matched(
        self,
        y_true: dict[str, np.ndarray],
        y_pred_before: dict[str, np.ndarray],
        y_pred_after: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Produce a true matched baseline-vs-adaptation comparison on the
        SAME held-out stars, including a paired significance test. This is
        the exact artifact that most published cross-survey studies are
        missing (see Figure 6 in the accompanying literature review: 0 of 28
        audited records had a fully matched before/after pair).
        """
        rows = []
        for target in self.target_columns:
            yt = np.asarray(y_true[target])
            before = np.asarray(y_pred_before[target])
            after = np.asarray(y_pred_after[target])

            m_before = M.summary_metrics(yt, before)
            m_after = M.summary_metrics(yt, after)
            p_value = M.paired_significance(before - yt, after - yt)

            rows.append(
                {
                    "target": target,
                    "mae_before": m_before["mae"],
                    "mae_after": m_after["mae"],
                    "rmse_before": m_before["rmse"],
                    "rmse_after": m_after["rmse"],
                    "bias_before": m_before["bias"],
                    "bias_after": m_after["bias"],
                    "scatter_before": m_before["scatter"],
                    "scatter_after": m_after["scatter"],
                    "r2_before": m_before["r2"],
                    "r2_after": m_after["r2"],
                    "scatter_improved": m_after["scatter"] < m_before["scatter"],
                    "wilcoxon_p": p_value,
                    "significant_at_0.05": p_value < 0.05,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def report(df_metrics: pd.DataFrame) -> str:
        """Render a short human-readable summary of an `evaluate()` result."""
        lines = []
        overall = df_metrics[df_metrics["stratum"] == "overall"]
        for _, row in overall.iterrows():
            lines.append(
                f"[{row['label']}] {row['target']}: "
                f"MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, "
                f"bias={row['bias']:.3f}, scatter={row['scatter']:.3f}, "
                f"R2={row['r2']:.3f}, n={row['n']}"
            )
        return "\n".join(lines)
