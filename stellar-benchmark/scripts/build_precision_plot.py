"""Build the paired temperature adaptation-effect interval figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "desi_reliability_v1"


def build() -> Path:
    effects = pd.read_csv(RESULTS / "adaptation_effect_intervals.csv")
    effects = effects.loc[effects["target"] == "teff"].copy()
    order = [
        "noise_augmented",
        "coral_unlabeled",
        "source_plus_target_retrained",
    ]
    effects["method"] = pd.Categorical(effects["method"], order, ordered=True)
    effects = effects.sort_values("method")
    labels = ["Noise augmentation", "CORAL", "Source + 90 labels"]
    estimates = effects["paired_mae_difference"].to_numpy()
    lower = effects["paired_difference_ci_lower"].to_numpy()
    upper = effects["paired_difference_ci_upper"].to_numpy()
    margin = float(effects["planning_margin_absolute"].iloc[0])

    figure, axis = plt.subplots(figsize=(8.6, 3.6))
    axis.axvspan(-margin, margin, color="#D9E4E8", alpha=0.62, zorder=0)
    axis.axvline(0, color="#193246", linewidth=1.2, zorder=1)
    colors = ["#159A9C", "#E98A35", "#3DB7D3"]
    positions = list(range(len(effects)))[::-1]
    for y, estimate, lo, hi, color in zip(
        positions, estimates, lower, upper, colors, strict=True
    ):
        axis.errorbar(
            estimate,
            y,
            xerr=[[estimate - lo], [hi - estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=4,
            markersize=7,
            zorder=3,
        )
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Paired change in temperature MAE (K); negative is better")
    axis.set_title("Adaptation effects: estimates and paired 95% bootstrap intervals")
    figure.text(
        0.99,
        0.018,
        f"Gray band: +/-5% planning margin (+/-{margin:.1f} K), not a preregistered equivalence test",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#5F7382",
    )
    axis.grid(axis="x", color="#D9E4E8", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    output = RESULTS / "adaptation_effect_intervals.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    print(build())
