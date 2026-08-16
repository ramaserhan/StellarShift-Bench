"""Build the deterministic StellarShift-Bench v1.2.3 report and brief."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / "results" / "lamost_dr2_to_desi_dr1_apogee_dr12"
DESI = ROOT / "results" / "desi_reliability_v1"
OUTPUT = ROOT / "output" / "pdf"
FIGURES = ROOT / "output" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

pdfmetrics.registerFont(TTFont("DV", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(
    TTFont("DV-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
)

NAVY = colors.HexColor("#102A43")
INK = colors.HexColor("#243B53")
TEAL = colors.HexColor("#0F8B8D")
CYAN = colors.HexColor("#2CB1BC")
ORANGE = colors.HexColor("#F2994A")
RED = colors.HexColor("#D64545")
GREEN = colors.HexColor("#2F855A")
LIGHT = colors.HexColor("#F3F7FA")
MID = colors.HexColor("#D9E2EC")
MUTED = colors.HexColor("#627D98")
WHITE = colors.white


def invariant_canvas(*args, **kwargs) -> Canvas:
    """Return a ReportLab canvas without clock-dependent PDF metadata."""

    kwargs["invariant"] = 1
    return Canvas(*args, **kwargs)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="DV-Bold", fontSize=28,
            leading=32, textColor=NAVY, alignment=TA_LEFT, spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="DV", fontSize=13,
            leading=18, textColor=MUTED, spaceAfter=6 * mm,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="DV-Bold", fontSize=18,
            leading=22, textColor=NAVY, spaceBefore=2 * mm, spaceAfter=4 * mm,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="DV-Bold", fontSize=11.5,
            leading=15, textColor=TEAL, spaceBefore=3 * mm, spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName="DV", fontSize=9.2,
            leading=13.1, textColor=INK, spaceAfter=2.4 * mm,
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontName="DV", fontSize=7.4,
            leading=9.8, textColor=MUTED, spaceAfter=1.5 * mm,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontName="DV", fontSize=8.9,
            leading=12.3, textColor=INK, leftIndent=5 * mm, firstLineIndent=-3 * mm,
            bulletIndent=1.5 * mm, spaceAfter=1.3 * mm,
        ),
        "card_value": ParagraphStyle(
            "card_value", parent=base["Normal"], fontName="DV-Bold", fontSize=18,
            leading=20, textColor=NAVY, alignment=TA_CENTER,
        ),
        "card_label": ParagraphStyle(
            "card_label", parent=base["Normal"], fontName="DV", fontSize=7.2,
            leading=9.2, textColor=MUTED, alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "callout", parent=base["BodyText"], fontName="DV-Bold", fontSize=10,
            leading=14, textColor=NAVY, leftIndent=5 * mm, rightIndent=5 * mm,
            spaceBefore=2 * mm, spaceAfter=2 * mm,
        ),
        "center": ParagraphStyle(
            "center", parent=base["BodyText"], fontName="DV", fontSize=8.5,
            leading=11, textColor=INK, alignment=TA_CENTER,
        ),
    }


S = make_styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(text, S["bullet"], bulletText="-")


def styled_table(data, widths, font_size=7.8, header=True) -> Table:
    body_style = ParagraphStyle(
        "table_cell_dynamic",
        parent=S["small"],
        fontName="DV",
        fontSize=font_size,
        leading=font_size + 2.0,
        textColor=INK,
        spaceAfter=0,
    )
    wrapped = []
    for row_index, row in enumerate(data):
        if header and row_index == 0:
            wrapped.append(row)
        else:
            wrapped.append([Paragraph(str(cell), body_style) for cell in row])
    result = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("FONTNAME", (0, 0), (-1, -1), "DV"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "DV-Bold"),
        ]
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT))
    result.setStyle(TableStyle(commands))
    return result


def cards(items: list[tuple[str, str]]) -> Table:
    cells = [[p(value, "card_value"), p(label, "card_label")] for value, label in items]
    result = Table([cells], colWidths=[52 * mm] * len(cells))
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.6, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, WHITE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return result


def footer(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    if doc.page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 29 * mm, width, 29 * mm, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.rect(0, 0, width, 7 * mm, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("DV-Bold", 9.5)
        canvas.drawString(21 * mm, height - 17 * mm, "STELLARSHIFT-BENCH / TECHNICAL REPORT")
    else:
        canvas.setStrokeColor(MID)
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, 14 * mm, width - 20 * mm, 14 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("DV", 7.3)
        canvas.drawString(20 * mm, 9 * mm, "StellarShift-Bench v1.2.3 | 16 August 2026")
        canvas.drawRightString(width - 20 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_shift_comparison() -> Path:
    real = pd.read_csv(CROSS / "domain_shift_summary.csv").set_index("target")
    noise = pd.read_csv(DESI / "nested_bootstrap_intervals.csv")
    noise = noise.loc[
        (noise["model"] == "source_only") & np.isclose(noise["noise_factor"], 2.0)
    ].set_index("target")
    labels = ["Teff", "log g", "[M/H]"]
    keys = ["teff", "logg", "feh"]
    synthetic = [noise.loc[key, "mae_change_percent"] for key in keys]
    survey = [real.loc[key, "cross_survey_mae_change_percent"] for key in keys]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=180)
    ax.bar(x - 0.18, synthetic, 0.36, color="#2CB1BC", label="DESI 2x noise")
    ax.bar(x + 0.18, survey, 0.36, color="#F2994A", label="LAMOST to DESI")
    ax.axhline(0, color="#627D98", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE change from in-domain baseline (%)")
    ax.set_title("Real cross-survey shift is much larger than controlled noise")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    for i, value in enumerate(synthetic):
        ax.text(i - 0.18, value + 5, f"{value:.1f}%", ha="center", fontsize=8)
    for i, value in enumerate(survey):
        ax.text(i + 0.18, value + 5, f"{value:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    path = FIGURES / "real_vs_controlled_shift.png"
    fig.savefig(path, bbox_inches="tight", metadata={"Software": "StellarShift-Bench v1.2.3"})
    plt.close(fig)
    return path


def build_support_chart() -> Path:
    domain = pd.read_csv(CROSS / "domain_shift_summary.csv").set_index("target")
    support = pd.read_csv(CROSS / "support_overlap_metrics.csv")
    support = support.loc[support["method"] == "source_only"].set_index("target")
    keys = ["teff", "logg", "feh"]
    labels = ["Teff", "log g", "[M/H]"]
    full = [domain.loc[key, "cross_survey_mae_change_percent"] for key in keys]
    inside = [
        100 * (support.loc[key, "mae"] / domain.loc[key, "source_holdout_mae"] - 1)
        for key in keys
    ]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=180)
    ax.bar(x - 0.18, full, 0.36, color="#D64545", label="Full target (n=1,088)")
    ax.bar(x + 0.18, inside, 0.36, color="#0F8B8D", label="Within source label bounds (n=918)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE change vs LAMOST holdout (%)")
    ax.set_title("Population support explains part, not all, of the shift")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGURES / "support_sensitivity.png"
    fig.savefig(path, bbox_inches="tight", metadata={"Software": "StellarShift-Bench v1.2.3"})
    plt.close(fig)
    return path


def build_adaptation_chart() -> Path:
    effects = pd.read_csv(CROSS / "adaptation_effect_intervals.csv")
    methods = ["coral_unlabeled", "source_plus_target_retrained"]
    labels = ["CORAL (unlabeled)", "Labeled retraining"]
    keys = ["teff", "logg", "feh"]
    target_labels = ["Teff", "log g", "[M/H]"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=180)
    colors_ = ["#F2994A", "#0F8B8D"]
    for index, (method, label) in enumerate(zip(methods, labels)):
        frame = effects.loc[effects["method"] == method].set_index("target")
        values = [frame.loc[key, "relative_difference_percent"] for key in keys]
        ax.bar(x + (index - 0.5) * 0.36, values, 0.36, label=label, color=colors_[index])
        for x_value, value in zip(x + (index - 0.5) * 0.36, values):
            offset = 2.0 if value >= 0 else -5.0
            ax.text(x_value, value + offset, f"{value:+.1f}%", ha="center", fontsize=8)
    ax.axhline(0, color="#627D98", linewidth=0.9)
    ax.set_xticks(x, target_labels)
    ax.set_ylabel("MAE change vs source-only transfer (%)")
    ax.set_title("Adaptation can improve, remain inconclusive, or cause harm")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGURES / "adaptation_relative_effects.png"
    fig.savefig(path, bbox_inches="tight", metadata={"Software": "StellarShift-Bench v1.2.3"})
    plt.close(fig)
    return path


def report_story(shift_chart: Path, support_chart: Path, adaptation_chart: Path) -> list:
    domain = pd.read_csv(CROSS / "domain_shift_summary.csv").set_index("target")
    boot = pd.read_csv(CROSS / "star_bootstrap_intervals.csv")
    effects = pd.read_csv(CROSS / "adaptation_effect_intervals.csv")
    calibration = pd.read_csv(CROSS / "calibration_cross_survey.csv")
    support = pd.read_csv(CROSS / "support_overlap_metrics.csv")
    label_budget = pd.read_csv(CROSS / "label_budget_trials.csv")
    ablation = pd.read_csv(CROSS / "model_ablation.csv")
    reference = pd.read_csv(CROSS / "reference_label_sensitivity.csv")
    story: list = []

    story += [
        Spacer(1, 17 * mm),
        p("StellarShift-Bench", "title"),
        p("Reliability under real cross-survey shift in stellar spectroscopy", "subtitle"),
        cards([
            ("1,088", "model-held-out DESI evaluation stars"),
            ("+108-242%", "source-only MAE degradation"),
            ("51", "passing tests"),
        ]),
        Spacer(1, 7 * mm),
        p(
            "<b>Executive finding.</b> A model that is accurate on held-out LAMOST "
            "spectra degrades sharply on disjoint DESI DR1 spectra even when both "
            "domains use the exact APOGEE DR12 ASPCAP v603 reference-label scale. "
            "The transferred model also becomes overconfident. Labeled target "
            "retraining recovers much of the loss; unlabeled CORAL does not and "
            "materially harms metallicity.",
            "callout",
        ),
        HRFlowable(width="100%", thickness=0.7, color=MID, spaceBefore=3 * mm, spaceAfter=4 * mm),
        p("What changed in v1.2.3", "h2"),
        bullet("Replaced the stale standalone notebook with the release-built version and made every code cell runnable against archived evidence."),
        bullet("Added an independent two-sample star bootstrap for source-to-target MAE change and auditable source-holdout predictions on full reruns."),
        bullet("Added the missing APOGEE-to-DESI crossmatch builder and the complete 4,662,192 to 1,827 to 1,576 to 1,555 selection audit."),
        bullet("Reframed the release as prespecified quality-gated and retained the deterministic v1.2.2 release and Colab safeguards."),
        Spacer(1, 3 * mm),
        p("Rama Serhan | Technical report | 16 August 2026", "small"),
        PageBreak(),
        p("1. Study design and prespecified quality gate", "h1"),
        p(
            "The source consists of public high-S/N LAMOST DR2 optical spectra from "
            "the Ho et al. tutorial. The target consists of APOGEE DR12 stars "
            "position-matched within 1.5 arcsec to primary DESI DR1 spectra and "
            "retrieved through SPARCL. APOGEE_ID is the cross-survey identity key; "
            "source-target overlap is removed before splitting."
        ),
        styled_table(
            [
                ["Partition", "Observed", "Gate", "Purpose"],
                ["LAMOST source train", "1,032", ">=1,000", "Feature/model fitting"],
                ["LAMOST source holdout", "259", ">=200", "In-domain error and calibration"],
                ["DESI target adaptation", "467", ">=100", "Adaptation/recalibration only"],
                ["DESI target evaluation", "1,088", ">=350", "model-held-out final evaluation"],
                ["Evaluation giants", "1,024", ">=50", "Subgroup support"],
                ["Evaluation metal-poor", "55", ">=50", "Out-of-support visibility"],
            ],
            [48 * mm, 27 * mm, 25 * mm, 72 * mm],
        ),
        Spacer(1, 4 * mm),
        p(
            "Version 1.2.3 independently resamples source-holdout and target-evaluation "
            "stars for the MAE-difference interval. The frozen public result directory "
            "predates source per-star prediction export, so its numerical two-sample "
            "interval is produced only when the non-redistributed survey contracts are rerun.",
            "small",
        ),
        Spacer(1, 2 * mm),
        p("Shared contract", "h2"),
        bullet("Exact label scale: APOGEE DR12 ASPCAP v603 calibrated TEFF, LOGG, and PARAM_M_H."),
        bullet("The legacy file field `feh` stores global [M/H], not elemental FE_H."),
        bullet("Rest-frame 4000-5500 A on a common log-wavelength grid; Gaussian resolution matching to R=1800; no sharpening."),
        bullet("Normalization, imputation, outlier thresholds, and 64-component PCA fit on source training only."),
        bullet("Default estimator: separate 400-tree ExtraTrees regressors; target access is tagged for every method."),
        Spacer(1, 3 * mm),
        p(
            "The gate was written before fitting and passed without changing a "
            "threshold after outcomes were observed. Input hashes and query metadata "
            "are frozen in the acquisition manifest.",
            "callout",
        ),
        PageBreak(),
        p("2. Real survey shift versus controlled noise", "h1"),
        Image(str(shift_chart), width=168 * mm, height=86 * mm),
        p(
            "At 2x recorded noise in the controlled DESI experiment, temperature MAE "
            "rose by 29.8%, while gravity and metallicity changed by about 4%. The "
            "real LAMOST-to-DESI penalty is much larger for every target. This is a "
            "descriptive comparison, not a causal claim that the two shifts share a mechanism."
        ),
        styled_table(
            [
                ["Target", "LAMOST MAE", "DESI MAE", "Change", "DESI MAE 95% CI", "DESI R2"],
                ["Teff", "52.5 K", "109.4 K", "+108.4%", "103.2-115.4 K", "0.656"],
                ["log g", "0.124", "0.260", "+110.0%", "0.240-0.279", "0.558"],
                ["[M/H]", "0.073", "0.250", "+241.6%", "0.228-0.270", "0.220"],
            ],
            [27 * mm, 30 * mm, 30 * mm, 27 * mm, 39 * mm, 22 * mm],
        ),
        Spacer(1, 4 * mm),
        p(
            "The target errors are not just noisier: source-only predictions acquire "
            "positive biases of +44.6 K, +0.191 dex in log g, and +0.220 dex in [M/H]. "
            "Operational reliability therefore requires bias and calibration checks, "
            "not only a rank score.",
            "callout",
        ),
        PageBreak(),
        p("3. Adaptation is method- and parameter-dependent", "h1"),
        Image(str(adaptation_chart), width=168 * mm, height=86 * mm),
        styled_table(
            [
                ["Method", "Target access", "Teff MAE", "log g MAE", "[M/H] MAE"],
                ["Source only", "none", "109.4 K", "0.260", "0.250"],
                ["CORAL", "467 unlabeled spectra", "113.5 K", "0.264", "0.393"],
                ["Retraining", "467 labeled stars", "78.4 K", "0.153", "0.100"],
            ],
            [37 * mm, 52 * mm, 29 * mm, 29 * mm, 29 * mm],
        ),
        Spacer(1, 3 * mm),
        bullet("CORAL Teff: +4.0 K paired difference, 95% CI -0.02 to +7.94 K - inconclusive, not equivalent."),
        bullet("CORAL log g: +0.004 dex, 95% CI -0.004 to +0.013 - inconclusive."),
        bullet("CORAL [M/H]: +0.144 dex, 95% CI +0.138 to +0.149 - detectable harm."),
        bullet("Labeled retraining improves all three targets by 28.3%-60.0%, with paired intervals excluding zero."),
        p(
            "This controlled three-way comparison supports a stronger conclusion than "
            "'adaptation helps': target access, parameter, and method determine whether "
            "adaptation helps, does nothing detectable, or causes harm.",
            "callout",
        ),
        PageBreak(),
        p("4. Uncertainty becomes overconfident under shift", "h1"),
        Image(str(CROSS / "calibration_cross_survey.png"), width=168 * mm, height=96 * mm),
        p(
            "Nominal 90% intervals calibrated on LAMOST holdout residuals under-cover "
            "on DESI. The failure is especially severe for source-only [M/H] (59.0%) "
            "and CORAL [M/H] (16.0%). Calibration is therefore a distinct deployment "
            "failure even when point accuracy appears acceptable."
        ),
        styled_table(
            [
                ["Method", "Calibration data", "Teff", "log g", "[M/H]"],
                ["Source only", "LAMOST holdout", "67.8%", "75.2%", "59.0%"],
                ["CORAL", "LAMOST holdout", "59.6%", "70.5%", "16.0%"],
                ["Retraining", "LAMOST holdout", "80.0%", "85.5%", "81.6%"],
                ["Source only", "DESI adaptation", "90.3%", "90.2%", "89.9%"],
                ["CORAL", "DESI adaptation", "90.1%", "89.3%", "89.0%"],
            ],
            [36 * mm, 52 * mm, 28 * mm, 28 * mm, 28 * mm],
        ),
        Spacer(1, 4 * mm),
        p(
            "Target recalibration uses labels from the 467-star adaptation partition "
            "only; no target-evaluation label is touched. Coverage returns close to "
            "nominal, but interval widths expand substantially - honest uncertainty is not free.",
            "callout",
        ),
        PageBreak(),
        p("5. Population support and label efficiency", "h1"),
        Image(str(support_chart), width=168 * mm, height=82 * mm),
        p(
            "The full target contains stellar labels outside the source-training range, "
            "including the metal-poor subgroup. Restricting to 918 stars jointly inside "
            "the source minima and maxima reduces degradation, but leaves a clear "
            "32%-76% penalty. Population extrapolation matters; it does not explain the whole result."
        ),
        p("Target-label budget", "h2"),
        Image(str(CROSS / "label_budget.png"), width=168 * mm, height=83 * mm),
        p(
            "Across ten repeated draws, five target labels increase median error for "
            "every parameter. At 100 labels, median MAE improves by 11.6% for Teff, "
            "31.1% for log g, and 46.0% for [M/H]. Label acquisition needs a budget "
            "curve, not a single convenient operating point.",
            "callout",
        ),
        PageBreak(),
        p("6. Robustness, ablations, and physical checks", "h1"),
        p("Formal reference-label errors", "h2"),
        p(
            "A paired 1,000-replicate sensitivity analysis perturbs APOGEE labels by "
            "their reported independent Gaussian one-sigma errors. Retraining remains "
            "better than source-only in every replicate for all targets; CORAL [M/H] "
            "remains worse in every replicate. Teff and log g CORAL differences remain inconclusive."
        ),
        p("Model-family ablation", "h2"),
        styled_table(
            [
                ["Family", "Target Teff MAE", "Target log g MAE", "Target [M/H] MAE"],
                ["Ridge", "92.7 K", "0.200", "0.164"],
                ["ExtraTrees", "110.6 K", "0.262", "0.251"],
                ["MLP", "130.3 K", "0.248", "0.185"],
            ],
            [45 * mm, 43 * mm, 43 * mm, 43 * mm],
        ),
        p(
            "Ridge has the best absolute target performance in this ablation, while "
            "the small MLP beats ExtraTrees on log g and [M/H] but not Teff. No single "
            "family wins every parameter, and all show a substantial source-to-target penalty."
        ),
        p("OOD and physical plausibility", "h2"),
        bullet("Keeping the 50% least-OOD source-only targets reduces MAE from 109.4 to 79.9 K, 0.260 to 0.163 dex, and 0.250 to 0.140 dex."),
        bullet("No prediction violates the declared hard bounds for temperature, gravity, or metallicity."),
        bullet("Isochrone-manifold consistency is explicitly not evaluated because no cited population-appropriate grid was supplied."),
        p(
            "Formal-error propagation is a sensitivity analysis, not independent "
            "validation of APOGEE. Hard bounds are necessary but weak physical tests; "
            "the unrun isochrone check remains a real limitation.",
            "callout",
        ),
        PageBreak(),
        p("7. Reproducibility, limitations, and claim boundary", "h1"),
        p("Reproducibility contract", "h2"),
        bullet("51 deterministic tests; exact YAML configuration; immutable source and target SHA-256 hashes."),
        bullet("Frozen prespecified quality gate and object-disjoint split manifest."),
        bullet("Per-star predictions, star-bootstrap intervals, paired adaptation effects, calibration, label budgets, support, OOD, subgroups, ablations, and physical checks."),
        bullet("Version-pinned acquisition URLs, Data Lab SQL, SPARCL response, duplicate rule, and public target selection."),
        bullet("Raw survey spectra and processed spectral arrays excluded from the public archive; builders and hashes included."),
        p("Limitations", "h2"),
        bullet("The common APOGEE scale is consistent reference data, not independent absolute truth; shared ASPCAP systematics may remain."),
        bullet("The primary effect combines instrument, reduction, S/N, and population/covariate shift. The 918-star support analysis is narrower, not perfectly causal."),
        bullet("The Ho source is high-S/N and has no stars below [M/H] = -1.5 after quality cuts; the DESI target is giant- and backup-program-dominated."),
        bullet("Resolution matching uses a Gaussian approximation, not exact per-spectrum DESI/LAMOST line-spread matrices."),
        bullet("No B+R+Z fusion, external-label replication, or isochrone-manifold result is claimed."),
        p("Claim boundary", "h2"),
        p(
            "Supported: a large, calibrated, object-disjoint LAMOST DR2 to DESI DR1 "
            "transfer penalty on the APOGEE DR12 scale; method-specific adaptation and "
            "calibration findings; label-budget, support, OOD, subgroup, ablation, "
            "formal-error-sensitivity, and hard-bound results for the frozen data."
        ),
        p(
            "Not supported: pure instrument causality, independent validation of "
            "APOGEE, universal cross-survey generalization, exact LSF conclusions, "
            "isochrone consistency, or a DOI before deposition."
        ),
        p("References and public data", "h2"),
        p(
            "Ho et al. (2017), ApJ 836, 5, doi:10.3847/1538-4357/836/1/5. "
            "The Cannon LAMOST tutorial: annayqho.github.io/TheCannon/lamost_tutorial.html. "
            "APOGEE DR12 allStar-v603: sdss4.org/dr12/data_access/bulk-data-downloads/. "
            "DESI DR1: data.desi.lbl.gov/doc/releases/dr1/. "
            "NOIRLab DESI/SPARCL: datalab.noirlab.edu/data/desi.",
            "small",
        ),
    ]
    return story


def build_report(shift_chart: Path, support_chart: Path, adaptation_chart: Path) -> Path:
    path = OUTPUT / "StellarShift_Bench_v1.2.3_Technical_Report.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=19 * mm, title="StellarShift-Bench v1.2.3 Technical Report",
        author="Rama Serhan",
    )
    doc.build(
        report_story(shift_chart, support_chart, adaptation_chart),
        onFirstPage=footer,
        onLaterPages=footer,
        canvasmaker=invariant_canvas,
    )
    return path


def build_brief(shift_chart: Path) -> Path:
    path = OUTPUT / "StellarShift_Bench_v1.2.3_Portfolio_Brief.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=14 * mm, bottomMargin=13 * mm, title="StellarShift-Bench v1.2.3 Portfolio Brief",
        author="Rama Serhan",
    )
    story = [
        p("StellarShift-Bench v1.2.3", "title"),
        p("A prespecified quality-gated LAMOST DR2 to DESI DR1 reliability benchmark", "subtitle"),
        cards([
            ("1,088", "model-held-out target stars"),
            ("+108-242%", "real-transfer MAE loss"),
            ("28-60%", "retraining recovery"),
        ]),
        Spacer(1, 4 * mm),
        Table(
            [[
                Image(str(shift_chart), width=104 * mm, height=53 * mm),
                [
                    p("The result", "h2"),
                    p("A LAMOST-trained model roughly doubles Teff/log g error and more than triples [M/H] error on DESI."),
                    p("Source-calibrated nominal 90% coverage collapses to 68%, 75%, and 59%. Target recalibration restores near-nominal coverage."),
                    p("CORAL harms [M/H] by 57.5%; labeled retraining improves all targets by 28%-60%."),
                ],
            ]],
            colWidths=[108 * mm, 70 * mm],
            style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2)]),
        ),
        Spacer(1, 3 * mm),
        styled_table(
            [
                ["Reliability layer", "What was built", "Why it matters"],
                ["Data", "Public LAMOST/APOGEE/DESI acquisition, exact hashes, object IDs", "Reproducible and leakage-safe"],
                ["Statistics", "Per-star predictions, bootstrap CIs, paired method effects", "Effect size, not p-value theater"],
                ["Uncertainty", "Split conformal under shift and target recalibration", "Detects deployment overconfidence"],
                ["Adaptation", "Source-only, CORAL, retraining, 5-100 label budgets", "Shows benefit, harm, and label efficiency"],
                ["Robustness", "Support, OOD, subgroups, model families, label errors", "Tests where conclusions survive"],
                ["Delivery", "51 tests, deterministic release, CLI, manifests, report, notebook", "Research-grade engineering"],
            ],
            [34 * mm, 76 * mm, 68 * mm],
            font_size=7.2,
        ),
        Spacer(1, 3 * mm),
        Table(
            [[
                [
                    p("Scientific judgment", "h2"),
                    bullet("The full effect mixes survey and population shift; a 918-star within-support audit still shows 32%-76% degradation."),
                    bullet("The common APOGEE scale controls label version, but is not independent truth."),
                    bullet("Hard bounds pass; isochrone consistency remains honestly unclaimed."),
                ],
                [
                    p("Role signal", "h2"),
                    bullet("Research: falsifiable estimands, negative results, limitations."),
                    bullet("ML: calibration, OOD, adaptation, ablation, label efficiency."),
                    bullet("Engineering: typed contracts, public APIs, tests, provenance, archival release."),
                ],
            ]],
            colWidths=[89 * mm, 89 * mm],
            style=TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.5, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, WHITE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        Spacer(1, 3 * mm),
        p(
            "Deliverables: technical report, runnable results notebook, source release, evidence bundle, frozen acquisition manifest, and machine-readable per-star results.",
            "callout",
        ),
        p("Rama Serhan | StellarShift-Bench v1.2.3 | 16 August 2026", "small"),
    ]
    doc.build(story, canvasmaker=invariant_canvas)
    return path


if __name__ == "__main__":
    shift = build_shift_comparison()
    support = build_support_chart()
    adaptation = build_adaptation_chart()
    print(build_report(shift, support, adaptation))
    print(build_brief(shift))
