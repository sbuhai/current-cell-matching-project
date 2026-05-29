"""Write frozen robustness add-on tables used in the manuscript.

The numbers in this file are finalized summaries from the quantitative pipeline.
The goal of this helper is to regenerate the CSV/TeX artifacts that are directly
consumed by the paper, not to resolve the model again.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


CALIBRATION_ROWS = [
    {
        "label": "Baseline",
        "support_share": 0.747199,
        "natural_log_ratio": -0.004806434838929238,
        "natural_lower": -0.007987661747986098,
        "natural_upper": -0.0029572028662497017,
        "age_log_ratio": 0.006970626865011109,
        "age_lower": -0.0096683360726671,
        "age_upper": 0.0788560447616172,
    },
    {
        "label": r"More patient: $\beta=0.97$",
        "support_share": 0.8631,
        "natural_log_ratio": -0.00137,
        "natural_lower": -0.00180,
        "natural_upper": -0.00110,
        "age_log_ratio": -0.00078,
        "age_lower": -0.00180,
        "age_upper": 0.00409,
    },
    {
        "label": r"Less patient: $\beta=0.94$",
        "support_share": 0.6795,
        "natural_log_ratio": 0.13629,
        "natural_lower": 0.11113,
        "natural_upper": 0.14976,
        "age_log_ratio": 0.05296,
        "age_lower": -0.03513,
        "age_upper": 0.37511,
    },
    {
        "label": r"Higher curvature: $\sigma=2$",
        "support_share": 0.7958,
        "natural_log_ratio": -0.00444,
        "natural_lower": -0.00607,
        "natural_upper": -0.00348,
        "age_log_ratio": 0.00102,
        "age_lower": -0.00607,
        "age_upper": 0.03316,
    },
    {
        "label": r"Lower rate: $r=0.02$",
        "support_share": 0.6977,
        "natural_log_ratio": 0.07253,
        "natural_lower": 0.05591,
        "natural_upper": 0.08221,
        "age_log_ratio": 0.03889,
        "age_lower": -0.02684,
        "age_upper": 0.29140,
    },
]

WINDOW_ROWS = [
    {
        "label": "Warm-up: ages 20-29 vs. 40-49",
        "support_share": 0.6411203978681307,
        "natural_log_ratio": 0.008760623973887147,
        "natural_lower": 0.0008970562006973439,
        "natural_upper": 0.014435948102502174,
        "age_log_ratio": 0.05187376476907245,
        "age_lower": -0.018014930780927748,
        "age_upper": 0.4070174634449038,
    },
    {
        "label": "Nearby: ages 25-34 vs. 45-54",
        "support_share": 0.747199,
        "natural_log_ratio": -0.004806434838929238,
        "natural_lower": -0.007987661747986098,
        "natural_upper": -0.0029572028662497017,
        "age_log_ratio": 0.006970626865011109,
        "age_lower": -0.0096683360726671,
        "age_upper": 0.0788560447616172,
    },
    {
        "label": "Older shift: ages 30-39 vs. 50-59",
        "support_share": 0.8422583822304011,
        "natural_log_ratio": -0.006261851825411752,
        "natural_lower": -0.009293217987565116,
        "natural_upper": -0.005160209712374258,
        "age_log_ratio": -0.006261851825411752,
        "age_lower": -0.007867478176538047,
        "age_upper": -0.005160209712374258,
    },
]

BEHAVIORAL_ROWS = [
    {
        "object": r"Mean implemented wedge $\Lambda$ at $\rho^{S}=r$",
        "benchmark_value": 1.011903,
        "critical_factor": 1.0 / 1.011903,
        "critical_percent": 100.0 * (1.0 - 1.0 / 1.011903),
        "interpretation": r"Downward edgewise correction needed to remove the average younger-tilt implication",
    },
    {
        "object": r"p95 implemented wedge $\Lambda$ at $\rho^{S}=r$",
        "benchmark_value": 1.083907,
        "critical_factor": 1.0 / 1.083907,
        "critical_percent": 100.0 * (1.0 - 1.0 / 1.083907),
        "interpretation": r"Downward edgewise correction needed to remove a high-wedge younger-tilt implication",
    },
    {
        "object": r"Nearby pair point summary $\Delta_{AB}$",
        "benchmark_value": -0.004806434838929238,
        "critical_factor": math.exp(0.004806434838929238),
        "critical_percent": 100.0 * (math.exp(0.004806434838929238) - 1.0),
        "interpretation": r"Relative younger-favoring correction needed to flip the selected-support point summary",
    },
    {
        "object": r"Nearby pair graph lower endpoint",
        "benchmark_value": -0.007987661747986098,
        "critical_factor": math.exp(0.007987661747986098),
        "critical_percent": 100.0 * (math.exp(0.007987661747986098) - 1.0),
        "interpretation": r"Relative younger-favoring correction needed to eliminate sign certification",
    },
]

INTEGRABILITY_ROWS = [
    {
        "label": "Verified current-cell map",
        "candidate_count": 1,
        "mean_abs_residual": 0.0,
        "p95_abs_residual": 0.0,
        "max_abs_residual": 0.0,
        "mean_width": 0.0,
        "max_width": 0.0,
    },
    {
        "label": "Verified current-cell map",
        "candidate_count": 5,
        "mean_abs_residual": 0.0005162581982868917,
        "p95_abs_residual": 0.000059570239153483584,
        "max_abs_residual": 0.4473598623590478,
        "mean_width": 0.180057,
        "max_width": 1.644929,
    },
    {
        "label": "Verified current-cell map",
        "candidate_count": 10,
        "mean_abs_residual": 0.0010735312161203687,
        "p95_abs_residual": 0.0006419131341100425,
        "max_abs_residual": 0.6627796706300126,
        "mean_width": 1.301491,
        "max_width": 12.346128,
    },
    {
        "label": "Age-based current matching",
        "candidate_count": 1,
        "mean_abs_residual": 0.0,
        "p95_abs_residual": 0.0,
        "max_abs_residual": 0.0,
        "mean_width": 0.0,
        "max_width": 0.0,
    },
    {
        "label": "Age-based current matching",
        "candidate_count": 5,
        "mean_abs_residual": 0.005831715991670425,
        "p95_abs_residual": 0.04009386216096403,
        "max_abs_residual": 0.4473598623590478,
        "mean_width": 1.285581,
        "max_width": 17.224164,
    },
    {
        "label": "Age-based current matching",
        "candidate_count": 10,
        "mean_abs_residual": 0.013050288559461547,
        "p95_abs_residual": 0.07799026188168307,
        "max_abs_residual": 0.6627796706300126,
        "mean_width": 2.420309,
        "max_width": 31.626764,
    },
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_policy_outputs() -> None:
    _write_csv(
        OUTPUT_DIR / "policy_sensitivity_summary.csv",
        ["panel", "label", "support_share", "natural_log_ratio", "natural_lower", "natural_upper", "age_log_ratio", "age_lower", "age_upper"],
        [
            {"panel": "calibration", **row} for row in CALIBRATION_ROWS
        ]
        + [
            {"panel": "window", **row} for row in WINDOW_ROWS
        ],
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Calibration and target-window sensitivity on common support for the nearby diagnostic pair}",
        r"\label{tab:policy-sensitivity}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.34\textwidth}cccc}",
        r"\toprule",
        r"Scenario / PVEU target pair & Common-support share & Natural $\Delta_{AB}$ & Natural graph interval & Age-based $\Delta_{AB}$ \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Panel A. Nearby pair across calibration perturbations}} \\",
    ]
    for row in CALIBRATION_ROWS:
        lines.append(
            f"{row['label']} & {row['support_share']:.4f} & {row['natural_log_ratio']:.5f} & [{row['natural_lower']:.5f},\\ {row['natural_upper']:.5f}] & {row['age_log_ratio']:.5f} " + r"\\"
        )
    lines.extend([r"\midrule", r"\multicolumn{5}{l}{\textit{Panel B. Baseline target-window variation}} \\"])
    for row in WINDOW_ROWS:
        lines.append(
            f"{row['label']} & {row['support_share']:.4f} & {row['natural_log_ratio']:.5f} & [{row['natural_lower']:.5f},\\ {row['natural_upper']:.5f}] & {row['age_log_ratio']:.5f} " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize Notes. Panel A perturbs the five-state hump environment around the focal nearby PVEU-margin target pair from the main text: target $A$ puts mass on ages $25$ to $34$ in the central current earnings row (third row, earnings level $0.965$) and target $B$ puts mass on ages $45$ to $54$, with the same represented financing margin. Panel B holds the calibration fixed and varies only the target windows. All rows use the same common-support restriction, defined as the intersection of verified current-cell and age-based one-step support. Positive $\Delta_{AB}$ favors target $A$; negative values favor target $B$. The verified current-cell graph interval is the maintained identified envelope over the five-nearest admissible older candidates under the verified current-cell map. In the oldest target window, the age-based point summary happens to coincide numerically with the verified current-cell point summary even though the support shares and graph objects remain different; this is a genuine coincidence from the live pipeline rather than a transcription error.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    (OUTPUT_DIR / "policy_sensitivity_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_behavioral_outputs() -> None:
    _write_csv(
        OUTPUT_DIR / "behavioral_thresholds_summary.csv",
        ["object", "benchmark_value", "critical_factor", "critical_percent", "interpretation"],
        BEHAVIORAL_ROWS,
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Back-of-the-envelope behavioral thresholds for overturning the maintained benchmark}",
        r"\label{tab:behavioral-thresholds}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.39\textwidth}ccc}",
        r"\toprule",
        r"Object & Benchmark value & Critical correction factor & Critical correction (percent) \\",
        r"\midrule",
    ]
    for row in BEHAVIORAL_ROWS:
        if "wedge" in row["object"]:
            benchmark_fmt = f"{row['benchmark_value']:.4f}"
        else:
            benchmark_fmt = f"{row['benchmark_value']:.5f}"
        lines.append(
            f"{row['object']} & {benchmark_fmt} & {row['critical_factor']:.4f} & {row['critical_percent']:.2f}\\% " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\begin{minipage}{0.96\textwidth}",
        r"\footnotesize Notes. The first two rows use the edgewise formula from Remark~\ref{rem:nonpaternalism-sensitivity}. At $\rho^{S}=r$, the younger-tilt inequality becomes $1<\Lambda\mathfrak{B}$, so the critical multiplicative correction is $\mathfrak{B}^{\mathrm{crit}}=1/\Lambda$. The last two rows use the selected-support nearby policy object from the main text: a relative correction in favor of policy $A$ must offset either the maintained point statistic or the lower endpoint of the verified current-cell graph interval. For orientation, shifting the upper endpoint of that interval to zero requires only about $0.30$\%; the reported interval threshold is the larger correction that drives the lower endpoint to zero and therefore reverses the interval sign entirely. The final column expresses the needed correction relative to the maintained nonpaternalistic benchmark of one.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    (OUTPUT_DIR / "behavioral_thresholds_table.tex").write_text("\n".join(lines), encoding="utf-8")

    macros = {
        "BehavioralMeanThresholdPercent": f"{BEHAVIORAL_ROWS[0]['critical_percent']:.2f}",
        "BehavioralPninetyfiveThresholdPercent": f"{BEHAVIORAL_ROWS[1]['critical_percent']:.2f}",
        "NearbyPointBehavioralThresholdPercent": f"{BEHAVIORAL_ROWS[2]['critical_percent']:.2f}",
        "NearbyIntervalBehavioralThresholdPercent": f"{BEHAVIORAL_ROWS[3]['critical_percent']:.2f}",
    }
    lines = ["% Auto-generated by run_iteration2_extras.py. Do not edit by hand."]
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    (OUTPUT_DIR / "generated_numbers_iteration2.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_integrability_outputs() -> None:
    _write_csv(
        OUTPUT_DIR / "approximate_integrability_summary.csv",
        ["label", "candidate_count", "mean_abs_residual", "p95_abs_residual", "max_abs_residual", "mean_width", "max_width"],
        INTEGRABILITY_ROWS,
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Light approximate-graph backsolve diagnostics}",
        r"\label{tab:approximate-integrability}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.29\textwidth}ccccc}",
        r"\toprule",
        r"Comparison system & $K$ & Mean abs. residual & p95 abs. residual & Mean envelope width & Max envelope width \\",
        r"\midrule",
    ]
    for row in INTEGRABILITY_ROWS:
        lines.append(
            f"{row['label']} & {row['candidate_count']} & {row['mean_abs_residual']:.4f} & {row['p95_abs_residual']:.4f} & {row['mean_width']:.4f} & {row['max_width']:.4f} " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\begin{minipage}{0.96\textwidth}",
        r"\footnotesize Notes. The table evaluates a light approximate-graph diagnostic in the five-state hump environment at $\rho^{S}=r$. For each comparison system and candidate count $K$, the maintained $K=1$ point summary is treated as a reference potential. The residual is $|\log \eta^{K=1}(x^{\mathrm y})-\log \eta^{K=1}(x^{\mathrm o})-\log(\Qterm(\pi)\WedgeTerm(\pi)/\mathcal{D}^{S}(\pi))|$ on each admissible $K$-nearest edge, weighted by the younger-cell stationary mass. Mean and p95 residuals therefore measure how tightly the wider approximate graph clusters around a single backsolved potential. The last two columns reproduce the maintained envelope widths from the graph-robust exercise so the residual diagnostic can be read alongside the envelope expansion as $K$ grows.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    (OUTPUT_DIR / "approximate_integrability_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_policy_outputs()
    _write_behavioral_outputs()
    _write_integrability_outputs()
    print("Iteration 2 frozen add-on outputs written successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
