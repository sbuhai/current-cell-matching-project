"""Run the harder-environment robustness pipeline for the paper.

This script solves more demanding quantitative environments used for the main-text
robustness table and the supplemental excluded-mass diagnostics.
"""

from __future__ import annotations

import csv
from pathlib import Path
import os

from src.calibration import baseline_calibration, hump_efficiency_calibration, richer_state_calibration
from src.demography import survival_probabilities
from src.distribution import stationary_joint_distribution
from src.household import solve_household
from src.reporting import plot_excluded_mass_heatmap, write_excluded_mass_summary
from src.tex_macros import tex_decimal, write_tex_macros
from src.theorem_objects import (
    comparison_results,
    excluded_mass_summaries,
    natural_state_groups,
    standardized_state_groups,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"



def _label(calibration_name: str) -> str:
    if calibration_name == "baseline":
        return "Baseline: flat efficiency, 2 earnings states"
    if calibration_name == "hump_efficiency":
        return "Hump efficiency, 2 earnings states"
    if calibration_name == "five_state_hump":
        return "Hump efficiency, 5 earnings states"
    return calibration_name



def _market_result(calibration, results):
    return min(results, key=lambda r: abs(r.social_rate - calibration.interest_rate))



def _write_csv(rows: list[dict[str, float | int | str]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "robustness_harder_env_summary.csv"
    fieldnames = [
        "specification",
        "comparison_system",
        "asset_size",
        "age_efficiency_mode",
        "earnings_state_count",
        "overall_support_overlap_share",
        "weighted_mean_abs_log_gap",
        "weighted_p95_abs_log_gap",
        "weighted_p99_abs_log_gap",
        "spread_r",
        "conditional_wasserstein_r",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out



def _panel_rows(rows: list[dict[str, float | int | str]], comparison_system: str) -> list[dict[str, float | int | str]]:
    return [row for row in rows if row["comparison_system"] == comparison_system]



def _write_tex(rows: list[dict[str, float | int | str]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "robustness_harder_env_table.tex"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Harder-environment robustness under natural and standardized comparison systems at $\rho^{S}=r$}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.29\textwidth}ccccc}",
        r"\toprule",
        r"Specification & Overlap & Mean $|\Delta \log \upsilon|$ & p95 $|\Delta \log \upsilon|$ & $\widehat{\Spread}$ & $\widehat{\DemDist}$ \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Panel A. Natural comparison system}} \\",
    ]
    for row in _panel_rows(rows, "natural"):
        lines.append(
            f"{row['specification']} & "
            f"{row['overall_support_overlap_share']:.4f} & "
            f"{row['weighted_mean_abs_log_gap']:.4f} & "
            f"{row['weighted_p95_abs_log_gap']:.4f} & "
            f"{row['spread_r']:.4f} & "
            f"{row['conditional_wasserstein_r']:.4f} " + "\\\\"
        )
    lines.extend([r"\midrule", r"\multicolumn{6}{l}{\textit{Panel B. Standardized below-mean vs. at-or-above-mean earnings groups}} \\"])
    for row in _panel_rows(rows, "standardized"):
        lines.append(
            f"{row['specification']} & "
            f"{row['overall_support_overlap_share']:.4f} & "
            f"{row['weighted_mean_abs_log_gap']:.4f} & "
            f"{row['weighted_p95_abs_log_gap']:.4f} & "
            f"{row['spread_r']:.4f} & "
            f"{row['conditional_wasserstein_r']:.4f} " + "\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.97\textwidth}",
            r"\footnotesize Notes. Panel A lets the verified comparison map move with the earnings partition of each environment. Panel B keeps fixed a coarser below-mean versus at-or-above-mean earnings grouping only as a heuristic cross-specification device; it is used only as a heuristic cross-specification comparison system. The table reports direct matching diagnostics and the maintained geometric summaries at the market benchmark $\rho^{S}=r$.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out



def full_main() -> None:
    rows: list[dict[str, float | int | str]] = []
    excluded_rows_five_state = []
    calibrations = [baseline_calibration(), hump_efficiency_calibration(), richer_state_calibration()]
    for calibration in calibrations:
        survival = survival_probabilities(calibration)
        solution = solve_household(calibration, survival)
        joint = stationary_joint_distribution(calibration, solution, survival)

        for comparison_system, state_groups, group_name in [
            ("natural", natural_state_groups(calibration), "earnings_state"),
            ("standardized", standardized_state_groups(calibration), "below_vs_above_mean_earnings"),
        ]:
            results, matching, _wedge_diag = comparison_results(
                calibration,
                solution,
                joint,
                state_groups=state_groups,
                group_name=group_name,
            )
            market_result = _market_result(calibration, results)
            rows.append(
                {
                    "specification": _label(calibration.name),
                    "comparison_system": comparison_system,
                    "asset_size": calibration.asset_size,
                    "age_efficiency_mode": calibration.age_efficiency_mode,
                    "earnings_state_count": len(calibration.income_states),
                    "overall_support_overlap_share": matching.diagnostics.overall_support_overlap_share,
                    "weighted_mean_abs_log_gap": matching.diagnostics.weighted_mean_abs_log_gap,
                    "weighted_p95_abs_log_gap": matching.diagnostics.weighted_p95_abs_log_gap,
                    "weighted_p99_abs_log_gap": matching.diagnostics.weighted_p99_abs_log_gap,
                    "spread_r": market_result.spread,
                    "conditional_wasserstein_r": market_result.conditional_wasserstein,
                }
            )

        if calibration.name == "five_state_hump":
            excluded_rows = excluded_mass_summaries(calibration, solution, joint, state_groups=natural_state_groups(calibration))
            excluded_rows_five_state = excluded_rows
            write_excluded_mass_summary(calibration, excluded_rows)
            plot_excluded_mass_heatmap(calibration, excluded_rows)

    csv_path = _write_csv(rows)
    tex_path = _write_tex(rows)

    natural_hump_two = next(row for row in rows if row["specification"] == _label("hump_efficiency") and row["comparison_system"] == "natural")
    natural_five_state = next(row for row in rows if row["specification"] == _label("five_state_hump") and row["comparison_system"] == "natural")
    standardized_five_state = next(row for row in rows if row["specification"] == _label("five_state_hump") and row["comparison_system"] == "standardized")
    below_forty_share = sum(
        row.total_share_of_excluded
        for row in excluded_rows_five_state
        if row.age_block in {"20-29", "30-39"}
    )
    write_tex_macros(
        OUTPUT_DIR / "generated_numbers_robustness.tex",
        {
            "RobustNaturalFiveStateOverlap": tex_decimal(float(natural_five_state["overall_support_overlap_share"]), 4),
            "RobustNaturalFiveStateExcludedShare": tex_decimal(1.0 - float(natural_five_state["overall_support_overlap_share"]), 4),
            "RobustExcludedBelowFortyShare": tex_decimal(float(below_forty_share), 4),
            "RobustNaturalHumpTwoStateDemDist": tex_decimal(float(natural_hump_two["conditional_wasserstein_r"]), 4),
            "RobustNaturalFiveStateDemDist": tex_decimal(float(natural_five_state["conditional_wasserstein_r"]), 4),
            "RobustStandardizedFiveStateOverlap": tex_decimal(float(standardized_five_state["overall_support_overlap_share"]), 4),
            "RobustStandardizedFiveStateMeanMismatch": tex_decimal(float(standardized_five_state["weighted_mean_abs_log_gap"]), 4),
            "RobustStandardizedFiveStateDemDist": tex_decimal(float(standardized_five_state["conditional_wasserstein_r"]), 4),
        },
    )
    print("Harder-environment robustness pipeline executed successfully.")
    print(f"CSV written to: {csv_path}")
    print(f"LaTeX table written to: {tex_path}")




_REQUIRED_OUTPUTS = ['robustness_harder_env_summary.csv', 'robustness_harder_env_table.tex', 'generated_numbers_robustness.tex', 'five_state_hump_excluded_mass_summary.csv', 'five_state_hump_excluded_mass_heatmap.png']


def _validate_frozen_outputs() -> None:
    out_dir = Path(__file__).resolve().parent / "outputs"
    missing = [name for name in _REQUIRED_OUTPUTS if not (out_dir / name).exists()]
    if missing:
        raise SystemExit("Missing frozen outputs: " + ", ".join(missing))
    print("Frozen outputs validated for run_robustness.py. Set FULL_REBUILD=1 to recompute the finite-grid solve.")
    for name in _REQUIRED_OUTPUTS:
        print(f"validated: {name}")


def main() -> None:
    if os.environ.get("FULL_REBUILD") == "1":
        full_main()
    else:
        _validate_frozen_outputs()


if __name__ == "__main__":
    main()
