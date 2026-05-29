"""Run the stylized liquid/illiquid extension for the appendix revision.

This script produces a compact richer-state quantitative block built around a
convex liquid/illiquid life-cycle problem. The output is intentionally compact:
a machine-readable summary, a paper-ready TeX table, and a small macro file used
for one-sentence main-text references.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.liquid_illiquid import (
    LiquidIlliquidCalibration,
    age_average_illiquid_holdings,
    age_only_groups,
    earnings_only_groups,
    exact_state_groups,
    illiquid_tercile_groups,
    mid_earnings_combined_state_indices,
    solve_liquid_illiquid,
    stationary_joint_distribution_liquid_illiquid,
    survival_probabilities,
)
from src.tex_macros import tex_decimal, tex_interval, tex_percent, write_tex_macros
from src.theorem_objects import (
    approximate_sufficiency_summary,
    policy_ranking_summary,
    policy_target_union_support_share,
    support_overlap_mask,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _write_summary_csv(rows: list[dict[str, float | str]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "liquid_illiquid_map_comparison_summary.csv"
    fieldnames = [
        "comparison_system",
        "one_step_overlap_share",
        "target_union_support_share",
        "target_a_support_share",
        "target_b_support_share",
        "kappa_mean_plp",
        "kappa_max_plp",
        "point_log_ratio",
        "graph_lower_log_ratio_k2",
        "graph_upper_log_ratio_k2",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _write_table_tex(rows: list[dict[str, float | str]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "liquid_illiquid_map_comparison_table.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stylized liquid/illiquid extension: current-map discipline beyond the one-asset benchmark}",
        r"\label{tab:liquid-illiquid-map-comparison}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.29\textwidth}cccccc}",
        r"\toprule",
        r"Comparison system & Overlap & Target-union share & Mean $\hat\kappa^{\mathrm{mean}}$ & Worst-case $\hat\kappa^{\max}$ & Point $\Delta_{AB}$ & $K=2$ graph interval \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison_system']} & {row['one_step_overlap_share']:.4f} & {row['target_union_support_share']:.4f} & "
            f"{row['kappa_mean_plp']:.2f} & {row['kappa_max_plp']:.2f} & {row['point_log_ratio']:.4f} & "
            f"[{row['graph_lower_log_ratio_k2']:.4f},\\ {row['graph_upper_log_ratio_k2']:.4f}] " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize Notes. The stylized appendix block augments the one-asset life-cycle environment with an illiquid asset, a convex quadratic adjustment cost, and the same represented one-step liquid self-transfer margin used in the main text. "
            r"The fine benchmark map is exact current illiquid index $\times$ current earnings state. The operational theorem-guided coarsening groups current illiquid holdings into terciles and keeps current earnings fixed. The two coarser benchmarks then drop current illiquid structure altogether (earnings-only) or drop all current non-age structure (age-based). "
            r"PVEU target $A$ is the nearby main-text target on ages $25$ to $34$ in the central current earnings row (third row, earnings level $0.965$). PVEU target $B$ is ages $45$ to $54$ on the same represented margin. "
            r"The overlap and target-union columns report retained younger mass on each map's own one-step support. The contamination columns, the point statistic, and the reported $K=2$ graph interval are all in percent log points. The point statistic uses the degenerate $K=1$ point summary, while the last column reports the graph-robust maintained interval for the same object under $K=2$. The block shows that when current illiquid structure matters for the represented margin, dropping it can materially change support and the direction of the maintained local PVEU-margin comparison.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    calibration = LiquidIlliquidCalibration()
    survival = survival_probabilities(calibration)
    solution = solve_liquid_illiquid(calibration, survival)
    joint = stationary_joint_distribution_liquid_illiquid(calibration, solution, survival)

    target_a_states = mid_earnings_combined_state_indices(calibration)
    specs = [
        ("Exact current $k$ $\\times$ earnings", exact_state_groups(calibration)),
        ("Illiquid-tercile $\\times$ earnings", illiquid_tercile_groups(calibration)),
        ("Earnings only", earnings_only_groups(calibration)),
        ("Age only", age_only_groups(calibration)),
    ]

    rows: list[dict[str, float | str]] = []
    for label, state_groups in specs:
        support_summary = approximate_sufficiency_summary(
            calibration,
            solution,
            joint,
            state_groups=state_groups,
            label=label,
        )
        support_mask = support_overlap_mask(solution, joint, state_groups)
        union_support_share = policy_target_union_support_share(
            joint,
            calibration,
            solution,
            support_mask,
            target_a_age_bounds=(25, 34),
            target_a_state_indices=target_a_states,
            target_b_age_bounds=(45, 54),
        )
        point_row = policy_ranking_summary(
            calibration,
            solution,
            joint,
            state_groups=state_groups,
            label=label,
            support_label="Own one-step support",
            support_share=union_support_share,
            kappa_mean=support_summary.weighted_mean_kappa,
            kappa_max=support_summary.max_kappa,
            support_mask=support_mask,
            target_a_age_bounds=(25, 34),
            target_a_state_indices=target_a_states,
            target_b_age_bounds=(45, 54),
            candidate_count=1,
        )
        graph_row = policy_ranking_summary(
            calibration,
            solution,
            joint,
            state_groups=state_groups,
            label=label,
            support_label="Own one-step support",
            support_share=union_support_share,
            kappa_mean=support_summary.weighted_mean_kappa,
            kappa_max=support_summary.max_kappa,
            support_mask=support_mask,
            target_a_age_bounds=(25, 34),
            target_a_state_indices=target_a_states,
            target_b_age_bounds=(45, 54),
            candidate_count=2,
        )
        rows.append(
            {
                "comparison_system": label,
                "one_step_overlap_share": support_summary.overall_support_overlap_share,
                "target_union_support_share": union_support_share,
                "target_a_support_share": point_row.target_a_support_share,
                "target_b_support_share": point_row.target_b_support_share,
                "kappa_mean_plp": 100.0 * support_summary.weighted_mean_kappa,
                "kappa_max_plp": 100.0 * support_summary.max_kappa,
                "point_log_ratio": point_row.log_ratio,
                "graph_lower_log_ratio_k2": graph_row.lower_log_ratio,
                "graph_upper_log_ratio_k2": graph_row.upper_log_ratio,
            }
        )

    csv_path = _write_summary_csv(rows)
    tex_path = _write_table_tex(rows)

    age_k = age_average_illiquid_holdings(calibration, solution, joint, ages_of_interest=(25, 35, 45))
    row_by_label = {row["comparison_system"]: row for row in rows}
    write_tex_macros(
        OUTPUT_DIR / "generated_numbers_liquid_illiquid.tex",
        {
            "LiquidIlliquidAgeTwentyFiveK": tex_decimal(age_k[25], 2),
            "LiquidIlliquidAgeThirtyFiveK": tex_decimal(age_k[35], 2),
            "LiquidIlliquidAgeFortyFiveK": tex_decimal(age_k[45], 2),
            "LiquidIlliquidExactOverlap": tex_decimal(float(row_by_label["Exact current $k$ $\\times$ earnings"]["one_step_overlap_share"]), 4),
            "LiquidIlliquidTercileOverlap": tex_decimal(float(row_by_label["Illiquid-tercile $\\times$ earnings"]["one_step_overlap_share"]), 4),
            "LiquidIlliquidEarningsOnlyOverlap": tex_decimal(float(row_by_label["Earnings only"]["one_step_overlap_share"]), 4),
            "LiquidIlliquidAgeOnlyOverlap": tex_decimal(float(row_by_label["Age only"]["one_step_overlap_share"]), 4),
            "LiquidIlliquidTercileMeanKappaPlp": tex_decimal(float(row_by_label["Illiquid-tercile $\\times$ earnings"]["kappa_mean_plp"]), 2),
            "LiquidIlliquidTercileMaxKappaPlp": tex_decimal(float(row_by_label["Illiquid-tercile $\\times$ earnings"]["kappa_max_plp"]), 2),
            "LiquidIlliquidTercilePointDelta": tex_decimal(float(row_by_label["Illiquid-tercile $\\times$ earnings"]["point_log_ratio"]), 4),
            "LiquidIlliquidEarningsOnlyPointDelta": tex_decimal(float(row_by_label["Earnings only"]["point_log_ratio"]), 4),
            "LiquidIlliquidAgeOnlyPointDelta": tex_decimal(float(row_by_label["Age only"]["point_log_ratio"]), 4),
            "LiquidIlliquidTercileGraphLower": tex_decimal(float(row_by_label["Illiquid-tercile $\\times$ earnings"]["graph_lower_log_ratio_k2"]), 4),
            "LiquidIlliquidTercileGraphUpper": tex_decimal(float(row_by_label["Illiquid-tercile $\\times$ earnings"]["graph_upper_log_ratio_k2"]), 4),
            "LiquidIlliquidEarningsOnlyGraphLower": tex_decimal(float(row_by_label["Earnings only"]["graph_lower_log_ratio_k2"]), 4),
            "LiquidIlliquidEarningsOnlyGraphUpper": tex_decimal(float(row_by_label["Earnings only"]["graph_upper_log_ratio_k2"]), 4),
            "LiquidIlliquidAgeOnlyGraphLower": tex_decimal(float(row_by_label["Age only"]["graph_lower_log_ratio_k2"]), 4),
            "LiquidIlliquidAgeOnlyGraphUpper": tex_decimal(float(row_by_label["Age only"]["graph_upper_log_ratio_k2"]), 4),
        },
    )

    print("Liquid/illiquid extension executed successfully.")
    print(f"CSV written to: {csv_path}")
    print(f"TeX table written to: {tex_path}")


if __name__ == "__main__":
    main()
