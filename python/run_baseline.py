"""Run the baseline quantitative pipeline for the paper.

Usage
-----
From the `python/` directory,

    python run_baseline.py

The script writes machine-readable summaries and paper-ready figures/tables to
`python/outputs/`.
"""

from __future__ import annotations

from pathlib import Path

from src.calibration import OUTPUT_DIR, baseline_calibration
from src.demography import survival_probabilities
from src.distribution import stationary_joint_distribution
from src.household import solve_household
from src.reporting import (
    plot_required_eta_bands_by_age,
    plot_required_eta_ratio_by_age,
    plot_spread_distance_by_social_rate,
    write_calibration_summary,
    write_comparison_summary,
    write_long_horizon_summary,
    write_matching_summary,
    write_net_term_sign_summary,
    write_net_term_sign_table_tex,
    write_results_table_tex,
    write_sensitivity_summary,
    write_sensitivity_table_tex,
    write_wedge_summary,
)
from src.tex_macros import tex_auto, tex_decimal, write_tex_macros
from src.theorem_objects import comparison_results, net_term_age_summaries, sensitivity_results



def main() -> None:
    calibration = baseline_calibration()
    survival = survival_probabilities(calibration)
    solution = solve_household(calibration, survival)
    joint = stationary_joint_distribution(calibration, solution, survival)
    results, matching, wedge_diag = comparison_results(calibration, solution, joint)
    sensitivity = sensitivity_results(calibration, solution, joint)
    net_term_rows = net_term_age_summaries(calibration, solution, joint)

    write_calibration_summary(calibration)
    write_matching_summary(calibration, matching)
    write_wedge_summary(calibration, wedge_diag)
    write_comparison_summary(calibration, results)
    write_long_horizon_summary(calibration, results, joint)
    write_results_table_tex(calibration, results)
    write_sensitivity_summary(calibration, sensitivity)
    write_sensitivity_table_tex(calibration, sensitivity)
    write_net_term_sign_summary(net_term_rows)
    write_net_term_sign_table_tex(net_term_rows)
    plot_required_eta_ratio_by_age(calibration, results, joint)
    plot_required_eta_bands_by_age(calibration, results, joint)
    plot_spread_distance_by_social_rate(calibration, results)

    outputs_dir = OUTPUT_DIR
    write_tex_macros(
        outputs_dir / "generated_numbers_baseline.tex",
        {
            "BaselineMatchMean": tex_auto(matching.diagnostics.weighted_mean_abs_log_gap),
            "BaselineMatchPNinetyFive": tex_auto(matching.diagnostics.weighted_p95_abs_log_gap),
            "BaselineMatchPNinetyNine": tex_auto(matching.diagnostics.weighted_p99_abs_log_gap),
            "BaselineMatchMax": tex_auto(matching.diagnostics.weighted_max_abs_log_gap),
            "BaselineOverlapOverall": tex_decimal(matching.diagnostics.overall_support_overlap_share, 6),
            "BaselineOverlapMinAge": tex_decimal(matching.diagnostics.min_age_support_overlap_share, 6),
            "BaselineRawEulerMean": tex_decimal(wedge_diag.raw_all.mean, 4),
            "BaselineRawEulerPNinetyFive": tex_decimal(wedge_diag.raw_all.p95, 4),
            "BaselineRawEulerSubunitShare": tex_decimal(wedge_diag.raw_all.subunit_mass_share, 4),
            "BaselineRawInteriorMeanAbsDeviation": tex_auto(wedge_diag.raw_strict_interior_mean_abs_deviation),
            "BaselineStrictInteriorShare": tex_decimal(wedge_diag.strict_interior_mass_share, 4),
            "BaselineBoundaryLoadingShare": tex_decimal(wedge_diag.lower_bound_mass_share, 4),
            "BaselineImplementedWedgeMean": tex_decimal(wedge_diag.implemented.mean, 4),
            "BaselineImplementedWedgePNinetyFive": tex_decimal(wedge_diag.implemented.p95, 4),
        },
    )
    print("Baseline quantitative pipeline executed successfully.")
    print(f"Outputs written to: {outputs_dir}")
    print(
        "Direct matching diagnostics: "
        f"mean={matching.diagnostics.mean_abs_log_gap:.6f}, "
        f"median={matching.diagnostics.median_abs_log_gap:.6f}, "
        f"p95={matching.diagnostics.p95_abs_log_gap:.6f}, "
        f"p99={matching.diagnostics.p99_abs_log_gap:.6f}, "
        f"max={matching.diagnostics.max_abs_log_gap:.6f}, "
        f"overall_overlap={matching.diagnostics.overall_support_overlap_share:.6f}, "
        f"min_age_overlap={matching.diagnostics.min_age_support_overlap_share:.6f}"
    )


if __name__ == "__main__":
    main()
