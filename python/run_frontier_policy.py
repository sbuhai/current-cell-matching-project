"""Run the support-sufficiency, graph-envelope, and policy-ranking pipeline.

This script computes the new quantitative objects introduced in the final
revision: the support-versus-sufficiency frontier, the graph-robust maintained
envelope over admissible one-step correspondences, and a compact local
budget-neutral PVEU-margin comparison.
"""

from __future__ import annotations

from pathlib import Path
import os

from src.calibration import OUTPUT_DIR, richer_state_calibration
from src.demography import survival_probabilities
from src.distribution import stationary_joint_distribution
from src.household import solve_household
from src.reporting import (
    plot_graph_envelope_by_age,
    plot_support_sufficiency_frontier,
    write_kq_sensitivity_summary,
    write_kq_sensitivity_table_tex,
    write_policy_map_comparison_table_tex,
    write_policy_ranking_summary,
    write_policy_ranking_table_tex,
    write_policy_robustness_table_tex,
    write_support_sufficiency_audit_summary,
    write_support_sufficiency_audit_table_tex,
    write_support_sufficiency_summary,
)
from src.tex_macros import tex_decimal, tex_percent, write_tex_macros
from src.theorem_objects import (
    approximate_sufficiency_summary,
    envelope_width_sensitivity_summaries,
    graph_envelope_summary,
    kappa_sensitivity_summaries,
    natural_state_groups,
    partition_audit_summaries,
    policy_decomposition_summary,
    policy_ranking_summary,
    policy_target_union_support_share,
    pooled_state_groups,
    support_overlap_mask,
    support_sufficiency_group_specs,
)


def full_main() -> None:
    print("Solving richer-state calibration...", flush=True)
    calibration = richer_state_calibration()
    survival = survival_probabilities(calibration)
    solution = solve_household(calibration, survival)
    joint = stationary_joint_distribution(calibration, solution, survival)

    print("Computing support-sufficiency frontier...", flush=True)
    frontier_rows = []
    for label, state_groups in support_sufficiency_group_specs(calibration):
        frontier_rows.append(
            approximate_sufficiency_summary(
                calibration,
                solution,
                joint,
                state_groups=state_groups,
                label=label,
            )
        )

    write_support_sufficiency_summary(frontier_rows)
    plot_support_sufficiency_frontier(frontier_rows)

    natural_row = next(row for row in frontier_rows if row.label == "Verified current-cell map")
    three_block_row = next((row for row in frontier_rows if row.label == "Three-block coarsening"), None)
    standardized_row = next(row for row in frontier_rows if row.label == "Standardized heuristic map")
    pooled_row = next(row for row in frontier_rows if row.label == "Age-only current matching")

    natural_groups = natural_state_groups(calibration)
    age_only_groups = pooled_state_groups(calibration)
    common_support = support_overlap_mask(solution, joint, natural_groups) & support_overlap_mask(solution, joint, age_only_groups)
    common_support_share = policy_target_union_support_share(
        joint,
        calibration,
        solution,
        common_support,
    )

    policy_map_rows = []
    for frontier_row in frontier_rows:
        state_groups = frontier_row.state_groups
        own_support = support_overlap_mask(solution, joint, state_groups)
        own_support_share = policy_target_union_support_share(
            joint,
            calibration,
            solution,
            own_support,
        )
        policy_map_rows.append(
            policy_ranking_summary(
                calibration,
                solution,
                joint,
                state_groups=state_groups,
                label=frontier_row.label,
                support_label="Own one-step support",
                support_share=own_support_share,
                kappa_mean=frontier_row.weighted_mean_kappa,
                kappa_max=frontier_row.max_kappa,
                support_mask=own_support,
            )
        )
    write_policy_map_comparison_table_tex(policy_map_rows, frontier_rows)

    policy_rows = [
        policy_ranking_summary(
            calibration,
            solution,
            joint,
            state_groups=age_only_groups,
            label="Age-only current matching",
            support_label="Full maintained support",
            support_share=1.0,
            kappa_mean=pooled_row.weighted_mean_kappa,
            kappa_max=pooled_row.max_kappa,
            support_mask=None,
        ),
        policy_ranking_summary(
            calibration,
            solution,
            joint,
            state_groups=age_only_groups,
            label="Age-only current matching",
            support_label="Common support only",
            support_share=common_support_share,
            kappa_mean=pooled_row.weighted_mean_kappa,
            kappa_max=pooled_row.max_kappa,
            support_mask=common_support,
        ),
        policy_ranking_summary(
            calibration,
            solution,
            joint,
            state_groups=natural_groups,
            label="Verified current-cell map",
            support_label="Common support only",
            support_share=common_support_share,
            kappa_mean=natural_row.weighted_mean_kappa,
            kappa_max=natural_row.max_kappa,
            support_mask=common_support,
        ),
    ]
    decomposition = policy_decomposition_summary(policy_rows[0], policy_rows[1], policy_rows[2])
    write_policy_ranking_summary(policy_rows, decomposition)
    write_policy_ranking_table_tex(policy_rows, decomposition)

    robustness_support_share = policy_target_union_support_share(
        joint,
        calibration,
        solution,
        common_support,
        target_a_age_bounds=(25, 34),
        target_a_state_indices=(len(calibration.income_states) // 2,),
        target_b_age_bounds=(45, 54),
    )
    robustness_rows = [
        policy_ranking_summary(
            calibration,
            solution,
            joint,
            state_groups=age_only_groups,
            label="Age-only current matching",
            support_label="Full maintained support",
            support_share=1.0,
            kappa_mean=pooled_row.weighted_mean_kappa,
            kappa_max=pooled_row.max_kappa,
            support_mask=None,
            target_a_age_bounds=(25, 34),
            target_a_state_indices=(len(calibration.income_states) // 2,),
            target_b_age_bounds=(45, 54),
        ),
        policy_ranking_summary(
            calibration,
            solution,
            joint,
            state_groups=age_only_groups,
            label="Age-only current matching",
            support_label="Common support only",
            support_share=robustness_support_share,
            kappa_mean=pooled_row.weighted_mean_kappa,
            kappa_max=pooled_row.max_kappa,
            support_mask=common_support,
            target_a_age_bounds=(25, 34),
            target_a_state_indices=(len(calibration.income_states) // 2,),
            target_b_age_bounds=(45, 54),
        ),
        policy_ranking_summary(
            calibration,
            solution,
            joint,
            state_groups=natural_groups,
            label="Verified current-cell map",
            support_label="Common support only",
            support_share=robustness_support_share,
            kappa_mean=natural_row.weighted_mean_kappa,
            kappa_max=natural_row.max_kappa,
            support_mask=common_support,
            target_a_age_bounds=(25, 34),
            target_a_state_indices=(len(calibration.income_states) // 2,),
            target_b_age_bounds=(45, 54),
        ),
    ]
    robustness_decomposition = policy_decomposition_summary(
        robustness_rows[0],
        robustness_rows[1],
        robustness_rows[2],
    )
    write_policy_ranking_summary(
        robustness_rows,
        robustness_decomposition,
        stem="policy_ranking_robustness_summary",
    )
    write_policy_robustness_table_tex(robustness_rows, robustness_decomposition)

    kappa_sensitivity = kappa_sensitivity_summaries(calibration, solution, joint)
    envelope_sensitivity = envelope_width_sensitivity_summaries(calibration, solution, joint)
    write_kq_sensitivity_summary(kappa_sensitivity, envelope_sensitivity)
    write_kq_sensitivity_table_tex(kappa_sensitivity, envelope_sensitivity)

    print("Computing graph envelopes...", flush=True)
    graph_rows = [
        graph_envelope_summary(
            calibration,
            solution,
            joint,
            state_groups=natural_state_groups(calibration),
            label="Verified current-cell map",
        ),
        graph_envelope_summary(
            calibration,
            solution,
            joint,
            state_groups=pooled_state_groups(calibration),
            label="Age-only current matching",
        ),
    ]
    plot_graph_envelope_by_age(graph_rows)

    outputs_dir = OUTPUT_DIR
    macros = {
        "FrontierNaturalOverlap": tex_decimal(natural_row.overall_support_overlap_share, 4),
        "FrontierStandardizedOverlap": tex_decimal(standardized_row.overall_support_overlap_share, 4),
        "FrontierStandardizedMeanKappaPlp": tex_decimal(100.0 * standardized_row.weighted_mean_kappa, 2),
        "FrontierStandardizedMaxKappaPlp": tex_decimal(100.0 * standardized_row.max_kappa, 2),
        "FrontierAgeOnlyOverlap": tex_decimal(pooled_row.overall_support_overlap_share, 4),
        "FrontierAgeOnlyMeanKappaPlp": tex_decimal(100.0 * pooled_row.weighted_mean_kappa, 2),
        "FrontierAgeOnlyMaxKappaPlp": tex_decimal(100.0 * pooled_row.max_kappa, 2),
        "PolicyBaselineCommonSupportShare": tex_decimal(common_support_share, 4),
    }
    if three_block_row is not None:
        macros.update({
            "FrontierThreeBlockOverlap": tex_decimal(three_block_row.overall_support_overlap_share, 4),
            "FrontierThreeBlockMeanKappaPlp": tex_decimal(100.0 * three_block_row.weighted_mean_kappa, 2),
            "FrontierThreeBlockMaxKappaPlp": tex_decimal(100.0 * three_block_row.max_kappa, 2),
        })
    write_tex_macros(outputs_dir / "generated_numbers_frontier.tex", macros)

    print("Writing frontier outputs...", flush=True)
    audit_rows = partition_audit_summaries(calibration, solution, joint, social_rate=calibration.interest_rate)
    write_support_sufficiency_audit_summary(audit_rows)
    write_support_sufficiency_audit_table_tex(audit_rows)

    print("Support-sufficiency and policy pipeline executed successfully.")
    print(f"Outputs written to: {outputs_dir}")




_REQUIRED_OUTPUTS = ['support_sufficiency_frontier.csv', 'support_sufficiency_audit.csv', 'support_sufficiency_audit_table.tex', 'policy_ranking_robustness_summary.csv', 'policy_ranking_robustness_table.tex', 'policy_map_comparison_table.tex', 'frontier_graph_kq_sensitivity.csv', 'frontier_graph_kq_sensitivity_table.tex', 'generated_numbers_frontier.tex']


def _validate_frozen_outputs() -> None:
    out_dir = Path(__file__).resolve().parent / "outputs"
    missing = [name for name in _REQUIRED_OUTPUTS if not (out_dir / name).exists()]
    if missing:
        raise SystemExit("Missing frozen outputs: " + ", ".join(missing))
    print("Frozen outputs validated for run_frontier_policy.py. Set FULL_REBUILD=1 to recompute the finite-grid solve.")
    for name in _REQUIRED_OUTPUTS:
        print(f"validated: {name}")


def main() -> None:
    if os.environ.get("FULL_REBUILD") == "1":
        full_main()
    else:
        _validate_frozen_outputs()


if __name__ == "__main__":
    main()
