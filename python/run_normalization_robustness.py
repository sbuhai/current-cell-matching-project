"""Run the alternative-normalization robustness exercise for the paper.

This script rebuilds the exact comparison graph under an alternative
age-state reference-slope private metric in the five-state hump environment,
then reports support, edge/component, and target-ranking diagnostics.
"""

from __future__ import annotations

from src.calibration import OUTPUT_DIR, richer_state_calibration
from src.demography import survival_probabilities
from src.distribution import stationary_joint_distribution
from src.household import solve_household
from src.reporting import (
    write_alternative_normalization_summary,
    write_alternative_normalization_table_tex,
)
from src.theorem_objects import (
    alternative_normalization_summary,
    natural_state_groups,
    policy_target_union_support_share,
    pooled_state_groups,
    support_overlap_mask,
)


def main() -> None:
    calibration = richer_state_calibration()
    survival = survival_probabilities(calibration)
    solution = solve_household(calibration, survival)
    joint = stationary_joint_distribution(calibration, solution, survival)

    natural_groups = natural_state_groups(calibration)
    age_only_groups = pooled_state_groups(calibration)
    common_support = support_overlap_mask(solution, joint, natural_groups) & support_overlap_mask(solution, joint, age_only_groups)

    middle_state = (len(calibration.income_states) // 2,)

    warm_up_support_share = policy_target_union_support_share(
        joint,
        calibration,
        solution,
        common_support,
        target_a_age_bounds=(20, 29),
        target_a_state_indices=middle_state,
        target_b_age_bounds=(40, 49),
    )
    nearby_support_share = policy_target_union_support_share(
        joint,
        calibration,
        solution,
        common_support,
        target_a_age_bounds=(25, 34),
        target_a_state_indices=middle_state,
        target_b_age_bounds=(45, 54),
    )

    rows = [
        alternative_normalization_summary(
            calibration,
            solution,
            joint,
            state_groups=natural_groups,
            label="Warm-up pair: ages 20-29 vs. 40-49",
            support_label="Common support only",
            support_share=warm_up_support_share,
            support_mask=common_support,
            target_a_age_bounds=(20, 29),
            target_a_state_indices=middle_state,
            target_b_age_bounds=(40, 49),
        ),
        alternative_normalization_summary(
            calibration,
            solution,
            joint,
            state_groups=natural_groups,
            label="Nearby pair: ages 25-34 vs. 45-54",
            support_label="Common support only",
            support_share=nearby_support_share,
            support_mask=common_support,
            target_a_age_bounds=(25, 34),
            target_a_state_indices=middle_state,
            target_b_age_bounds=(45, 54),
        ),
    ]

    write_alternative_normalization_summary(rows)
    write_alternative_normalization_table_tex(rows)

    print("Alternative-normalization robustness exercise executed successfully.")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
