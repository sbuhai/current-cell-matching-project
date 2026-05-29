"""Compute monotone lower-tail omitted-support bounds for the five-state block."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from src.calibration import richer_state_calibration
from src.demography import survival_probabilities
from src.distribution import stationary_joint_distribution
from src.household import solve_household
from src.tex_macros import tex_decimal, tex_interval, write_tex_macros
from src.theorem_objects import (
    build_matching_structure,
    implemented_wedge,
    natural_state_groups,
    reconstruct_eta_profile,
    required_eta_ratio,
    support_overlap_mask,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _policy_selector(
    shape: tuple[int, int, int],
    ages: np.ndarray,
    age_bounds: tuple[int, int],
    state_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    selector = np.zeros(shape, dtype=bool)
    age_mask = (ages >= age_bounds[0]) & (ages <= age_bounds[1])
    selector[age_mask, :, :] = True
    if state_indices is not None:
        state_mask = np.zeros(shape[2], dtype=bool)
        state_mask[np.asarray(state_indices, dtype=int)] = True
        selector &= state_mask[None, None, :]
    return selector


def _monotone_lower_tail_completion(
    eta: np.ndarray,
    mu: np.ndarray,
    support_mask: np.ndarray,
    ages: np.ndarray,
    age_bounds: tuple[int, int],
    state_indices: tuple[int, ...] | None,
) -> dict[str, float]:
    """Bound one target-group mean under lower-tail omissions and asset monotonicity.

    The completion rule uses the verified current-cell support mask
    current map. When omitted mass lies below the first supported asset in a
    given age-state block and the maintained point summary is weakly decreasing
    in current liquid assets, the omitted block mean lies between the first
    supported boundary value and the maximum value over the omitted lower-tail.
    """

    selector = _policy_selector(mu.shape, ages, age_bounds, state_indices)
    support_ext = np.zeros_like(mu, dtype=bool)
    support_ext[:-1, :, :] = support_mask
    support_ext[-1, :, :] = True

    selector_support = selector & support_ext
    selector_excluded = selector & (~support_ext)
    total_mass = float(np.sum(mu[selector]))
    support_mass = float(np.sum(mu[selector_support]))
    excluded_mass = float(np.sum(mu[selector_excluded]))
    support_share = support_mass / total_mass if total_mass > 0.0 else 0.0
    support_score = float(np.sum(eta[selector_support] * mu[selector_support]) / support_mass) if support_mass > 0.0 else float("nan")

    boundary_num = 0.0
    upper_num = 0.0
    monotonicity_shares: list[float] = []
    max_positive_diffs: list[float] = []

    for j in range(mu.shape[0]):
        for e in range(mu.shape[2]):
            block_selector = selector[j, :, e]
            if not np.any(block_selector):
                continue
            diffs = np.diff(eta[j, :, e])
            monotonicity_shares.append(float(np.mean(diffs <= 1e-10)))
            max_positive_diffs.append(float(np.max(diffs)))

            excluded_block = selector_excluded[j, :, e]
            excluded_block_mass = float(np.sum(mu[j, :, e][excluded_block]))
            if excluded_block_mass <= 0.0:
                continue

            supported_block = selector_support[j, :, e]
            if np.any(supported_block):
                first_supported = int(np.min(np.where(supported_block)[0]))
                boundary_value = float(eta[j, first_supported, e])
            else:
                boundary_value = float(np.min(eta[j, :, e][block_selector]))

            omitted_max = float(np.max(eta[j, :, e][excluded_block]))
            boundary_num += excluded_block_mass * boundary_value
            upper_num += excluded_block_mass * omitted_max

    excluded_lower = boundary_num / excluded_mass if excluded_mass > 0.0 else float("nan")
    excluded_upper = upper_num / excluded_mass if excluded_mass > 0.0 else float("nan")
    full_lower = (support_mass * support_score + excluded_mass * excluded_lower) / total_mass if total_mass > 0.0 else float("nan")
    full_upper = (support_mass * support_score + excluded_mass * excluded_upper) / total_mass if total_mass > 0.0 else float("nan")

    return {
        "target_support_share": support_share,
        "support_score": support_score,
        "excluded_mass_share": excluded_mass / total_mass if total_mass > 0.0 else 0.0,
        "excluded_lower_score": excluded_lower,
        "excluded_upper_score": excluded_upper,
        "full_lower_score": full_lower,
        "full_upper_score": full_upper,
        "min_nonincreasing_share": float(min(monotonicity_shares)) if monotonicity_shares else float("nan"),
        "mean_nonincreasing_share": float(np.mean(monotonicity_shares)) if monotonicity_shares else float("nan"),
        "max_positive_diff": float(max(max_positive_diffs)) if max_positive_diffs else float("nan"),
    }


def _log_ratio_interval(a_lower: float, a_upper: float, b_lower: float, b_upper: float) -> tuple[float, float]:
    return (math.log(a_lower / b_upper), math.log(a_upper / b_lower))


def _write_csv(rows: list[dict[str, float | str]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "omitted_support_bounds_summary.csv"
    fieldnames = [
        "target_pair",
        "union_support_share",
        "target_a_support_share",
        "target_b_support_share",
        "common_support_log_ratio",
        "full_target_lower_log_ratio",
        "full_target_upper_log_ratio",
        "target_a_monotonicity_min_share",
        "target_b_monotonicity_min_share",
        "target_a_max_positive_diff",
        "target_b_max_positive_diff",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _write_table(rows: list[dict[str, float | str]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "omitted_support_bounds_table.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Model-implied omitted-support bounds from monotone lower-tail completion}",
        r"\label{tab:omitted-support-bounds}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.23\textwidth}cccccc}",
        r"\toprule",
        r"Target pair & Union share & $A$ share & $B$ share & Common-support $\Delta_{AB}$ & Monotone full-target interval & Min nonincreasing share \\",
        r"\midrule",
    ]
    for row in rows:
        min_share = min(float(row["target_a_monotonicity_min_share"]), float(row["target_b_monotonicity_min_share"]))
        lines.append(
            f"{row['target_pair']} & {row['union_support_share']:.4f} & {row['target_a_support_share']:.4f} & {row['target_b_support_share']:.4f} & "
            f"{row['common_support_log_ratio']:.5f} & [{row['full_target_lower_log_ratio']:.5f},\\ {row['full_target_upper_log_ratio']:.5f}] & {min_share:.4f} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize Notes. The table tightens the generic common-support decomposition in Proposition~\ref{prop:common-support-selection} inside the maintained five-state hump calibration. For each PVEU target pair, the omitted cells all lie below the first supported liquid-asset point in the relevant age-state blocks under the verified current-cell map. We therefore complete the omitted mass using a monotone lower-tail restriction on the maintained point-summary normalized multiplier: within each relevant age-state block, $\eta$ is taken to be weakly decreasing in current liquid assets. The lower endpoint uses the first supported boundary value in each omitted block; the upper endpoint uses the maximum value over the omitted lower-tail in that block. The last column reports the minimum share of nonpositive asset-to-asset differences in the maintained point summary across the relevant age-state blocks. In this calibration those shares are at least $0.9933$ for the warm-up pair and at least $0.9978$ for the nearby pair, so the lower-tail monotonicity completion is numerically well aligned with the maintained point summary.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    calibration = richer_state_calibration()
    survival = survival_probabilities(calibration)
    solution = solve_household(calibration, survival)
    joint = stationary_joint_distribution(calibration, solution, survival)
    ages = np.asarray(calibration.ages, dtype=int)

    groups = natural_state_groups(calibration)
    ratio = required_eta_ratio(calibration, implemented_wedge(calibration, solution), calibration.interest_rate)
    matching = build_matching_structure(calibration, solution, joint, state_groups=groups)
    eta = reconstruct_eta_profile(ratio, matching.older_asset_index, matching.older_state_index)
    support_mask = support_overlap_mask(solution, joint, groups)

    target_pairs = [
        ("Warm-up pair", (20, 29), (len(calibration.income_states) // 2,), (40, 49), None),
        ("Nearby pair", (25, 34), (len(calibration.income_states) // 2,), (45, 54), None),
    ]

    rows: list[dict[str, float | str]] = []
    for label, a_bounds, a_states, b_bounds, b_states in target_pairs:
        a_bounds_summary = _monotone_lower_tail_completion(eta, joint, support_mask, ages, a_bounds, a_states)
        b_bounds_summary = _monotone_lower_tail_completion(eta, joint, support_mask, ages, b_bounds, b_states)

        support_selector_a = _policy_selector(joint.shape, ages, a_bounds, a_states)
        support_selector_b = _policy_selector(joint.shape, ages, b_bounds, b_states)
        support_ext = np.zeros_like(joint, dtype=bool)
        support_ext[:-1, :, :] = support_mask
        support_ext[-1, :, :] = True
        union_selector = support_selector_a | support_selector_b
        union_mass = float(np.sum(joint[union_selector]))
        union_support_mass = float(np.sum(joint[union_selector & support_ext]))
        union_support_share = union_support_mass / union_mass if union_mass > 0.0 else 0.0

        common_support_log_ratio = math.log(a_bounds_summary["support_score"] / b_bounds_summary["support_score"])
        lower_log_ratio, upper_log_ratio = _log_ratio_interval(
            a_bounds_summary["full_lower_score"],
            a_bounds_summary["full_upper_score"],
            b_bounds_summary["full_lower_score"],
            b_bounds_summary["full_upper_score"],
        )

        rows.append(
            {
                "target_pair": label,
                "union_support_share": union_support_share,
                "target_a_support_share": a_bounds_summary["target_support_share"],
                "target_b_support_share": b_bounds_summary["target_support_share"],
                "common_support_log_ratio": common_support_log_ratio,
                "full_target_lower_log_ratio": lower_log_ratio,
                "full_target_upper_log_ratio": upper_log_ratio,
                "target_a_monotonicity_min_share": a_bounds_summary["min_nonincreasing_share"],
                "target_b_monotonicity_min_share": b_bounds_summary["min_nonincreasing_share"],
                "target_a_max_positive_diff": a_bounds_summary["max_positive_diff"],
                "target_b_max_positive_diff": b_bounds_summary["max_positive_diff"],
            }
        )

    csv_path = _write_csv(rows)
    tex_path = _write_table(rows)

    row_by_label = {row["target_pair"]: row for row in rows}
    write_tex_macros(
        OUTPUT_DIR / "generated_numbers_omitted_support.tex",
        {
            "WarmupCommonSupportDelta": tex_decimal(float(row_by_label["Warm-up pair"]["common_support_log_ratio"]), 5),
            "WarmupMonotoneFullTargetInterval": tex_interval(
                float(row_by_label["Warm-up pair"]["full_target_lower_log_ratio"]),
                float(row_by_label["Warm-up pair"]["full_target_upper_log_ratio"]),
                5,
            ),
            "NearbyCommonSupportDelta": tex_decimal(float(row_by_label["Nearby pair"]["common_support_log_ratio"]), 5),
            "NearbyMonotoneFullTargetInterval": tex_interval(
                float(row_by_label["Nearby pair"]["full_target_lower_log_ratio"]),
                float(row_by_label["Nearby pair"]["full_target_upper_log_ratio"]),
                5,
            ),
        },
    )

    print("Omitted-support bounds executed successfully.")
    print(f"CSV written to: {csv_path}")
    print(f"TeX table written to: {tex_path}")


if __name__ == "__main__":
    main()
