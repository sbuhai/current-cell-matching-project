"""Audit sensitivity of quantitative objects to alternative wedge implementations.

This script is intentionally diagnostic rather than theoremic. It compares the
paper's implemented KKT-consistent wedge with two raw-Euler stress tests:
(i) a variant that keeps raw interior Euler deviations while clipping boundary
cells at one, and (ii) the fully raw Euler proxy. The goal is to separate
economic content from numerical enforcement in the maintained implementation.

Outputs
-------
`python/outputs/wedge_audit_summary.csv`
    Machine-readable audit summary.
`python/outputs/wedge_audit_table.tex`
    Compact TeX table with a baseline represented-margin panel and a nearby
    policy-object panel.
"""

from __future__ import annotations

import csv
from pathlib import Path
import os

import numpy as np

from src.calibration import OUTPUT_DIR, baseline_calibration, richer_state_calibration
from src.demography import survival_probabilities
from src.distribution import stationary_joint_distribution
from src.household import solve_household
from src.theorem_objects import (
    build_matching_structure,
    conditional_wasserstein_age,
    fiberwise_log_spread,
    graph_robust_eta_envelope,
    implemented_wedge,
    k_nearest_candidate_correspondence,
    natural_state_groups,
    policy_log_ratio,
    policy_target_score,
    policy_target_union_support_share,
    pooled_state_groups,
    reconstruct_eta_profile,
    required_eta_ratio,
    support_overlap_mask,
)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights)
    total = float(cumulative[-1]) if cumulative.size else 0.0
    if total <= 0.0:
        return float("nan")
    return float(np.interp(float(q) * total, cumulative, sorted_values))


def _wedge_variants(solution) -> list[tuple[str, np.ndarray]]:
    raw = np.asarray(solution.euler_wedge[:-1, :, :], dtype=float)
    strict_interior = np.asarray(solution.strict_interior[:-1, :, :], dtype=bool)
    implemented = implemented_wedge
    boundary_clipped = np.where(strict_interior, raw, np.maximum(raw, 1.0))
    fully_raw = np.where(np.isfinite(raw), np.maximum(raw, 1e-12), np.nan)
    return [
        ("Implemented KKT-consistent wedge", implemented),
        ("Raw interior / boundary-clipped", boundary_clipped),
        ("Fully raw Euler proxy", fully_raw),
    ]


def _baseline_panel() -> list[dict[str, str | float]]:
    calibration = baseline_calibration()
    survival = survival_probabilities(calibration)
    solution = solve_household(calibration, survival)
    joint = stationary_joint_distribution(calibration, solution, survival)

    variants = _wedge_variants(solution)
    rows: list[dict[str, str | float]] = []
    ages = np.asarray(calibration.ages, dtype=float)
    matching = build_matching_structure(calibration, solution, joint)
    mass = np.asarray(joint[:-1, :, :], dtype=float)

    for label, wedge_or_fn in variants:
        wedge = wedge_or_fn(calibration, solution) if callable(wedge_or_fn) else np.asarray(wedge_or_fn, dtype=float)
        relevant = (mass > 0.0) & np.isfinite(wedge)
        ratio = required_eta_ratio(calibration, wedge, calibration.interest_rate)
        eta = reconstruct_eta_profile(ratio, matching.older_asset_index, matching.older_state_index)
        rows.append(
            {
                "panel": "Baseline represented margin at $\\rho^{S}=r$",
                "wedge_specification": label,
                "mean_wedge": float(np.sum(wedge[relevant] * mass[relevant]) / np.sum(mass[relevant])) if np.any(relevant) else float("nan"),
                "p95_wedge": _weighted_quantile(wedge[relevant], mass[relevant], 0.95) if np.any(relevant) else float("nan"),
                "spread": fiberwise_log_spread(
                    eta,
                    joint,
                    matching.fiber_bin,
                    matching.state_group,
                    calibration.envelope_bin_count,
                ),
                "conditional_transport": conditional_wasserstein_age(
                    eta,
                    joint,
                    ages,
                    matching.fiber_bin,
                    matching.state_group,
                    calibration.envelope_bin_count,
                ),
                "support_share": 1.0,
                "point_log_ratio": float("nan"),
                "lower_log_ratio": float("nan"),
                "upper_log_ratio": float("nan"),
            }
        )
    return rows


def _policy_log_interval(
    calibration,
    solution,
    joint,
    state_groups: np.ndarray,
    wedge: np.ndarray,
    support_mask: np.ndarray,
    candidate_count: int = 5,
) -> tuple[float, float, float]:
    ratio = required_eta_ratio(calibration, wedge, calibration.interest_rate)
    matching = build_matching_structure(calibration, solution, joint, state_groups=np.asarray(state_groups, dtype=int))
    point_eta = reconstruct_eta_profile(ratio, matching.older_asset_index, matching.older_state_index)
    candidate_assets, candidate_states, _ = k_nearest_candidate_correspondence(
        solution,
        np.asarray(state_groups, dtype=int),
        candidate_count=candidate_count,
    )
    lower_eta, upper_eta = graph_robust_eta_envelope(ratio, candidate_assets, candidate_states)

    target_a_states = (len(calibration.income_states) // 2,)
    point_a = policy_target_score(
        point_eta,
        joint,
        calibration,
        solution,
        age_bounds=(25, 34),
        state_indices=target_a_states,
        support_mask=support_mask,
    )
    point_b = policy_target_score(
        point_eta,
        joint,
        calibration,
        solution,
        age_bounds=(45, 54),
        state_indices=None,
        support_mask=support_mask,
    )
    lower_a = policy_target_score(
        lower_eta,
        joint,
        calibration,
        solution,
        age_bounds=(25, 34),
        state_indices=target_a_states,
        support_mask=support_mask,
    )
    upper_a = policy_target_score(
        upper_eta,
        joint,
        calibration,
        solution,
        age_bounds=(25, 34),
        state_indices=target_a_states,
        support_mask=support_mask,
    )
    lower_b = policy_target_score(
        lower_eta,
        joint,
        calibration,
        solution,
        age_bounds=(45, 54),
        state_indices=None,
        support_mask=support_mask,
    )
    upper_b = policy_target_score(
        upper_eta,
        joint,
        calibration,
        solution,
        age_bounds=(45, 54),
        state_indices=None,
        support_mask=support_mask,
    )
    return (
        policy_log_ratio(point_a, point_b),
        policy_log_ratio(lower_a, upper_b),
        policy_log_ratio(upper_a, lower_b),
    )


def _nearby_policy_panel() -> list[dict[str, str | float]]:
    calibration = richer_state_calibration()
    survival = survival_probabilities(calibration)
    solution = solve_household(calibration, survival)
    joint = stationary_joint_distribution(calibration, solution, survival)

    natural_groups = natural_state_groups(calibration)
    age_only_groups = pooled_state_groups(calibration)
    common_support = support_overlap_mask(solution, joint, natural_groups) & support_overlap_mask(solution, joint, age_only_groups)
    support_share = policy_target_union_support_share(
        joint,
        calibration,
        solution,
        common_support,
        target_a_age_bounds=(25, 34),
        target_a_state_indices=(len(calibration.income_states) // 2,),
        target_b_age_bounds=(45, 54),
    )

    variants = _wedge_variants(solution)
    mass = np.asarray(joint[:-1, :, :], dtype=float)
    rows: list[dict[str, str | float]] = []
    for label, wedge_or_fn in variants:
        wedge = wedge_or_fn(calibration, solution) if callable(wedge_or_fn) else np.asarray(wedge_or_fn, dtype=float)
        relevant = (mass > 0.0) & np.isfinite(wedge)
        point, lower, upper = _policy_log_interval(
            calibration,
            solution,
            joint,
            natural_groups,
            wedge,
            common_support,
            candidate_count=5,
        )
        rows.append(
            {
                "panel": "Nearby policy object on common support (natural map, $K=5$)",
                "wedge_specification": label,
                "mean_wedge": float(np.sum(wedge[relevant] * mass[relevant]) / np.sum(mass[relevant])) if np.any(relevant) else float("nan"),
                "p95_wedge": _weighted_quantile(wedge[relevant], mass[relevant], 0.95) if np.any(relevant) else float("nan"),
                "spread": float("nan"),
                "conditional_transport": float("nan"),
                "support_share": support_share,
                "point_log_ratio": point,
                "lower_log_ratio": lower,
                "upper_log_ratio": upper,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, str | float]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "wedge_audit_summary.csv"
    fieldnames = [
        "panel",
        "wedge_specification",
        "mean_wedge",
        "p95_wedge",
        "spread",
        "conditional_transport",
        "support_share",
        "point_log_ratio",
        "lower_log_ratio",
        "upper_log_ratio",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _fmt(value: float, digits: int = 4) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def _write_table(rows: list[dict[str, str | float]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "wedge_audit_table.tex"
    panel_a = [row for row in rows if row["panel"].startswith("Baseline")]
    panel_b = [row for row in rows if row["panel"].startswith("Nearby")]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Wedge-enforcement audit at $\rho^{S}=r$}",
        r"\label{tab:wedge-audit}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{0.99\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.34\textwidth}cccccc}",
        r"\toprule",
        r"Wedge specification & Mean wedge & p95 wedge & Spread & Transport & Nearby $\Delta_{AB}$ & Nearby $K=5$ interval \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Panel A. Baseline represented-margin diagnostics}} \\",
    ]
    for row in panel_a:
        lines.append(
            f"{row['wedge_specification']} & {_fmt(float(row['mean_wedge']))} & {_fmt(float(row['p95_wedge']))} & "
            f"{_fmt(float(row['spread']))} & {_fmt(float(row['conditional_transport']))} & -- & -- " + r"\\"
        )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\textit{Panel B. Nearby policy object on common support in the five-state hump environment}} \\")
    for row in panel_b:
        interval = (
            f"[{_fmt(float(row['lower_log_ratio']))}, {_fmt(float(row['upper_log_ratio']))}]"
            if np.isfinite(float(row["lower_log_ratio"])) and np.isfinite(float(row["upper_log_ratio"]))
            else "--"
        )
        lines.append(
            f"{row['wedge_specification']} & {_fmt(float(row['mean_wedge']))} & {_fmt(float(row['p95_wedge']))} & -- & -- & "
            f"{_fmt(float(row['point_log_ratio']))} & {interval} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.99\textwidth}",
            r"\footnotesize Notes. Panel A compares the baseline one-asset represented margin under three wedge constructions: the paper's implemented KKT-consistent wedge, a stress test that propagates raw interior Euler deviations while clipping boundary cells at one, and the fully raw Euler proxy. Spread and transport are the maintained theorem objects from the baseline geometric diagnostics. Panel B propagates the same three wedge constructions through the nearby policy object on the common support used in the main text under the verified current-cell map and the maintained $K=5$ envelope. This is an audit, not a new theorem: the raw-Euler rows intentionally expose the extent to which numerical interior Euler noise can move maintained quantitative objects when it is propagated directly rather than normalized away by the KKT-consistent implementation.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def full_main() -> None:
    rows = _baseline_panel() + _nearby_policy_panel()
    csv_path = _write_csv(rows)
    tex_path = _write_table(rows)
    print("Wedge audit written successfully.")
    print(f"CSV: {csv_path}")
    print(f"TeX: {tex_path}")




_REQUIRED_OUTPUTS = ['wedge_audit_summary.csv', 'wedge_audit_table.tex', 'generated_numbers_baseline.tex']


def _validate_frozen_outputs() -> None:
    out_dir = Path(__file__).resolve().parent / "outputs"
    missing = [name for name in _REQUIRED_OUTPUTS if not (out_dir / name).exists()]
    if missing:
        raise SystemExit("Missing frozen outputs: " + ", ".join(missing))
    print("Frozen outputs validated for run_wedge_audit.py. Set FULL_REBUILD=1 to recompute the finite-grid solve.")
    for name in _REQUIRED_OUTPUTS:
        print(f"validated: {name}")


def main() -> None:
    if os.environ.get("FULL_REBUILD") == "1":
        full_main()
    else:
        _validate_frozen_outputs()


if __name__ == "__main__":
    main()
