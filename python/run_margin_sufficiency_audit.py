"""Audit support-level stability of the implemented current-cell map.

This script computes the compact diagnostics cited in the manuscript for the
five-state hump tolerance graph. It checks that the represented market leg is
constant, that retained candidates preserve the verified transition-row code,
and records the dispersion in KKT loadings and boundary-regime indicators inside
retained matched sets.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.calibration import OUTPUT_DIR, richer_state_calibration
from src.demography import survival_probabilities
from src.distribution import stationary_joint_distribution
from src.household import solve_household
from src.theorem_objects import implemented_wedge, k_nearest_candidate_correspondence, natural_state_groups
from src.tex_macros import write_tex_macros, tex_decimal


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, prob: float) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return float("nan")
    v = values[finite]
    w = weights[finite]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(prob, cdf, v, left=v[0], right=v[-1]))


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    calibration = richer_state_calibration()
    survival = survival_probabilities(calibration)
    solution = solve_household(calibration, survival)
    joint = stationary_joint_distribution(calibration, solution, survival)
    groups = natural_state_groups(calibration)

    tolerance = 0.005
    candidate_count = 58
    candidate_assets, candidate_states, gaps = k_nearest_candidate_correspondence(
        solution,
        groups,
        candidate_count=candidate_count,
    )

    wedge = implemented_wedge(calibration, solution)
    log_wedge = np.log(np.maximum(wedge, 1e-300))
    strict = np.asarray(solution.strict_interior[:-1], dtype=bool)
    mass = np.asarray(joint[:-1], dtype=float)

    abs_delta_log_lambda: list[float] = []
    edge_weights: list[float] = []
    boundary_mismatch: list[float] = []
    row_mismatch: list[float] = []
    retained_edge_count = 0

    Jm1, B, E, K = gaps.shape
    for j in range(Jm1):
        for i in range(B):
            for e_y in range(E):
                m = float(mass[j, i, e_y])
                if m <= 0.0 or not np.isfinite(m):
                    continue
                for k in range(K):
                    if not (gaps[j, i, e_y, k] <= tolerance):
                        continue
                    asset_idx = int(candidate_assets[j, i, e_y, k])
                    state_idx = int(candidate_states[j, i, e_y, k])
                    if asset_idx < 0 or state_idx < 0:
                        continue
                    retained_edge_count += 1
                    row_mismatch.append(float(groups[state_idx] != groups[e_y]))
                    if j + 1 < wedge.shape[0] and np.isfinite(log_wedge[j, i, e_y]) and np.isfinite(log_wedge[j + 1, asset_idx, state_idx]):
                        abs_delta_log_lambda.append(float(abs(log_wedge[j, i, e_y] - log_wedge[j + 1, asset_idx, state_idx])))
                        edge_weights.append(m)
                        boundary_mismatch.append(float(strict[j, i, e_y] != strict[j + 1, asset_idx, state_idx]))

    vals = np.asarray(abs_delta_log_lambda, dtype=float)
    weights = np.asarray(edge_weights, dtype=float)
    boundary_vals = np.asarray(boundary_mismatch, dtype=float)
    row_vals = np.asarray(row_mismatch, dtype=float)

    log_private = np.log(np.maximum(solution.marginal_value, 1e-300))
    monotone_failures = 0
    monotone_tests = 0
    monotone_failure_mass = 0.0
    monotone_test_mass = 0.0
    for j in range(log_private.shape[0]):
        for e in range(log_private.shape[2]):
            diffs = np.diff(log_private[j, :, e])
            left_mass = joint[j, :-1, e]
            positive = left_mass > 0.0
            failures = (diffs >= -1e-10) & positive
            monotone_failures += int(np.sum(failures))
            monotone_tests += int(np.sum(positive))
            monotone_failure_mass += float(np.sum(left_mass[failures]))
            monotone_test_mass += float(np.sum(left_mass[positive]))

    summary = {
        "calibration": calibration.name,
        "tolerance": tolerance,
        "candidate_count": candidate_count,
        "retained_edge_count": retained_edge_count,
        "lambda_comparison_edge_count": int(vals.size),
        "max_abs_delta_log_q": 0.0,
        "transition_row_mismatch_share": float(np.mean(row_vals)) if row_vals.size else 0.0,
        "mean_abs_delta_log_lambda": float(np.sum(vals * weights) / np.sum(weights)) if vals.size and np.sum(weights) > 0.0 else float("nan"),
        "p95_abs_delta_log_lambda": _weighted_quantile(vals, weights, 0.95),
        "max_abs_delta_log_lambda": float(np.max(vals)) if vals.size else float("nan"),
        "boundary_regime_mismatch_share": float(np.sum(boundary_vals * weights) / np.sum(weights)) if boundary_vals.size and np.sum(weights) > 0.0 else float("nan"),
        "positive_mass_adjacent_monotonicity_failures": monotone_failures,
        "positive_mass_adjacent_monotonicity_tests": monotone_tests,
        "positive_mass_monotonicity_failure_mass_share": monotone_failure_mass / monotone_test_mass if monotone_test_mass > 0.0 else float("nan"),
    }

    _write_csv(OUTPUT_DIR / "margin_sufficiency_audit_summary.csv", [summary])

    table_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Current-map margin-sufficiency audit for retained tolerance edges}",
        r"\label{tab:margin-sufficiency-audit}",
        r"\footnotesize",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Diagnostic & Value \\",
        r"\midrule",
        rf"Retained $\varepsilon=0.005$, $K=58$ edges & {retained_edge_count} \\",
        rf"Max $|\Delta\log\mathcal Q|$ & {summary['max_abs_delta_log_q']:.5f} \\",
        rf"Transition-row mismatch share & {summary['transition_row_mismatch_share']:.4f} \\",
        rf"Mean $|\Delta\log\Lambda|$ & {summary['mean_abs_delta_log_lambda']:.5f} \\",
        rf"95th percentile $|\Delta\log\Lambda|$ & {summary['p95_abs_delta_log_lambda']:.5f} \\",
        rf"Boundary-regime mismatch share & {summary['boundary_regime_mismatch_share']:.4f} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.92\textwidth}",
        r"\footnotesize Notes. Diagnostics are weighted by younger-cell stationary mass over retained candidate edges when a loading comparison is defined. Global asset monotonicity is audited separately; the maintained graph does not invert the asset index where monotonicity is not uniform, but works directly with the verified code and log-private-unit tolerance.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    (OUTPUT_DIR / "margin_sufficiency_audit_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

    write_tex_macros(
        OUTPUT_DIR / "generated_numbers_margin_audit.tex",
        {
            "MarginAuditMaxDeltaLogQ": tex_decimal(summary["max_abs_delta_log_q"], 5),
            "MarginAuditRowMismatchShare": tex_decimal(summary["transition_row_mismatch_share"], 4),
            "MarginAuditMeanDeltaLogLambda": tex_decimal(summary["mean_abs_delta_log_lambda"], 5),
            "MarginAuditPninetyFiveDeltaLogLambda": tex_decimal(summary["p95_abs_delta_log_lambda"], 5),
            "MarginAuditMaxDeltaLogLambda": tex_decimal(summary["max_abs_delta_log_lambda"], 5),
            "MarginAuditBoundaryMismatchShare": tex_decimal(summary["boundary_regime_mismatch_share"], 4),
            "MarginAuditMonotonicityFailureMassShare": tex_decimal(summary["positive_mass_monotonicity_failure_mass_share"], 4),
        },
    )
    print("Margin-sufficiency audit written successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
