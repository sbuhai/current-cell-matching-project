"""Reporting utilities for manuscript figures, tables, and machine-readable summaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .calibration import Calibration, OUTPUT_DIR, SOURCE_NOTES
from .theorem_objects import (
    AlternativeNormalizationSummary,
    ApproximateSufficiencySummary,
    ComparisonResults,
    EnvelopeWidthSensitivitySummary,
    ExcludedMassSummary,
    GraphEnvelopeSummary,
    NetTermAgeSummary,
    KappaSensitivitySummary,
    MatchingStructure,
    PolicyDecompositionSummary,
    PolicyRankingSummary,
    SensitivityResult,
    WedgeDiagnostics,
    age_averaged_eta,
    age_averaged_required_ratio,
    age_quantile_required_ratio,
)


def ensure_output_dir() -> Path:
    """Create the output directory if it does not already exist."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _to_percent_log_points(value: float) -> float:
    """Convert natural-log units to percent log points."""

    return 100.0 * float(value)



def _save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    """Save a Matplotlib figure as both PDF and PNG."""

    out_dir = ensure_output_dir()
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path



def write_calibration_summary(calibration: Calibration) -> Path:
    """Write a machine-readable calibration summary."""

    out = ensure_output_dir() / f"{calibration.name}_calibration_summary.csv"
    earnings_note = (
        SOURCE_NOTES["earnings_two_state"]
        if len(calibration.income_states) == 2
        else SOURCE_NOTES["earnings_five_state"]
    )
    efficiency_note = (
        SOURCE_NOTES["flat_efficiency"]
        if calibration.age_efficiency_mode == "flat"
        else SOURCE_NOTES["hump_efficiency"]
    )
    rows = [
        ("entry_age", calibration.min_age, SOURCE_NOTES["entry_retirement_terminal"]),
        ("retirement_age", calibration.retirement_age, SOURCE_NOTES["entry_retirement_terminal"]),
        ("terminal_age", calibration.max_age, SOURCE_NOTES["entry_retirement_terminal"]),
        ("population_growth", calibration.population_growth, SOURCE_NOTES["population_growth"]),
        ("beta", calibration.beta, SOURCE_NOTES["beta_sigma_r"]),
        ("sigma", calibration.sigma, SOURCE_NOTES["beta_sigma_r"]),
        ("interest_rate", calibration.interest_rate, SOURCE_NOTES["beta_sigma_r"]),
        ("replacement_ratio", calibration.pension_replacement, SOURCE_NOTES["replacement_ratio"]),
        ("asset_size", calibration.asset_size, "Curved asset grid used in the finite-grid solution"),
        ("income_states", calibration.income_states, earnings_note),
        ("income_transition", calibration.income_transition, earnings_note),
        ("borrowing_limit", calibration.borrowing_limit, SOURCE_NOTES["borrowing_limit"]),
        ("age_efficiency_mode", calibration.age_efficiency_mode, efficiency_note),
        ("envelope_bin_count", calibration.envelope_bin_count, SOURCE_NOTES["envelope_bins"]),
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", "source_or_rationale"])
        writer.writerows(rows)
    return out



def write_matching_summary(calibration: Calibration, matching: MatchingStructure) -> Path:
    """Write the matching diagnostics used in the quantitative section."""

    out = ensure_output_dir() / f"{calibration.name}_matching_summary.csv"
    diagnostics = matching.diagnostics
    rows = [
        ("mean_abs_log_gap", diagnostics.mean_abs_log_gap),
        ("median_abs_log_gap", diagnostics.median_abs_log_gap),
        ("p95_abs_log_gap", diagnostics.p95_abs_log_gap),
        ("p99_abs_log_gap", diagnostics.p99_abs_log_gap),
        ("max_abs_log_gap", diagnostics.max_abs_log_gap),
        ("weighted_mean_abs_log_gap", diagnostics.weighted_mean_abs_log_gap),
        ("weighted_p95_abs_log_gap", diagnostics.weighted_p95_abs_log_gap),
        ("weighted_p99_abs_log_gap", diagnostics.weighted_p99_abs_log_gap),
        ("weighted_max_abs_log_gap", diagnostics.weighted_max_abs_log_gap),
        ("overall_support_overlap_share", diagnostics.overall_support_overlap_share),
        ("min_age_support_overlap_share", diagnostics.min_age_support_overlap_share),
        ("comparison_attribute", matching.group_name),
        ("pairwise_matching_rule", matching.rule_name),
        ("aggregation_fibers", f"{matching.group_name}_x_{calibration.envelope_bin_count}_log_contv_bins"),
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "value"])
        writer.writerows(rows)
    return out



def write_wedge_summary(calibration: Calibration, diagnostics: WedgeDiagnostics) -> Path:
    """Write raw-versus-implemented wedge diagnostics."""

    out = ensure_output_dir() / f"{calibration.name}_wedge_summary.csv"
    rows = [
        ("raw_all_mean", diagnostics.raw_all.mean),
        ("raw_all_median", diagnostics.raw_all.median),
        ("raw_all_p05", diagnostics.raw_all.p05),
        ("raw_all_p95", diagnostics.raw_all.p95),
        ("raw_all_p99", diagnostics.raw_all.p99),
        ("raw_all_max", diagnostics.raw_all.maximum),
        ("raw_all_subunit_mass_share", diagnostics.raw_all.subunit_mass_share),
        ("raw_strict_interior_mean", diagnostics.raw_strict_interior.mean),
        ("raw_strict_interior_median", diagnostics.raw_strict_interior.median),
        ("raw_strict_interior_p05", diagnostics.raw_strict_interior.p05),
        ("raw_strict_interior_p95", diagnostics.raw_strict_interior.p95),
        ("raw_strict_interior_p99", diagnostics.raw_strict_interior.p99),
        ("raw_strict_interior_max", diagnostics.raw_strict_interior.maximum),
        ("raw_strict_interior_subunit_mass_share", diagnostics.raw_strict_interior.subunit_mass_share),
        ("raw_strict_interior_mean_abs_deviation", diagnostics.raw_strict_interior_mean_abs_deviation),
        ("raw_strict_interior_p95_abs_deviation", diagnostics.raw_strict_interior_p95_abs_deviation),
        ("raw_strict_interior_p99_abs_deviation", diagnostics.raw_strict_interior_p99_abs_deviation),
        ("implemented_mean", diagnostics.implemented.mean),
        ("implemented_median", diagnostics.implemented.median),
        ("implemented_p95", diagnostics.implemented.p95),
        ("implemented_p99", diagnostics.implemented.p99),
        ("implemented_max", diagnostics.implemented.maximum),
        ("strict_interior_mass_share", diagnostics.strict_interior_mass_share),
        ("lower_bound_mass_share", diagnostics.lower_bound_mass_share),
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "value"])
        writer.writerows(rows)
    return out



def write_comparison_summary(calibration: Calibration, results: Sequence[ComparisonResults]) -> Path:
    """Write the rate-by-rate theorem summary for the baseline grid."""

    out = ensure_output_dir() / f"{calibration.name}_comparison_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "social_rate",
                "gross_social_factor",
                "mean_required_ratio",
                "peak_age_average_ratio",
                "spread",
                "conditional_wasserstein",
                "epsilon_adjusted_spread",
                "epsilon_adjusted_conditional_wasserstein",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.social_rate,
                    result.gross_social_factor,
                    result.mean_required_ratio,
                    result.peak_age_average_ratio,
                    result.spread,
                    result.conditional_wasserstein,
                    result.epsilon_adjusted_spread,
                    result.epsilon_adjusted_conditional_wasserstein,
                ]
            )
    return out



def write_results_table_tex(calibration: Calibration, results: Sequence[ComparisonResults]) -> Path:
    """Write the LaTeX table used for the baseline quantitative summary."""

    out = ensure_output_dir() / f"{calibration.name}_results_table.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline lower-envelope summary by candidate annual social discount rate}",
        r"\label{tab:baseline-results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Candidate $\rho^{S}$ & Gross $\mathcal{D}^{S}$ & Mean $\eta^{y}/\eta^{o}$ & Peak age-profile $\eta^{y}/\eta^{o}$ & $\widehat{\Spread}$ & $\widehat{\DemDist}$ " + "\\\\",
        r"\midrule",
    ]
    for result in results:
        lines.append(
            f"{100.0 * result.social_rate:0.2f}\\% & {result.gross_social_factor:0.4f} & {result.mean_required_ratio:0.4f} & {result.peak_age_average_ratio:0.4f} & {result.spread:0.4f} & {result.conditional_wasserstein:0.4f} " + "\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}",
            r"\vspace{0.35em}",
            r"\footnotesize Notes. The table reports one maintained point-selector lower-envelope realization generated by the maintained baseline comparison system. $\widehat{\Spread}$ is the maximal fiberwise log-spread implied by the maintained comparison graph, and $\widehat{\DemDist}$ is the corresponding conditional transport-distance summary. Both are geometric diagnostics on the maintained graph, not identified demographic weights.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out



def write_long_horizon_summary(calibration: Calibration, results: Sequence[ComparisonResults], mu: np.ndarray) -> Path:
    """Write mass-weighted long-horizon normalized-weight ratios by social rate."""

    out = ensure_output_dir() / f"{calibration.name}_long_horizon_summary.csv"
    ages = np.array(calibration.ages, dtype=int)
    targets = [65, 75, 85]
    target_idx = {age: int(np.where(ages == age)[0][0]) for age in targets}
    age20_idx = int(np.where(ages == 20)[0][0])

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "social_rate",
            "mean_eta_age20_over_age65",
            "mean_eta_age20_over_age75",
            "mean_eta_age20_over_age85",
        ])
        for result in results:
            age_mass = np.maximum(mu.sum(axis=(1, 2)), 1e-300)
            age_mean_eta = (result.eta_profile * mu).sum(axis=(1, 2)) / age_mass
            writer.writerow([
                result.social_rate,
                age_mean_eta[age20_idx] / age_mean_eta[target_idx[65]],
                age_mean_eta[age20_idx] / age_mean_eta[target_idx[75]],
                age_mean_eta[age20_idx] / age_mean_eta[target_idx[85]],
            ])
    return out



def write_sensitivity_summary(calibration: Calibration, rows: Sequence[SensitivityResult]) -> Path:
    """Write graph-sensitivity diagnostics across social rates."""

    out = ensure_output_dir() / f"{calibration.name}_sensitivity_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["social_rate", "specification", "mean_required_ratio", "peak_age_average_ratio", "spread", "conditional_wasserstein"])
        for row in rows:
            writer.writerow([row.social_rate, row.specification, row.mean_required_ratio, row.peak_age_average_ratio, row.spread, row.conditional_wasserstein])
    return out



def write_sensitivity_table_tex(calibration: Calibration, rows: Sequence[SensitivityResult]) -> Path:
    """Write the appendix table on matching-rule sensitivity away from $\rho^S=r$."""

    out = ensure_output_dir() / f"{calibration.name}_sensitivity_table.tex"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Sensitivity of lower-envelope geometry to matching graphs at three social discount rates}",
        r"\label{tab:baseline-sensitivity}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"$\rho^{S}$ & Matching rule & Mean $\eta^{y}/\eta^{o}$ & Peak age ratio & $\widehat{\Spread}$ & $\widehat{\DemDist}$ " + "\\\\",
        r"\midrule",
    ]
    current_rate = None
    for row in rows:
        rate_label = f"{100.0 * row.social_rate:0.2f}\\%"
        display_rate = rate_label if current_rate != row.social_rate else ""
        current_rate = row.social_rate
        lines.append(
            f"{display_rate} & {row.specification} & {row.mean_required_ratio:0.4f} & {row.peak_age_average_ratio:0.4f} & {row.spread:0.4f} & {row.conditional_wasserstein:0.4f} " + "\\\\"
        )
        if row.specification.startswith("Upper"):
            lines.append(r"\addlinespace[0.15em]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\textwidth}",
            r"\footnotesize Notes. The table varies the one-step matching graph while holding fixed the benchmark environment and the earnings-state comparison map. Reporting rates away from $\rho^{S}=r$ avoids the special case in which slack interior cells propagate $\eta=1$ exactly and the spread is pinned by the borrowing-constraint chain at $b=0$.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out



def write_excluded_mass_summary(calibration: Calibration, rows: Sequence[ExcludedMassSummary]) -> Path:
    """Write a machine-readable summary of excluded mass in the hardest environment."""

    out = ensure_output_dir() / f"{calibration.name}_excluded_mass_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["age_block", "earnings_group", "low_asset_share", "mid_asset_share", "high_asset_share", "total_share_of_excluded"])
        for row in rows:
            writer.writerow([row.age_block, row.earnings_group, row.low_asset_share, row.mid_asset_share, row.high_asset_share, row.total_share_of_excluded])
    return out



def plot_required_eta_ratio_by_age(
    calibration: Calibration,
    results: Sequence[ComparisonResults],
    mu: np.ndarray,
) -> tuple[Path, Path]:
    """Plot the mass-weighted minimal younger-versus-older ratio by age."""

    ages = np.array(calibration.ages[:-1])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for result in results:
        age_avg = age_averaged_required_ratio(result.required_eta_ratio, mu)
        ax.plot(ages, age_avg, label=f"annual rate = {result.social_rate:.4f}")
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Age of the younger cell")
    ax.set_ylabel(r"Required $\eta^{y}/\eta^{o}$ ratio")
    ax.set_title("Minimal younger-versus-older normalized-weight ratio by age")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, f"{calibration.name}_required_eta_ratio_by_age")



def plot_required_eta_bands_by_age(
    calibration: Calibration,
    results: Sequence[ComparisonResults],
    mu: np.ndarray,
    selected_rates: Sequence[float] | None = None,
) -> tuple[Path, Path]:
    """Plot age-by-age heterogeneity bands for selected social discount rates."""

    if selected_rates is None:
        selected_rates = (0.0, calibration.interest_rate)
    result_map = {round(result.social_rate, 8): result for result in results}
    ages = np.array(calibration.ages[:-1])
    fig, axes = plt.subplots(1, len(selected_rates), figsize=(7.6, 3.6), constrained_layout=True)
    if len(selected_rates) == 1:
        axes = [axes]
    for ax, rate in zip(axes, selected_rates):
        result = result_map[round(float(rate), 8)]
        quantiles = age_quantile_required_ratio(result.required_eta_ratio, mu, np.array([0.10, 0.50, 0.90]))
        ax.fill_between(ages, quantiles[0], quantiles[2], alpha=0.25)
        ax.plot(ages, quantiles[1], linewidth=1.5)
        ax.axhline(1.0, linestyle="--", linewidth=1.0)
        label = r"$\rho^{S}=r$" if abs(rate - calibration.interest_rate) < 1e-10 else rf"$\rho^S={100.0 * rate:0.1f}\%$"
        ax.set_title(label)
        ax.set_xlabel("Age of the younger cell")
        ax.set_ylabel(r"Required $\eta^{y}/\eta^{o}$ ratio")
    return _save_figure(fig, f"{calibration.name}_required_eta_bands_by_age")



def plot_spread_distance_by_social_rate(
    calibration: Calibration,
    results: Sequence[ComparisonResults],
) -> tuple[Path, Path]:
    """Plot the spread and transport-distance summaries against the social rate."""

    social_rates = np.array([100.0 * result.social_rate for result in results])
    spreads = np.array([result.spread for result in results])
    distances = np.array([result.conditional_wasserstein for result in results])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
    axes[0].plot(social_rates, spreads, marker="o")
    axes[0].set_xlabel(r"Candidate annual $\rho^S$ (percent)")
    axes[0].set_ylabel("Spread summary")
    axes[0].set_title("Fiberwise log-spread")

    axes[1].plot(social_rates, distances, marker="o")
    axes[1].set_xlabel(r"Candidate annual $\rho^S$ (percent)")
    axes[1].set_ylabel("Transport distance")
    axes[1].set_title("Conditional transport distance")

    return _save_figure(fig, f"{calibration.name}_spread_distance_by_social_rate")



def plot_excluded_mass_heatmap(calibration: Calibration, rows: Sequence[ExcludedMassSummary]) -> tuple[Path, Path]:
    """Visualize excluded younger mass by age block, asset tercile, and earnings panel."""

    if not rows:
        fig, ax = plt.subplots(figsize=(4.0, 2.0))
        ax.axis("off")
        ax.text(0.5, 0.5, "No excluded mass in this specification.", ha="center", va="center")
        return _save_figure(fig, f"{calibration.name}_excluded_mass_heatmap")

    age_blocks = []
    earnings_groups = []
    for row in rows:
        if row.age_block not in age_blocks:
            age_blocks.append(row.age_block)
        if row.earnings_group not in earnings_groups:
            earnings_groups.append(row.earnings_group)

    fig, axes = plt.subplots(1, len(earnings_groups), figsize=(8.6, 3.2), constrained_layout=True)
    if len(earnings_groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, earnings_groups):
        matrix = np.zeros((len(age_blocks), 3), dtype=float)
        shares = np.zeros(len(age_blocks), dtype=float)
        for i, age_block in enumerate(age_blocks):
            match = [row for row in rows if row.age_block == age_block and row.earnings_group == group]
            if not match:
                continue
            entry = match[0]
            matrix[i, 0] = entry.low_asset_share
            matrix[i, 1] = entry.mid_asset_share
            matrix[i, 2] = entry.high_asset_share
            shares[i] = entry.total_share_of_excluded
        im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=max(0.6, float(np.nanmax(matrix))))
        ax.set_xticks([0, 1, 2], labels=["Low", "Middle", "High"])
        ax.set_yticks(np.arange(len(age_blocks)), labels=age_blocks)
        ax.set_title(group)
        ax.set_xlabel("Asset tercile")
        for i in range(len(age_blocks)):
            ax.text(2.45, i, f"{shares[i]:0.2f}", va="center", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("Age block")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Asset-tercile share within excluded block")
    return _save_figure(fig, f"{calibration.name}_excluded_mass_heatmap")


def write_net_term_sign_summary(
    rows: Sequence[NetTermAgeSummary],
    stem: str = "baseline_net_term_sign_summary",
) -> Path:
    """Write the age-block sign diagnostic for log Lambda minus mismatch."""

    out = ensure_output_dir() / f"{stem}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "age_block",
            "support_share",
            "positive_share",
            "near_zero_share",
            "negative_share",
            "mean_net_term_plp",
        ])
        for row in rows:
            writer.writerow([
                row.age_block,
                row.support_share,
                row.positive_share,
                row.near_zero_share,
                row.negative_share,
                _to_percent_log_points(row.mean_net_term),
            ])
    return out



def write_net_term_sign_table_tex(
    rows: Sequence[NetTermAgeSummary],
    stem: str = "baseline_net_term_sign_table",
) -> Path:
    """Write the age-block sign diagnostic table for the appendix."""

    out = ensure_output_dir() / f"{stem}.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Age-block sign diagnostic for $\log \Lambda-\varepsilon$ in the baseline benchmark}",
        r"\label{tab:baseline-net-term-sign}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Age block & On-support share & Positive & Near zero & Negative & Mean $\log\Lambda-\varepsilon$ (plp) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.age_block} & {row.support_share:.4f} & {row.positive_share:.4f} & "
            f"{row.near_zero_share:.4f} & {row.negative_share:.4f} & "
            f"{_to_percent_log_points(row.mean_net_term):.2f} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.90\textwidth}",
            r"\footnotesize Notes. ``On-support share'' is the share of younger mass in the age block that lies on one-step overlap support under the verified current-cell comparison map. Positive, near-zero, and negative columns are shares of on-support younger mass classified by the sign of $\log\Lambda-\varepsilon$. Values in the final column are reported in percent log points.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_support_sufficiency_summary(rows: Sequence[ApproximateSufficiencySummary], stem: str = "support_sufficiency_frontier") -> Path:
    """Write the support-versus-sufficiency diagnostics as CSV."""

    out = ensure_output_dir() / f"{stem}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "label",
            "partition_label",
            "bin_count",
            "overall_support_overlap_share",
            "weighted_mean_oscillation_plp",
            "weighted_p95_oscillation_plp",
            "weighted_p99_oscillation_plp",
            "max_oscillation_plp",
            "weighted_mean_kappa_plp",
            "weighted_p95_kappa_plp",
            "weighted_p99_kappa_plp",
            "max_kappa_plp",
        ])
        for row in rows:
            writer.writerow([
                row.label,
                row.partition_label,
                row.bin_count,
                row.overall_support_overlap_share,
                _to_percent_log_points(row.weighted_mean_oscillation),
                _to_percent_log_points(row.weighted_p95_oscillation),
                _to_percent_log_points(row.weighted_p99_oscillation),
                _to_percent_log_points(row.max_oscillation),
                _to_percent_log_points(row.weighted_mean_kappa),
                _to_percent_log_points(row.weighted_p95_kappa),
                _to_percent_log_points(row.weighted_p99_kappa),
                _to_percent_log_points(row.max_kappa),
            ])
    return out


def plot_support_sufficiency_frontier(rows: Sequence[ApproximateSufficiencySummary], stem: str = "support_sufficiency_frontier") -> tuple[Path, Path]:
    """Plot mean and worst-case overlap-contamination frontiers across coarsened maps."""

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    x = np.array([100.0 * row.overall_support_overlap_share for row in rows], dtype=float)
    y_mean = np.array([100.0 * row.weighted_mean_kappa for row in rows], dtype=float)
    y_max = np.array([100.0 * row.max_kappa for row in rows], dtype=float)

    display_labels = {
        "Verified current-cell map": "Verified map",
        "Three-block coarsening": "Three-block",
        "Standardized heuristic map": "Standardized",
        "Age-based current matching": "Age-based",
    }
    annotation_specs = {
        "mean": {
            "Verified current-cell map": ((4, 2), "left"),
            "Three-block coarsening": ((4, 4), "left"),
            "Standardized heuristic map": ((3, -4), "left"),
            "Age-based current matching": ((5, 7), "left"),
        },
        "max": {
            "Verified current-cell map": ((4, 2), "left"),
            "Three-block coarsening": ((4, 4), "left"),
            "Standardized heuristic map": ((0, -10), "center"),
            "Age-based current matching": ((12, 2), "left"),
        },
    }

    panels = [
        (axes[0], y_mean, r"Mean estimated contamination radius $\hat\kappa^{\mathrm{mean}}$ (percent log points)", "mean"),
        (axes[1], y_max, r"Worst-case contamination radius $\hat\kappa^{\max}$ (percent log points)", "max"),
    ]
    for ax, y, ylabel, panel_key in panels:
        ax.scatter(x, y, s=28)
        for xi, yi, row in zip(x, y, rows):
            (dx, dy), ha = annotation_specs.get(panel_key, {}).get(row.label, ((5, 5), "left"))
            ax.annotate(
                display_labels.get(row.label, row.label),
                (xi, yi),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                ha=ha,
            )
        ax.set_xlabel("One-step support overlap of younger mass (percent)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(max(0.0, np.min(x) - 3.0), min(100.0, np.max(x) + 5.5))
        ymax = float(np.max(y)) if y.size else 0.0
        ax.set_ylim(0.0, max(0.6, ymax * 1.15 if ymax > 0.0 else 0.6))
    axes[0].set_title("Support-contamination tradeoff")
    axes[1].set_title("Theorem-relevant worst case")
    return _save_figure(fig, stem)


def write_support_sufficiency_audit_summary(
    rows: Sequence[ApproximateSufficiencySummary],
    stem: str = "support_sufficiency_audit",
) -> Path:
    """Write the partition-robust contamination audit as CSV."""

    out = ensure_output_dir() / f"{stem}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "label",
            "partition_label",
            "bin_count",
            "overall_support_overlap_share",
            "weighted_mean_kappa_plp",
            "max_kappa_plp",
        ])
        for row in rows:
            writer.writerow([
                row.label,
                row.partition_label,
                row.bin_count,
                row.overall_support_overlap_share,
                _to_percent_log_points(row.weighted_mean_kappa),
                _to_percent_log_points(row.max_kappa),
            ])
    return out



def write_support_sufficiency_audit_table_tex(
    rows: Sequence[ApproximateSufficiencySummary],
    stem: str = "support_sufficiency_audit_table",
) -> Path:
    """Write the partition-robust contamination audit table."""

    out = ensure_output_dir() / f"{stem}.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Partition-robust contamination audit in the five-state hump environment at $\rho^{S}=r$}",
        r"\label{tab:support-sufficiency-audit}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.22\textwidth}>{\raggedright\arraybackslash}p{0.36\textwidth}ccc}",
        r"\toprule",
        r"Comparison system & Operative partition audit & Overlap & Mean $\hat\kappa^{\mathrm{mean}}$ & Worst-case $\hat\kappa^{\max}$ \\",
        r"\midrule",
    ]
    grouped_labels: list[str] = []
    for row in rows:
        if row.label not in grouped_labels:
            grouped_labels.append(row.label)
    for label in grouped_labels:
        match_rows = [row for row in rows if row.label == label]
        if not match_rows:
            continue
        for idx, row in enumerate(match_rows):
            label_cell = row.label if idx == 0 else ""
            partition_label_tex = row.partition_label.replace("N_q", "$N_q$")
            lines.append(
                f"{label_cell} & {partition_label_tex} & {row.overall_support_overlap_share:.4f} & "
                f"{_to_percent_log_points(row.weighted_mean_kappa):.2f} & "
                f"{_to_percent_log_points(row.max_kappa):.2f} " + r"\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.97\textwidth}",
            r"\footnotesize Notes. The table audits the contamination diagnostics for each coarsened current comparison map under the baseline operative partition, nested-bin variants, shifted weighted-quantile bins, age-specific fixed cutoffs, and fully pooled cutoffs. Overlap is repeated because it is a property of the current comparison map rather than of the partition. The theorem-relevant object is the worst-case radius $\hat\kappa^{\max}$; the mean column is a descriptive mass-weighted average radius. Under the midpoint benchmark used in the theorem, $\hat\kappa^{\max}$ equals one half of the corresponding within-partition oscillation of the theorem-relevant benchmark term.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _policy_table_tex_lines(
    rows: Sequence[PolicyRankingSummary],
    decomposition: PolicyDecompositionSummary,
    *,
    caption: str,
    label: str,
    policy_a_desc: str,
    policy_b_desc: str,
    extra_note: str = "",
) -> list[str]:
    """Assemble a policy-ranking table with a common-support decomposition."""

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.20\textwidth}>{\raggedright\arraybackslash}p{0.16\textwidth}ccccccc}",
        r"\toprule",
        r"Comparison system & Support set & Union share & $A$ share & $B$ share & Mean $\hat\kappa^{\mathrm{mean}}$ & Worst-case $\hat\kappa^{\max}$ & $\Delta_{AB}=\log(\bar\eta(A)/\bar\eta(B))$ & Diagnostic interval \\",
        r"\midrule",
        r"\multicolumn{9}{l}{\textit{Panel A. Scale-free target comparison}} \\",
    ]
    for row in rows:
        lines.append(
            f"{row.label} & {row.support_label} & {row.support_share:.4f} & "
            f"{row.target_a_support_share:.4f} & {row.target_b_support_share:.4f} & "
            f"{_to_percent_log_points(row.kappa_mean):.2f} & "
            f"{_to_percent_log_points(row.kappa_max):.2f} & "
            f"{row.log_ratio:.5f} & [{row.lower_log_ratio:.5f},\\ {row.upper_log_ratio:.5f}] " + r"\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{9}{l}{\textit{Panel B. Decomposition of the shift from age-based full support to verified current-cell common support}} \\",
            r"Component & \multicolumn{8}{l}{Point contribution to $\Delta_{AB}$ (log points)} \\",
            f"Support exclusion & \\multicolumn{{8}}{{l}}{{{decomposition.support_exclusion:.5f}}} " + r"\\",
            f"Within-support rematching & \\multicolumn{{8}}{{l}}{{{decomposition.within_support_rematching:.5f}}} " + r"\\",
            f"Total change & \\multicolumn{{8}}{{l}}{{{decomposition.total_change:.5f}}} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.98\textwidth}",
            rf"\footnotesize Notes. Policy $A$ is {policy_a_desc}. Policy $B$ is {policy_b_desc}. "
            r"Both are cross-sectional target-group reallocations evaluated on the same maintained one-period financing margin; the common financing rule defines the comparison margin and does not mean that the policies are literally self-transfers. "
            r"The statistic $\Delta_{AB}$ is scale-free: positive values favor target $A$, negative values favor target $B$, and zero is the knife-edge. "
            r"``Common support'' is the intersection of the one-step support sets under verified current-cell and age-based current matching; the union, $A$, and $B$ support columns report how much of the target mass remains on the stated support set. "
            r"The contamination columns are reported in percent log points, whereas $\Delta_{AB}$, its diagnostic interval, and the decomposition in Panel B are reported in log points. "
            r"The diagnostic interval is the maintained identified set over all selectors from the five-nearest admissible older cells within the reported comparison system, conditional on that maintained admissible correspondence. "
            r"Panel B uses the point log-ratios in Panel A to decompose the shift into support exclusion and within-support rematching; it is computed from the underlying unrounded point statistics, so displayed entries need not match arithmetic on separately rounded numbers beyond the last shown decimal."
            + ((" " + extra_note) if extra_note else ""),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return lines


def write_policy_ranking_table_tex(
    rows: Sequence[PolicyRankingSummary],
    decomposition: PolicyDecompositionSummary,
    stem: str = "policy_ranking_table",
) -> Path:
    """Write the baseline policy-ranking table with common-support decomposition."""

    out = ensure_output_dir() / f"{stem}.tex"
    lines = _policy_table_tex_lines(
        rows,
        decomposition,
        caption="Scale-free local target comparison and common-support decomposition",
        label="tab:policy-ranking",
        policy_a_desc="a cross-sectional transfer targeted to ages $20$ to $29$ in the central current earnings row (third row, earnings level $0.965$), evaluated on the maintained one-period financing margin",
        policy_b_desc="the same cross-sectional transfer targeted to ages $40$ to $49$ on that maintained financing margin",
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_policy_robustness_table_tex(
    rows: Sequence[PolicyRankingSummary],
    decomposition: PolicyDecompositionSummary,
    stem: str = "policy_ranking_robustness_table",
) -> Path:
    """Write a nearby PVEU target-pair robustness table."""

    out = ensure_output_dir() / f"{stem}.tex"
    lines = _policy_table_tex_lines(
        rows,
        decomposition,
        caption="PVEU-margin robustness for a nearby target pair",
        label="tab:policy-ranking-robustness",
        policy_a_desc="the same cross-sectional transfer targeted to ages $25$ to $34$ in the central current earnings row (third row, earnings level $0.965$) on the maintained financing margin",
        policy_b_desc="the same cross-sectional transfer targeted to ages $45$ to $54$ on that maintained financing margin",
        extra_note="This nearby PVEU-margin target-pair exercise uses the same financing rule and represented local margin as the main-text comparison. In the current calibration it yields a convention-dependent ordering: age-based matching on full support slightly favors target $A$, whereas the verified current-cell map on common support slightly favors target $B$.",
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_policy_ranking_summary(
    rows: Sequence[PolicyRankingSummary],
    decomposition: PolicyDecompositionSummary,
    stem: str = "policy_ranking_summary",
) -> Path:
    """Write machine-readable policy-ranking diagnostics and decomposition."""

    out = ensure_output_dir() / f"{stem}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "label",
            "support_label",
            "support_share",
            "target_a_support_share",
            "target_b_support_share",
            "kappa_mean_plp",
            "kappa_max_plp",
            "target_a_score",
            "target_b_score",
            "log_ratio",
            "lower_log_ratio",
            "upper_log_ratio",
        ])
        for row in rows:
            writer.writerow([
                row.label,
                row.support_label,
                row.support_share,
                row.target_a_support_share,
                row.target_b_support_share,
                _to_percent_log_points(row.kappa_mean),
                _to_percent_log_points(row.kappa_max),
                row.target_a_score,
                row.target_b_score,
                row.log_ratio,
                row.lower_log_ratio,
                row.upper_log_ratio,
            ])
        writer.writerow([])
        writer.writerow(["decomposition_component", "value"])
        writer.writerow(["support_exclusion", decomposition.support_exclusion])
        writer.writerow(["within_support_rematching", decomposition.within_support_rematching])
        writer.writerow(["total_change", decomposition.total_change])
    return out


def write_policy_map_comparison_table_tex(
    rows: Sequence[PolicyRankingSummary],
    frontier_rows: Sequence[ApproximateSufficiencySummary],
    stem: str = "policy_map_comparison_table",
) -> Path:
    """Write a compact comparison of alternative current maps on one policy object."""

    overlap_by_label = {row.label: row.overall_support_overlap_share for row in frontier_rows}
    out = ensure_output_dir() / f"{stem}.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Same policy object under alternative current matching systems}",
        r"\label{tab:policy-map-comparison}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.28\textwidth}cccccc}",
        r"\toprule",
        r"Comparison system & One-step overlap & Target-union share & Mean $\hat\kappa^{\mathrm{mean}}$ & Worst-case $\hat\kappa^{\max}$ & $\Delta_{AB}$ & Diagnostic interval \\",
        r"\midrule",
    ]
    for row in rows:
        overlap = overlap_by_label.get(row.label, float("nan"))
        lines.append(
            f"{row.label} & {overlap:.4f} & {row.support_share:.4f} & "
            f"{_to_percent_log_points(row.kappa_mean):.2f} & "
            f"{_to_percent_log_points(row.kappa_max):.2f} & "
            f"{row.log_ratio:.5f} & [{row.lower_log_ratio:.5f},\\ {row.upper_log_ratio:.5f}] " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize Notes. Every row evaluates the same appendix warm-up local PVEU-margin comparison (policy $A$: ages $20$ to $29$ in the central current earnings row (third row, earnings level $0.965$); policy $B$: ages $40$ to $49$ on the same financing margin), but under the stated current matching system and on that system's own one-step support. The overlap column reports the younger-mass one-step support share for the map; the target-union share reports how much of the union of the two policy target groups remains on that map's support. The contamination columns are in percent log points, whereas $\Delta_{AB}$ and the diagnostic interval are in log points. In the current benchmark object, the standardized heuristic and age-based rows happen to coincide in the policy columns because they retain the same target-union support and induce the same maintained selector summaries there; the difference between them shows up only in the theorem-relevant support and contamination diagnostics. The table is therefore a direct bite check for the verified current-cell map discipline: benign coarsenings can recover support with only modest widening, while aggressively coarse maps can look attractive on support yet remain much looser in contamination and graph-robustness.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_kq_sensitivity_summary(


    kappa_rows: Sequence[KappaSensitivitySummary],
    envelope_rows: Sequence[EnvelopeWidthSensitivitySummary],
    stem: str = "frontier_graph_kq_sensitivity",
) -> Path:
    """Write machine-readable N_q/K sensitivity diagnostics."""

    out = ensure_output_dir() / f"{stem}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "label", "parameter", "value_1", "value_2"])
        for row in kappa_rows:
            writer.writerow(["kappa_plp", row.label, row.bin_count, _to_percent_log_points(row.weighted_mean_kappa), _to_percent_log_points(row.max_kappa)])
        for row in envelope_rows:
            writer.writerow(["envelope", row.label, row.candidate_count, row.mean_width, row.max_width])
    return out


def write_kq_sensitivity_table_tex(
    kappa_rows: Sequence[KappaSensitivitySummary],
    envelope_rows: Sequence[EnvelopeWidthSensitivitySummary],
    stem: str = "frontier_graph_kq_sensitivity_table",
) -> Path:
    """Write the supplemental N_q/K sensitivity table."""

    out = ensure_output_dir() / f"{stem}.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity of contamination and graph-envelope diagnostics to $N_q$ and $K$}",
        r"\label{tab:frontier-graph-kq-sensitivity}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{0.97\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.30\textwidth}cccc}",
        r"\toprule",
        r"Object & Tuning parameter & Mean diagnostic & Worst-case / max diagnostic & Notes \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Panel A. Contamination diagnostics across weighted-$\log\contv$ bin counts $N_q$}} \\",
    ]
    for row in kappa_rows:
        lines.append(
            f"{row.label} & $N_q={row.bin_count}$ & {_to_percent_log_points(row.weighted_mean_kappa):.2f} & {_to_percent_log_points(row.max_kappa):.2f} & " + r"$\hat\kappa^{\mathrm{mean}},\ \hat\kappa^{\max}$ \\"
        )
    lines.extend([r"\midrule", r"\multicolumn{5}{l}{\textit{Panel B. Graph-envelope width across selector counts $K$}} \\"])
    for row in envelope_rows:
        lines.append(
            f"{row.label} & $K={row.candidate_count}$ & {row.mean_width:.4f} & {row.max_width:.4f} & " + r"Mean and max age-profile band width \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.97\textwidth}",
            r"\footnotesize Notes. Panel A varies the number of weighted-$\log\contv$ bins used to aggregate omitted-state contamination within coarsened current classes in the five-state hump environment at $\rho^{S}=r$. The weighted-quantile cutoffs are recomputed separately for each $N_q$ on a discrete grid, so the $N_q=20$ rows are not nested refinements of the $N_q=10$ rows; nonmonotonic changes across $N_q$ therefore reflect changes in the operative partition rather than violations of the theorem. Panel A diagnostics are reported in percent log points. Panel B varies the number $K$ of admissible older candidates used in the graph-robust maintained envelope. Corollary~\ref{cor:graph-envelope-monotone} implies that envelope width is weakly increasing in $K$ whenever the candidate correspondences are nested, so the $K=1$ row is the degenerate point-selector benchmark and larger $K$ values widen the maintained identified set monotonically.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out




def write_alternative_normalization_summary(
    rows: Sequence[AlternativeNormalizationSummary],
    stem: str = "alternative_normalization_summary",
) -> Path:
    """Write machine-readable diagnostics for a recomputed alternative-unit graph."""

    out = ensure_output_dir() / f"{stem}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "label",
            "support_label",
            "support_share",
            "target_a_support_share",
            "target_b_support_share",
            "alternative_support_label",
            "alternative_support_share",
            "alternative_target_a_support_share",
            "alternative_target_b_support_share",
            "baseline_edge_count",
            "baseline_component_count",
            "alternative_edge_count",
            "alternative_component_count",
            "baseline_log_ratio",
            "baseline_lower_log_ratio",
            "baseline_upper_log_ratio",
            "alternative_log_ratio",
            "alternative_lower_log_ratio",
            "alternative_upper_log_ratio",
        ])
        for row in rows:
            writer.writerow([
                row.label,
                row.support_label,
                row.support_share,
                row.target_a_support_share,
                row.target_b_support_share,
                row.alternative_support_label,
                row.alternative_support_share,
                row.alternative_target_a_support_share,
                row.alternative_target_b_support_share,
                row.baseline_edge_count,
                row.baseline_component_count,
                row.alternative_edge_count,
                row.alternative_component_count,
                row.baseline_log_ratio,
                row.baseline_lower_log_ratio,
                row.baseline_upper_log_ratio,
                row.alternative_log_ratio,
                row.alternative_lower_log_ratio,
                row.alternative_upper_log_ratio,
            ])
    return out



def write_alternative_normalization_table_tex(
    rows: Sequence[AlternativeNormalizationSummary],
    stem: str = "alternative_normalization_table",
) -> Path:
    """Write a compact table for the recomputed alternative-unit graph."""

    out = ensure_output_dir() / f"{stem}.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Target comparisons after recomputing an alternative private-unit graph}",
        r"\label{tab:alternative-normalization}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.25\textwidth}cccccc}",
        r"\toprule",
        r"Target pair & Baseline union & Alt. union & Baseline $E/C$ & Alt. $E/C$ & Baseline $\Delta_{AB}$ & Alt. $\Delta_{AB}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.label} & {row.support_share:.4f} & {row.alternative_support_share:.4f} & "
            f"{row.baseline_edge_count}/{row.baseline_component_count} & "
            f"{row.alternative_edge_count}/{row.alternative_component_count} & "
            f"{row.baseline_log_ratio:.5f} [{row.baseline_lower_log_ratio:.5f},\\ {row.baseline_upper_log_ratio:.5f}] & "
            f"{row.alternative_log_ratio:.5f} [{row.alternative_lower_log_ratio:.5f},\\ {row.alternative_upper_log_ratio:.5f}] " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize Notes. Baseline columns use the maintained current-goods private unit and the stated baseline support. Alternative columns rescale the private metric to $\widetilde\upsilon_{aik}=\upsilon_{aik}/\bar\upsilon_a(k)$, where $\bar\upsilon_a(k)$ is the positive-mass geometric mean within the age--earnings cell, and then recompute one-step support, $K=5$ nearest-candidate edges, graph envelopes, and component anchors. $E/C$ reports finite edge and connected-component counts for the maintained graph used by the target comparison.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out



def plot_graph_envelope_by_age(rows: Sequence[GraphEnvelopeSummary], stem: str = "five_state_hump_graph_envelope_by_age") -> tuple[Path, Path]:
    """Plot point and graph-robust age envelopes for selected comparison systems."""

    fig, axes = plt.subplots(1, len(rows), figsize=(8.4, 3.4), constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, row in zip(axes, rows):
        ages = np.asarray(row.ages, dtype=int)
        ax.fill_between(ages, row.lower_age_average, row.upper_age_average, alpha=0.25)
        ax.plot(ages, row.point_age_average, linewidth=1.5)
        ax.set_title(row.label)
        ax.set_xlabel("Age")
        ax.set_ylabel(r"Age-average $\eta$")
    return _save_figure(fig, stem)
