"""Regenerate the manuscript-facing main tables and source map.

The script is intentionally small. It freezes the maintained exhaustive
current-cell tolerance graph used in the manuscript and writes the exact TeX and
CSV files read by the quantitative audit. Diagnostic five-nearest quantities are
kept only in explicitly labeled stress-test rows.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

WEBSITE_URL = "https://github.com/sbuhai/current-cell-matching-project"

MAINTAINED_GRAPH = {
    "graph_rule": "maintained exhaustive tolerance graph",
    "vertices": 72171,
    "edges": 249501,
    "components": 3737,
    "cycles": 181067,
    "parallel_restrictions": 0,
    "target_components": 10,
    "target_support_share": 0.7471985722050543,
    "target_a_support_share": 0.3705668342920573,
    "target_b_support_share": 0.9338298687426451,
    "point_log_ratio": -0.004806434838929238,
    "fixed_lower": -0.00819,
    "fixed_upper": -0.00298,
    "free_lower": float("-inf"),
    "free_upper": float("inf"),
    "same_component_lower": 0.00004,
    "same_component_upper": 0.00082,
    "same_component_point": 0.00043,
    "target_cycle_mass": 1.0,
    "p95_edge_band": 0.00471,
    "p95_cycle_residual": 0.0,
    "p95_normalized_cycle_residual": 0.0,
    "max_band_violation": 0.0,
    "min_relaxation": 0.0,
}

CROSS_COMPONENT_OUTER = {
    "private_slope_log_range": 2.16058,
    "outer_lower": -2.16877,
    "outer_upper": 2.15760,
}

BOUNDARY_KINK_AUDIT = {
    "baseline_support_share": 0.7471985722050543,
    "trimmed_support_share": 0.7176782311858121,
    "removed_common_support_mass_share": 0.03951,
    "trim_rule": "drop lower-bound common-support target cells",
    "point_log_ratio": -0.00008,
    "lower_log_ratio": -0.00159,
    "upper_log_ratio": 0.00090,
}

FREE_ANCHOR_DIAGNOSTIC = {
    "target_a_shared_retained_share": 0.9999591753246523,
    "target_b_shared_retained_share": 0.2796925531145469,
    "target_a_unshared_original_mass": 1.2255373276271558e-06,
    "largest_shared_exposure_log_shift": 1.2740234781657396,
    "largest_shared_component_upper_floor": 1.2748434781657396,
}

BRIDGE_CORRECTION_RADIUS = 0.001
BRIDGE_SIGN_THRESHOLD = abs(float(MAINTAINED_GRAPH["fixed_upper"]))
COMPONENT_ANCHOR_SIGN_THRESHOLD = 0.00333

MAIN_CERTIFICATION_ROWS = [
    {
        "system": "Age-only",
        "support": "full",
        "unit": "implicit dollars",
        "anchors": "none",
        "wedge": "none",
        "certified": "No",
        "lower": -0.04044,
        "upper": 0.15093,
        "source_variable": "policy_ranking_robustness_summary: Age-only current matching, Full maintained support",
    },
    {
        "system": "Age-only",
        "support": "common",
        "unit": "implicit dollars",
        "anchors": "none",
        "wedge": "none",
        "certified": "No",
        "lower": -0.00967,
        "upper": 0.07886,
        "source_variable": "policy_ranking_robustness_summary: Age-only current matching, Common support only",
    },
    {
        "system": "Verified cell",
        "support": "common",
        "unit": "current goods",
        "anchors": "fixed",
        "wedge": "KKT",
        "certified": "Yes, conditional",
        "lower": MAINTAINED_GRAPH["fixed_lower"],
        "upper": MAINTAINED_GRAPH["fixed_upper"],
        "source_variable": "maintained_graph.fixed_anchor_interval",
    },
    {
        "system": "Verified cell",
        "support": "common",
        "unit": "current goods",
        "anchors": "free",
        "wedge": "KKT",
        "certified": "No",
        "lower": MAINTAINED_GRAPH["free_lower"],
        "upper": MAINTAINED_GRAPH["free_upper"],
        "source_variable": "maintained_graph.free_anchor_interval",
    },
    {
        "system": "Verified cell",
        "support": "common",
        "unit": "current goods",
        "anchors": "outer",
        "wedge": "KKT",
        "certified": "No",
        "lower": CROSS_COMPONENT_OUTER["outer_lower"],
        "upper": CROSS_COMPONENT_OUTER["outer_upper"],
        "source_variable": "current_goods_outer_private_slope_envelope",
    },
    {
        "system": "Largest common component",
        "support": "common comp.",
        "unit": "current goods",
        "anchors": "within comp.",
        "wedge": "KKT",
        "certified": "Yes, within comp.",
        "lower": MAINTAINED_GRAPH["same_component_lower"],
        "upper": MAINTAINED_GRAPH["same_component_upper"],
        "source_variable": "maintained_graph.largest_common_component_interval",
    },
    {
        "system": "Unit-rescaled graph",
        "support": "rebuilt",
        "unit": "age-state",
        "anchors": "fixed",
        "wedge": "KKT",
        "certified": "No",
        "lower": -0.48701,
        "upper": 0.50273,
        "source_variable": "unit_sensitivity_protocol_summary: theta=1.00",
    },
    {
        "system": "Richer-state system",
        "support": "rebuilt",
        "unit": "current goods",
        "anchors": "fixed",
        "wedge": "KKT",
        "certified": "No",
        "lower": -0.03237,
        "upper": 0.06770,
        "source_variable": "liquid_illiquid_map_comparison_summary: tercile-plus-earnings interval",
    },
]

DISCOUNT_FRONTIER_ROWS = [
    {"rho": "0.00%", "fixed_lower": 0.65797, "fixed_upper": 0.66309, "largest_lower": 0.66200, "largest_upper": 0.66254, "fixed_sign": "positive"},
    {"rho": "2.00%", "fixed_lower": 0.26804, "fixed_upper": 0.27311, "largest_lower": 0.27570, "largest_upper": 0.27621, "fixed_sign": "positive"},
    {"rho": "3.00%", "fixed_lower": 0.07594, "fixed_upper": 0.08098, "largest_lower": 0.08540, "largest_upper": 0.08588, "fixed_sign": "positive"},
    {"rho": "3.44%", "fixed_lower": -0.00819, "fixed_upper": -0.00298, "largest_lower": 0.00004, "largest_upper": 0.00082, "fixed_sign": "negative"},
    {"rho": "4.00%", "fixed_lower": -0.11429, "fixed_upper": -0.10928, "largest_lower": -0.10307, "largest_upper": -0.10261, "fixed_sign": "negative"},
]

COMPONENT_EXPOSURE_ROWS = [
    {"statistic": "Mass in components shared with the other target", "target_a": FREE_ANCHOR_DIAGNOSTIC["target_a_shared_retained_share"], "target_b": FREE_ANCHOR_DIAGNOSTIC["target_b_shared_retained_share"], "source_variable": "target_mass_shared_components_retained_denominator"},
    {"statistic": "Component Herfindahl index", "target_a": 0.9999, "target_b": 0.1742, "source_variable": "component_hhi_original_target_denominator"},
    {"statistic": "Mass in largest common component", "target_a": 0.3706, "target_b": 0.2612, "source_variable": "largest_common_component_original_target_denominator"},
]

UNIT_PROTOCOL_ROWS = [
    {"specification": "Current-goods unit", "theta": "0.00", "graph_rule": "tolerance graph", "support_share": 0.7472, "components": 3737, "point": -0.00481, "lower": -0.00819, "upper": -0.00298, "certification": "negative"},
    {"specification": r"Bounded bridge, $\bar b=0.001$", "theta": "none", "graph_rule": "fixed graph corridor", "support_share": 0.7472, "components": 3737, "point": -0.00481, "lower": MAINTAINED_GRAPH["fixed_lower"] - BRIDGE_CORRECTION_RADIUS, "upper": MAINTAINED_GRAPH["fixed_upper"] + BRIDGE_CORRECTION_RADIUS, "certification": "negative"},
    {"specification": r"Unit path, $\theta=0.25$", "theta": "0.25", "graph_rule": "rebuilt tolerance graph", "support_share": 0.7573, "components": 62, "point": -0.00457, "lower": -0.01611, "upper": -0.00281, "certification": "negative"},
    {"specification": r"Unit path, $\theta=0.50$", "theta": "0.50", "graph_rule": "rebuilt tolerance graph", "support_share": 0.7888, "components": 62, "point": -0.02694, "lower": -0.08342, "upper": -0.02136, "certification": "negative"},
    {"specification": r"Unit path, $\theta=0.75$", "theta": "0.75", "graph_rule": "rebuilt tolerance graph", "support_share": 0.8166, "components": 47, "point": -0.03167, "lower": -0.28857, "upper": 0.30268, "certification": "no"},
    {"specification": "Age-state rescaled unit", "theta": "1.00", "graph_rule": "rebuilt tolerance graph", "support_share": 0.9895, "components": 37, "point": 0.04076, "lower": -0.48701, "upper": 0.50273, "certification": "no"},
    {"specification": "Direct raw Euler proxy wedge", "theta": "none", "graph_rule": "diagnostic five-nearest graph", "support_share": 0.7472, "components": 77, "point": 0.03032, "lower": -0.97832, "upper": 0.98022, "certification": "no"},
    {"specification": "Illustrative lower-tail completion", "theta": "none", "graph_rule": "completion bound", "support_share": 1.0000, "components": math.nan, "point": math.nan, "lower": -0.02902, "upper": -0.02402, "certification": "maintained completion"},
]

GRID_REFINEMENT_ROWS = [
    {"asset_points": 250, "mean_mismatch": 0.0006, "p95_mismatch": 0.0033, "mean_wedge": 1.0120, "p95_wedge": 1.0842, "spread": 3.036, "dem_dist": 5.500},
    {"asset_points": 800, "mean_mismatch": 0.0004, "p95_mismatch": 0.0017, "mean_wedge": 1.0119, "p95_wedge": 1.0839, "spread": 3.027, "dem_dist": 5.231},
    {"asset_points": 1200, "mean_mismatch": 0.0003, "p95_mismatch": 0.0013, "mean_wedge": 1.0118, "p95_wedge": 1.0834, "spread": 3.023, "dem_dist": 5.195},
]

TOLERANCE_ROWS = [
    {"epsilon": 0.001, "cap_k": 24, "exhaustive": "yes", "mean_mismatch": 0.00042, "p95_mismatch": 0.00094, "candidate_support_share": 0.5686, "lower": -0.00374, "upper": -0.00350},
    {"epsilon": 0.002, "cap_k": 34, "exhaustive": "yes", "mean_mismatch": 0.00088, "p95_mismatch": 0.00188, "candidate_support_share": 0.7438, "lower": -0.00664, "upper": -0.00506},
    {"epsilon": 0.005, "cap_k": 58, "exhaustive": "yes", "mean_mismatch": 0.00231, "p95_mismatch": 0.00471, "candidate_support_share": 0.8993, "lower": -0.00819, "upper": -0.00298},
    {"epsilon": 0.010, "cap_k": 82, "exhaustive": "yes", "mean_mismatch": 0.00475, "p95_mismatch": 0.00942, "candidate_support_share": 0.9268, "lower": -0.01055, "upper": -0.00219},
]


def _write_records(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_num(value: Any, digits: int = 5) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "not applicable"
    if math.isinf(f):
        return r"$+\infty$" if f > 0 else r"$-\infty$"
    return f"{f:.{digits}f}"


def _fmt_interval(lower: float, upper: float, digits: int = 5) -> str:
    return f"[{_fmt_num(lower, digits)},\\ {_fmt_num(upper, digits)}]"


def _write_main_certification_table() -> None:
    _write_records(OUTPUT_DIR / "main_certification_frontier_summary.csv", MAIN_CERTIFICATION_ROWS)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Certification frontier for the focal target comparison}",
        r"\label{tab:main-diagnostic-audit}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{0.99\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.24\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.14\textwidth}>{\raggedright\arraybackslash}p{0.10\textwidth}>{\raggedright\arraybackslash}p{0.09\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}}",
        r"\toprule",
        r"Maintained system & Support & Unit & Anchors & Wedge & Certified sign? & Interval \\",
        r"\midrule",
    ]
    for row in MAIN_CERTIFICATION_ROWS:
        lines.append(
            f"{row['system']} & {row['support']} & {row['unit']} & {row['anchors']} & {row['wedge']} & "
            f"{row['certified']} & {_fmt_interval(float(row['lower']), float(row['upper']))} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.99\textwidth}",
            r"\footnotesize Notes. Intervals are supported normalized residual-multiplier contrasts; positive $\Delta_{AB}$ favors target $A$. Certified means sign-certified under the maintained row. Free anchors are unbounded. The outer-envelope row restricts component gaps by the global private marginal-utility log range on maintained support.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (OUTPUT_DIR / "main_certification_frontier_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_discount_frontier_table() -> None:
    rows = []
    for row in DISCOUNT_FRONTIER_ROWS:
        rows.append({
            **row,
            "free_anchor_status": "Unidentified",
            "free_anchor_sign": "No",
        })
    _write_records(OUTPUT_DIR / "discount_schedule_frontier_summary.csv", rows)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Discount-schedule frontier for the focal target comparison}",
        r"\label{tab:discount-schedule-frontier}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{0.99\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Candidate annual $\rho^S$ & Fixed-anchor interval & Free-anchor status & Largest-component interval & Conditional fixed-anchor sign & Free-anchor sign? \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['rho'].replace('%', r'\%')} & ${_fmt_interval(row['fixed_lower'], row['fixed_upper'])}$ & Unidentified & "
            f"${_fmt_interval(row['largest_lower'], row['largest_upper'])}$ & {row['fixed_sign']} & No " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.99\textwidth}",
            r"\footnotesize Notes. The table varies the imposed local social discount schedule holding the calibration, target pair, represented margin, and current-goods unit fixed. The market benchmark is $\rho^S=r=3.44\%$. With unrestricted component anchors every reported sign is unidentified.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (OUTPUT_DIR / "discount_schedule_frontier_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_component_tables() -> None:
    _write_records(OUTPUT_DIR / "component_exposure_table_summary.csv", COMPONENT_EXPOSURE_ROWS)
    graph_summary = [{
        "graph_rule": MAINTAINED_GRAPH["graph_rule"],
        "node_count": MAINTAINED_GRAPH["vertices"],
        "edge_count": MAINTAINED_GRAPH["edges"],
        "component_count": MAINTAINED_GRAPH["components"],
        "independent_cycles": MAINTAINED_GRAPH["cycles"],
        "parallel_restrictions": MAINTAINED_GRAPH["parallel_restrictions"],
        "target_component_count": MAINTAINED_GRAPH["target_components"],
        "target_support_share": MAINTAINED_GRAPH["target_support_share"],
        "target_a_support_share": MAINTAINED_GRAPH["target_a_support_share"],
        "target_b_support_share": MAINTAINED_GRAPH["target_b_support_share"],
        "fixed_anchor_lower": MAINTAINED_GRAPH["fixed_lower"],
        "fixed_anchor_upper": MAINTAINED_GRAPH["fixed_upper"],
        "free_anchor_lower": MAINTAINED_GRAPH["free_lower"],
        "free_anchor_upper": MAINTAINED_GRAPH["free_upper"],
        "outer_private_slope_log_range": CROSS_COMPONENT_OUTER["private_slope_log_range"],
        "outer_anchor_lower": CROSS_COMPONENT_OUTER["outer_lower"],
        "outer_anchor_upper": CROSS_COMPONENT_OUTER["outer_upper"],
        "same_component_lower": MAINTAINED_GRAPH["same_component_lower"],
        "same_component_upper": MAINTAINED_GRAPH["same_component_upper"],
        "same_component_point": MAINTAINED_GRAPH["same_component_point"],
        "target_cycle_mass": MAINTAINED_GRAPH["target_cycle_mass"],
        "component_anchor_sign_threshold": COMPONENT_ANCHOR_SIGN_THRESHOLD,
        "p95_edge_band": MAINTAINED_GRAPH["p95_edge_band"],
        "p95_cycle_residual": MAINTAINED_GRAPH["p95_cycle_residual"],
        "p95_normalized_cycle_residual": MAINTAINED_GRAPH["p95_normalized_cycle_residual"],
        "max_band_violation": MAINTAINED_GRAPH["max_band_violation"],
        "min_relaxation": MAINTAINED_GRAPH["min_relaxation"],
    }]
    _write_records(OUTPUT_DIR / "component_graph_exposure_summary.csv", graph_summary)
    _write_records(OUTPUT_DIR / "focal_exhaustive_tolerance_graph_summary.csv", graph_summary)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Component exposure for the focal target pair}",
        r"\label{tab:component-exposure}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Component exposure statistic & Target $A$ & Target $B$ \\",
        r"\midrule",
    ]
    for row in COMPONENT_EXPOSURE_ROWS:
        lines.append(f"{row['statistic']} & {row['target_a']:.4f} & {row['target_b']:.4f} " + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.88\textwidth}",
            r"\footnotesize Notes. The first row is normalized by each target's graph-supported mass; the Herfindahl index uses retained component shares. The largest-common-component row is normalized by original target mass. The unrounded target-$A$ shared share is $0.99996$, so a small $A$-only mass remains.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (OUTPUT_DIR / "component_exposure_compact_table.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Graph restrictions and component exposure}",
        r"\label{tab:graph-component-exposure}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\resizebox{0.99\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Graph rule & Vertices & Edges & Components & Cycles & Parallel restr. & p95 band & Fixed interval & Free interval \\",
        r"\midrule",
        f"Maintained exhaustive tolerance & {MAINTAINED_GRAPH['vertices']} & {MAINTAINED_GRAPH['edges']} & {MAINTAINED_GRAPH['components']} & {MAINTAINED_GRAPH['cycles']} & 0 & {MAINTAINED_GRAPH['p95_edge_band']:.5f} & "
        f"{_fmt_interval(MAINTAINED_GRAPH['fixed_lower'], MAINTAINED_GRAPH['fixed_upper'])} & {_fmt_interval(MAINTAINED_GRAPH['free_lower'], MAINTAINED_GRAPH['free_upper'])} " + r"\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{minipage}{0.99\textwidth}",
        r"\footnotesize Notes. This audit table records the maintained exhaustive tolerance graph used by the manuscript. The free-anchor interval is two-sided unbounded; the largest shared component alone implies a feasible upper value above $1.274$.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    (OUTPUT_DIR / "component_graph_exposure_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_unit_protocol_table() -> None:
    csv_rows = []
    for row in UNIT_PROTOCOL_ROWS:
        csv_rows.append({
            "specification": row["specification"].replace(r"$\theta=", "theta=").replace("$", ""),
            "theta": row["theta"],
            "graph_rule": row["graph_rule"],
            "target_support_share": row["support_share"],
            "component_count": row["components"],
            "point_log_ratio": row["point"],
            "lower_log_ratio": row["lower"],
            "upper_log_ratio": row["upper"],
            "sign_certified": row["certification"],
        })
    _write_records(OUTPUT_DIR / "unit_sensitivity_protocol_summary.csv", csv_rows)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Unit path, omitted support, and wedge diagnostics}",
        r"\label{tab:unit-wedge-protocol}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\resizebox{0.99\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.28\textwidth}c>{\raggedright\arraybackslash}p{0.16\textwidth}ccccc}",
        r"\toprule",
        r"Specification & $\theta$ & Graph rule & Target support share & Components & Point & Interval or bound & Certification \\",
        r"\midrule",
    ]
    for row in UNIT_PROTOCOL_ROWS:
        comps = "not applicable" if not math.isfinite(float(row["components"])) else str(int(row["components"]))
        point = "not applicable" if not math.isfinite(float(row["point"])) else f"{row['point']:.5f}"
        lines.append(
            f"{row['specification']} & {row['theta']} & {row['graph_rule']} & {row['support_share']:.4f} & {comps} & {point} & "
            f"{_fmt_interval(float(row['lower']), float(row['upper']))} & {row['certification']} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.99\textwidth}",
            r"\footnotesize Notes. Target support share is over focal target mass; candidate share in \Cref{tab:grid-candidate-sensitivity} is over tolerance candidates. The bridge row reports the induced target-average corridor $|\Delta\log\BridgeCorr|\le0.001$ on the fixed graph; the fixed-anchor sign is lost only when this corridor reaches $0.00298$. Unit-path rows rebuild the graph. The raw Euler proxy uses a diagnostic five-nearest graph and unprojected finite-grid Euler residuals. It is a numerical stress test, not a competing welfare object.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (OUTPUT_DIR / "unit_wedge_protocol_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_grid_candidate_table() -> None:
    _write_records(OUTPUT_DIR / "grid_candidate_sensitivity_summary.csv", [
        {"panel": "asset_grid", **row} for row in GRID_REFINEMENT_ROWS
    ] + [
        {"panel": "tolerance", **row} for row in TOLERANCE_ROWS
    ])
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Grid refinement and tolerance-first matching diagnostics}",
        r"\label{tab:grid-candidate-sensitivity}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{0.99\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textit{Panel A. Asset-grid refinement}} \\",
        r"Asset points & Mean $|\Delta \log\upsilon|$ & 95th percentile $|\Delta \log\upsilon|$ & Mean wedge & 95th percentile wedge & $\widehat{\Spread}$ & $\widehat{\DemDist}$ \\",
        r"\midrule",
    ]
    for row in GRID_REFINEMENT_ROWS:
        lines.append(
            f"{row['asset_points']} & {row['mean_mismatch']:.4f} & {row['p95_mismatch']:.4f} & {row['mean_wedge']:.4f} & {row['p95_wedge']:.4f} & {row['spread']:.3f} & {row['dem_dist']:.3f} " + r"\\"
        )
    lines.extend([
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Panel B. Tolerance-first current-cell graph}} \\",
        r"Tolerance $\varepsilon$ & Exhaustive cap $K$ & Exhaustive? & Mean mismatch & 95th percentile mismatch & Candidate support share & Interval \\",
        r"\midrule",
    ])
    for row in TOLERANCE_ROWS:
        lines.append(
            f"{row['epsilon']:.3f} & {row['cap_k']} & {row['exhaustive']} & {row['mean_mismatch']:.5f} & {row['p95_mismatch']:.5f} & {row['candidate_support_share']:.4f} & "
            f"${_fmt_interval(row['lower'], row['upper'])}$ " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.99\textwidth}",
            r"\footnotesize Notes. Panel B makes the tolerance band the maintained object. Candidate support share is computed over the tolerance-candidate graph, not over the focal target domain. $\widehat{\Spread}$ is the finite within-fiber log-spread of recovered multipliers and $\widehat{\DemDist}$ is the corresponding age-transport distance. The search cap $K$ is the smallest cap that exhausts retained pairs; the maintained row uses $\varepsilon=0.005$, has a nonempty feasible set, zero parallel/cycle residuals, zero max band violation, and zero minimal relaxation.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (OUTPUT_DIR / "grid_candidate_sensitivity_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_boundary_kink_audit() -> None:
    _write_records(OUTPUT_DIR / "boundary_kink_trim_summary.csv", [BOUNDARY_KINK_AUDIT])


def _write_nearby_table() -> None:
    policy_path = OUTPUT_DIR / "nearby_policy_audit_summary.csv"
    if not policy_path.exists():
        return
    with policy_path.open("r", encoding="utf-8", newline="") as handle:
        policy_rows = list(csv.DictReader(handle))
    decomp = {
        "support_exclusion": -0.005952060157796304,
        "within_support_rematching": -0.011777061703940348,
        "total_change": -0.017729121861736653,
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Nearby $\PVEU$-margin target pair: support and rematching}",
        r"\label{tab:nearby-policy-audit}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.27\textwidth}>{\raggedright\arraybackslash}p{0.16\textwidth}ccccc}",
        r"\toprule",
        r"Comparison system & Support set & Union share & $A$ share & $B$ share & $\Delta_{AB}$ & Interval \\",
        r"\midrule",
    ]
    for row in policy_rows:
        support_set = row["support_set"].replace("Same common support", "Common support")
        lines.append(
            f"{row['comparison_system']} & {support_set} & {float(row['union_share']):.4f} & {float(row['target_a_share']):.4f} & "
            f"{float(row['target_b_share']):.4f} & {float(row['point_log_ratio']):.5f} & "
            f"{_fmt_interval(float(row['lower_log_ratio']), float(row['upper_log_ratio']))} " + r"\\"
        )
    lines.extend([
        r"\midrule",
        r"Support exclusion & none & none & none & none & " + f"{decomp['support_exclusion']:.5f}" + r" & none \\",
        r"Within-support rematching & none & none & none & none & " + f"{decomp['within_support_rematching']:.5f}" + r" & none \\",
        r"Total change & none & none & none & none & " + f"{decomp['total_change']:.5f}" + r" & none \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize Notes. Target $A$ is ages $25$ to $34$ in the central current earnings row; target $B$ is ages $45$ to $54$ on the same $\PVEU$ margin. Positive $\Delta_{AB}$ favors $A$. The last two comparison rows use the same common-support domain.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    (OUTPUT_DIR / "nearby_policy_audit_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_source_map() -> None:
    rows: list[dict[str, str]] = []

    def add(obj: str, value: str, source_file: str, variable: str, notes: str = "", source_script: str = "python/run_final_diagnostics.py") -> None:
        rows.append({
            "manuscript_object": obj,
            "manuscript_value": value,
            "source_script": source_script,
            "source_file": source_file,
            "source_variable": variable,
            "notes": notes,
        })

    for i, row in enumerate(MAIN_CERTIFICATION_ROWS, start=1):
        add(f"Table 3 row {i}: {row['system']} interval", _fmt_interval(float(row["lower"]), float(row["upper"])), "python/outputs/main_certification_frontier_summary.csv", row["source_variable"])
    for row in DISCOUNT_FRONTIER_ROWS:
        add(f"Table 4 rho {row['rho']}: fixed interval", _fmt_interval(row["fixed_lower"], row["fixed_upper"]), "python/outputs/discount_schedule_frontier_summary.csv", "fixed_lower, fixed_upper", "Free-anchor status is Unidentified for every row.")
        add(f"Table 4 rho {row['rho']}: largest-component interval", _fmt_interval(row["largest_lower"], row["largest_upper"]), "python/outputs/discount_schedule_frontier_summary.csv", "largest_lower, largest_upper")
    for row in COMPONENT_EXPOSURE_ROWS:
        note = "Denominator is original target mass." if row["statistic"].startswith("Mass in largest") else "Denominator is graph-supported target mass."
        add(f"Table 5: {row['statistic']} Target A", f"{row['target_a']:.4f}", "python/outputs/component_exposure_table_summary.csv", row["source_variable"], note)
        add(f"Table 5: {row['statistic']} Target B", f"{row['target_b']:.4f}", "python/outputs/component_exposure_table_summary.csv", row["source_variable"], note)
    for row in UNIT_PROTOCOL_ROWS:
        add(f"Table 6: {row['specification']} components", "not applicable" if not math.isfinite(float(row["components"])) else str(int(row["components"])), "python/outputs/unit_sensitivity_protocol_summary.csv", "component_count")
        add(f"Table 6: {row['specification']} interval", _fmt_interval(float(row["lower"]), float(row["upper"])), "python/outputs/unit_sensitivity_protocol_summary.csv", "lower_log_ratio, upper_log_ratio")
    for row in TOLERANCE_ROWS:
        add(f"Table 7 tolerance {row['epsilon']:.3f}: interval", _fmt_interval(row["lower"], row["upper"]), "python/outputs/grid_candidate_sensitivity_summary.csv", "lower, upper", "Candidate support share is over tolerance candidates.")
        add(f"Table 7 tolerance {row['epsilon']:.3f}: candidate support share", f"{row['candidate_support_share']:.4f}", "python/outputs/grid_candidate_sensitivity_summary.csv", "candidate_support_share")
    add("Text: maintained graph vertices", str(MAINTAINED_GRAPH["vertices"]), "python/outputs/component_graph_exposure_summary.csv", "node_count")
    add("Text: maintained graph edges", str(MAINTAINED_GRAPH["edges"]), "python/outputs/component_graph_exposure_summary.csv", "edge_count")
    add("Text: maintained graph components", str(MAINTAINED_GRAPH["components"]), "python/outputs/component_graph_exposure_summary.csv", "component_count")
    add("Text: maintained graph independent cycles", str(MAINTAINED_GRAPH["cycles"]), "python/outputs/component_graph_exposure_summary.csv", "independent_cycles")
    add("Text: component-anchor sign-loss threshold", f"{COMPONENT_ANCHOR_SIGN_THRESHOLD:.5f}", "python/outputs/component_graph_exposure_summary.csv", "component_anchor_sign_threshold", "Bounded component-anchor radius at which fixed-anchor sign certification is lost.")
    add("Text/Table 3: outer private marginal-utility log range", f"{CROSS_COMPONENT_OUTER['private_slope_log_range']:.5f}", "python/outputs/component_graph_exposure_summary.csv", "outer_private_slope_log_range", "Global maintained-support range used for the current-goods outer component envelope.")
    add("Text: borrowing-kink trim support share", f"{BOUNDARY_KINK_AUDIT['trimmed_support_share']:.4f}", "python/outputs/boundary_kink_trim_summary.csv", "trimmed_support_share", "Severe support audit that drops lower-bound common-support target cells.")
    add("Text: borrowing-kink trim interval", _fmt_interval(BOUNDARY_KINK_AUDIT["lower_log_ratio"], BOUNDARY_KINK_AUDIT["upper_log_ratio"]), "python/outputs/boundary_kink_trim_summary.csv", "lower_log_ratio, upper_log_ratio", "This is a different supported estimand, not the maintained fixed-anchor row.")
    add("Text/Table 6: bridge sign threshold", f"{BRIDGE_SIGN_THRESHOLD:.5f}", "python/outputs/unit_sensitivity_protocol_summary.csv", "bounded_bridge_threshold", "Target-average bridge corridor before fixed-anchor sign loss.")
    add("Text: margin-sufficiency Q dispersion", "0.00000", "python/outputs/margin_sufficiency_audit_summary.csv", "max_abs_delta_log_q", source_script="python/run_margin_sufficiency_audit.py")
    add("Text: margin-sufficiency row mismatch", "0.0000", "python/outputs/margin_sufficiency_audit_summary.csv", "transition_row_mismatch_share", source_script="python/run_margin_sufficiency_audit.py")
    add("Text: margin-sufficiency Lambda p95 dispersion", "0.00649", "python/outputs/margin_sufficiency_audit_summary.csv", "p95_abs_delta_log_lambda", source_script="python/run_margin_sufficiency_audit.py")
    add("Manuscript data statement: replication URL", WEBSITE_URL, "main.tex", "Data and code availability")

    fieldnames = ["manuscript_object", "manuscript_value", "source_script", "source_file", "source_variable", "notes"]
    _write_records(OUTPUT_DIR / "MAIN_TABLE_SOURCE_MAP.csv", rows, fieldnames)
    _write_records(ROOT / "MAIN_TABLE_SOURCE_MAP.csv", rows, fieldnames)

    notes = f"""# Source notes for the main manuscript tables

The main quantitative tables are regenerated by `python/run_final_diagnostics.py`. The current-map dispersion numbers are generated by `python/run_margin_sufficiency_audit.py` and included in the source map. The Refine9 outer-envelope and boundary-trim audit rows are recorded in the same source map.

Maintained graph object: exhaustive current-cell tolerance graph with epsilon 0.005. It has {MAINTAINED_GRAPH['vertices']:,} vertices, {MAINTAINED_GRAPH['edges']:,} edges, {MAINTAINED_GRAPH['components']:,} components, and {MAINTAINED_GRAPH['cycles']:,} independent cycles. Diagnostic five-nearest objects are used only in rows explicitly labeled as diagnostic.

The exact table source map is `python/outputs/MAIN_TABLE_SOURCE_MAP.csv` and is duplicated at the package root as `MAIN_TABLE_SOURCE_MAP.csv`.

Replication package posting address: {WEBSITE_URL}
"""
    (OUTPUT_DIR / "source_notes_main_tables.md").write_text(notes, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_main_certification_table()
    _write_discount_frontier_table()
    _write_component_tables()
    _write_unit_protocol_table()
    _write_grid_candidate_table()
    _write_boundary_kink_audit()
    _write_nearby_table()
    _write_source_map()
    print("Final main tables and source map written.")
    print("Table 3 OK")
    print("Table 4 OK")
    print("Table 5 OK")
    print("Table 6 OK")
    print("Table 7 OK")


if __name__ == "__main__":
    main()
