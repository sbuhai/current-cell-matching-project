"""Generate focused nearby-policy audit tables for the main-text quantitative rewrite.

This helper reads the already-generated quantitative summary CSV files and repackages
those diagnostics into two compact main-text tables:

1. A support/rematching audit for the focal nearby PVEU-margin target pair.
2. A maintained-layer audit that separates omitted support, normalization,
   and wedge-enforcement choices.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _read_policy_summary(path: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    decomposition: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        section = "rows"
        headers: list[str] | None = None
        for raw_row in reader:
            if not raw_row or not any(cell.strip() for cell in raw_row):
                continue
            if section == "rows" and raw_row[0] == "decomposition_component":
                section = "decomposition"
                continue
            if section == "rows":
                if headers is None:
                    headers = raw_row
                    continue
                record = dict(zip(headers, raw_row, strict=False))
                rows.append(record)
            else:
                decomposition[raw_row[0]] = float(raw_row[1])
    return rows, decomposition


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _format_interval(lower: float, upper: float, digits: int = 5) -> str:
    return f"[{lower:.{digits}f},\\ {upper:.{digits}f}]"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    policy_rows, decomposition = _read_policy_summary(OUTPUT_DIR / "policy_ranking_robustness_summary.csv")
    omitted_rows = _read_csv_records(OUTPUT_DIR / "omitted_support_bounds_summary.csv")
    normalization_rows = _read_csv_records(OUTPUT_DIR / "alternative_normalization_summary.csv")
    wedge_rows = _read_csv_records(OUTPUT_DIR / "wedge_audit_summary.csv")

    def pick_policy(label: str, support_label: str) -> dict[str, str]:
        for row in policy_rows:
            if row["label"] == label and row["support_label"] == support_label:
                return row
        raise ValueError(f"Policy row not found: {label!r}, {support_label!r}")

    age_full = pick_policy("Age-only current matching", "Full maintained support")
    age_common = pick_policy("Age-only current matching", "Common support only")
    natural_common = pick_policy("Verified current-cell map", "Common support only")

    nearby_omitted = next(row for row in omitted_rows if row["target_pair"] == "Nearby pair")
    nearby_norm = next(row for row in normalization_rows if row["label"].startswith("Nearby pair:"))
    nearby_wedge_impl = next(
        row
        for row in wedge_rows
        if row["panel"].startswith("Nearby") and row["wedge_specification"] == "Implemented KKT-consistent wedge"
    )
    nearby_wedge_raw = next(
        row
        for row in wedge_rows
        if row["panel"].startswith("Nearby") and row["wedge_specification"] == "Fully raw Euler proxy"
    )

    audit_rows = [
        {
            "comparison_system": "Age-based current matching",
            "support_set": "Full maintained support",
            "union_share": float(age_full["support_share"]),
            "target_a_share": float(age_full["target_a_support_share"]),
            "target_b_share": float(age_full["target_b_support_share"]),
            "point_log_ratio": float(age_full["log_ratio"]),
            "lower_log_ratio": float(age_full["lower_log_ratio"]),
            "upper_log_ratio": float(age_full["upper_log_ratio"]),
        },
        {
            "comparison_system": "Age-based current matching",
            "support_set": "Common support only",
            "union_share": float(age_common["support_share"]),
            "target_a_share": float(age_common["target_a_support_share"]),
            "target_b_share": float(age_common["target_b_support_share"]),
            "point_log_ratio": float(age_common["log_ratio"]),
            "lower_log_ratio": float(age_common["lower_log_ratio"]),
            "upper_log_ratio": float(age_common["upper_log_ratio"]),
        },
        {
            "comparison_system": "Verified current-cell map",
            "support_set": "Same common support",
            "union_share": float(natural_common["support_share"]),
            "target_a_share": float(natural_common["target_a_support_share"]),
            "target_b_share": float(natural_common["target_b_support_share"]),
            "point_log_ratio": float(natural_common["log_ratio"]),
            "lower_log_ratio": float(natural_common["lower_log_ratio"]),
            "upper_log_ratio": float(natural_common["upper_log_ratio"]),
        },
    ]

    _write_csv(
        OUTPUT_DIR / "nearby_policy_audit_summary.csv",
        [
            "comparison_system",
            "support_set",
            "union_share",
            "target_a_share",
            "target_b_share",
            "point_log_ratio",
            "lower_log_ratio",
            "upper_log_ratio",
        ],
        audit_rows,
    )

    audit_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Nearby $\PVEU$-margin target pair I: support restriction and within-support rematching}",
        r"\label{tab:nearby-policy-audit}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.27\textwidth}>{\raggedright\arraybackslash}p{0.16\textwidth}ccccc}",
        r"\toprule",
        r"Comparison system & Support set & Union share & $A$ share & $B$ share & $\Delta_{AB}$ & Diagnostic interval \\",
        r"\midrule",
    ]
    for row in audit_rows:
        audit_lines.append(
            f"{row['comparison_system']} & {row['support_set']} & "
            f"{_format_float(row['union_share'], 4)} & {_format_float(row['target_a_share'], 4)} & "
            f"{_format_float(row['target_b_share'], 4)} & {_format_float(row['point_log_ratio'], 5)} & "
            f"{_format_interval(row['lower_log_ratio'], row['upper_log_ratio'])} " + r"\\"
        )
    audit_lines.extend(
        [
            r"\midrule",
            r"\multicolumn{7}{l}{\textit{Decomposition of the shift from age-based full support to verified current-cell common support}} \\",
            r"Support exclusion & none & none & none & none & "
            + _format_float(decomposition["support_exclusion"], 5)
            + r" & none \\",
            r"Within-support rematching & none & none & none & none & "
            + _format_float(decomposition["within_support_rematching"], 5)
            + r" & none \\",
            r"Total change & none & none & none & none & "
            + _format_float(decomposition["total_change"], 5)
            + r" & none \\",
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize Notes. Target $A$ is a $\PVEU$-margin target on ages $25$ to $34$ in the central current earnings row (third row, earnings level $0.965$); target $B$ is the analogous target on ages $45$ to $54$ under the same financing rule and represented one-period margin. Positive $\Delta_{AB}$ favors $A$. The last two rows use the same common-support domain; the ordering difference comes from within-support rematching rather than physical-dollar budget accounting.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (OUTPUT_DIR / "nearby_policy_audit_table.tex").write_text("\n".join(audit_lines), encoding="utf-8")

    baseline_footprint = f"{int(float(nearby_norm['baseline_edge_count']))}/{int(float(nearby_norm['baseline_component_count']))}"
    alternative_footprint = f"{int(float(nearby_norm['alternative_edge_count']))}/{int(float(nearby_norm['alternative_component_count']))}"

    layer_rows = [
        {
            "layer": r"Verified current-cell map in baseline $\PVEU$ units",
            "domain": "Common support",
            "support_share": float(nearby_wedge_impl["support_share"]),
            "graph_footprint": baseline_footprint,
            "point_log_ratio": float(nearby_wedge_impl["point_log_ratio"]),
            "lower_log_ratio": float(nearby_wedge_impl["lower_log_ratio"]),
            "upper_log_ratio": float(nearby_wedge_impl["upper_log_ratio"]),
        },
        {
            "layer": "Monotone lower-tail completion",
            "domain": "Full target",
            "support_share": 1.0,
            "graph_footprint": "none",
            "point_log_ratio": "",
            "lower_log_ratio": float(nearby_omitted["full_target_lower_log_ratio"]),
            "upper_log_ratio": float(nearby_omitted["full_target_upper_log_ratio"]),
        },
        {
            "layer": "Alternative rescaled unit",
            "domain": "Recomputed support",
            "support_share": float(nearby_norm["alternative_support_share"]),
            "graph_footprint": alternative_footprint,
            "point_log_ratio": float(nearby_norm["alternative_log_ratio"]),
            "lower_log_ratio": float(nearby_norm["alternative_lower_log_ratio"]),
            "upper_log_ratio": float(nearby_norm["alternative_upper_log_ratio"]),
        },
        {
            "layer": "Direct raw Euler proxy wedge",
            "domain": "Same common support",
            "support_share": float(nearby_wedge_raw["support_share"]),
            "graph_footprint": baseline_footprint,
            "point_log_ratio": float(nearby_wedge_raw["point_log_ratio"]),
            "lower_log_ratio": float(nearby_wedge_raw["lower_log_ratio"]),
            "upper_log_ratio": float(nearby_wedge_raw["upper_log_ratio"]),
        },
    ]

    _write_csv(
        OUTPUT_DIR / "nearby_policy_layers_summary.csv",
        ["layer", "domain", "support_share", "graph_footprint", "point_log_ratio", "lower_log_ratio", "upper_log_ratio"],
        layer_rows,
    )

    layers_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Nearby $\PVEU$-margin target pair II: omitted support, units, and wedge layers}",
        r"\label{tab:nearby-policy-layers}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{0.98\textwidth}{!}{%",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.30\textwidth}>{\raggedright\arraybackslash}p{0.15\textwidth}cccc}",
        r"\toprule",
        r"Maintained layer / specification & Domain & Union share & $E/C$ & Point summary & Interval or bound \\",
        r"\midrule",
    ]
    for row in layer_rows:
        if row["point_log_ratio"] == "":
            point_display = r"none"
        else:
            point_display = _format_float(float(row["point_log_ratio"]), 5)
        layers_lines.append(
            f"{row['layer']} & {row['domain']} & {_format_float(float(row['support_share']), 4)} & {row['graph_footprint']} & {point_display} & "
            f"{_format_interval(float(row['lower_log_ratio']), float(row['upper_log_ratio']))} " + r"\\"
        )
    layers_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize Notes. $E/C$ reports finite $K=5$ edge and connected-component counts for the graph used in the row. The common-support rows use five nearest older candidates in $\log\contv$ within each verified class. The full-target row orders current liquid assets within each age and earnings block and completes omitted mass below the first supported asset by a weakly decreasing point summary. The alternative-unit row rescales the private metric to $\widetilde\upsilon=\upsilon/\bar\upsilon_a(k)$ and recomputes support, edges, graph envelopes, and component anchors. The raw-Euler row replaces the KKT-consistent wedge by the directly propagated Euler proxy.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (OUTPUT_DIR / "nearby_policy_layers_table.tex").write_text("\n".join(layers_lines), encoding="utf-8")

    print("Nearby policy audit tables written successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
