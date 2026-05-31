"""Regenerate manuscript Tables 3-7 and verify key synchronization values."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

EXPECTED = {
    "component_count": "3737",
    "edge_count": "249501",
    "node_count": "72171",
    "independent_cycles": "181067",
}


def read_one(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def main() -> int:
    subprocess.run([PYTHON, str(ROOT / "run_margin_sufficiency_audit.py")], check=True)
    subprocess.run([PYTHON, str(ROOT / "run_final_diagnostics.py")], check=True)
    graph = read_one(ROOT / "outputs" / "component_graph_exposure_summary.csv")
    for key, expected in EXPECTED.items():
        got = graph.get(key)
        if got != expected:
            raise SystemExit(f"{key} mismatch: expected {expected}, got {got}")
    unit_rows = list(csv.DictReader((ROOT / "outputs" / "unit_sensitivity_protocol_summary.csv").open("r", encoding="utf-8", newline="")))
    first = unit_rows[0]
    if first["component_count"] != "3737":
        raise SystemExit("Table 6 baseline component count mismatch")
    if float(first["lower_log_ratio"]) != -0.00819 or float(first["upper_log_ratio"]) != -0.00298:
        raise SystemExit("Table 6 baseline interval mismatch")
    table3 = (ROOT / "outputs" / "main_certification_frontier_table.tex").read_text(encoding="utf-8")
    if "0.91480" not in table3:
        raise SystemExit("Table 3 free-anchor upper bound missing")
    margin = read_one(ROOT / "outputs" / "margin_sufficiency_audit_summary.csv")
    if margin.get("transition_row_mismatch_share") != "0.0":
        raise SystemExit("margin-sufficiency transition-row audit mismatch")
    if float(margin.get("max_abs_delta_log_q", "nan")) != 0.0:
        raise SystemExit("represented-leg Q dispersion audit mismatch")
    for label in ["Table 3", "Table 4", "Table 5", "Table 6", "Table 7"]:
        print(f"{label} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
