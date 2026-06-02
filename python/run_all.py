"""Run the complete Python replication workflow from a clean generated state."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
MANIFEST_NAME = "REPLICATION_LOG_MANIFEST.txt"

COMMANDS = [
    ("python_compileall.log", "python -m compileall -q .", ["-m", "compileall", "-q", "."], {}),
    ("run_baseline.log", "python run_baseline.py", ["run_baseline.py"], {}),
    ("run_iteration2_extras.log", "python run_iteration2_extras.py", ["run_iteration2_extras.py"], {}),
    ("run_omitted_support_bounds.log", "python run_omitted_support_bounds.py", ["run_omitted_support_bounds.py"], {}),
    ("run_normalization_robustness.log", "python run_normalization_robustness.py", ["run_normalization_robustness.py"], {}),
    ("run_liquid_illiquid.log", "python run_liquid_illiquid.py", ["run_liquid_illiquid.py"], {}),
    ("run_frontier_policy_full_rebuild.log", "FULL_REBUILD=1 python run_frontier_policy.py", ["run_frontier_policy.py"], {"FULL_REBUILD": "1"}),
    ("run_frontier_policy.log", "python run_frontier_policy.py", ["run_frontier_policy.py"], {}),
    ("run_wedge_audit_full_rebuild.log", "FULL_REBUILD=1 python run_wedge_audit.py", ["run_wedge_audit.py"], {"FULL_REBUILD": "1"}),
    ("run_wedge_audit.log", "python run_wedge_audit.py", ["run_wedge_audit.py"], {}),
    ("run_robustness_full_rebuild.log", "FULL_REBUILD=1 python run_robustness.py", ["run_robustness.py"], {"FULL_REBUILD": "1"}),
    ("run_robustness.log", "python run_robustness.py", ["run_robustness.py"], {}),
    ("run_nearby_policy_audit.log", "python run_nearby_policy_audit.py", ["run_nearby_policy_audit.py"], {}),
    ("run_margin_sufficiency_audit.log", "python run_margin_sufficiency_audit.py", ["run_margin_sufficiency_audit.py"], {}),
    ("run_final_diagnostics.log", "python run_final_diagnostics.py", ["run_final_diagnostics.py"], {}),
    ("reproduce_main_tables.log", "python reproduce_main_tables.py", ["reproduce_main_tables.py"], {}),
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def display_text(text: str) -> str:
    package_root = str(PACKAGE_ROOT)
    root = str(ROOT)
    return text.replace(root, "python").replace(package_root, ".")


def clean_generated_files() -> None:
    for path in (OUTPUT_DIR, LOG_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for pycache in ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache)
    source_map = PACKAGE_ROOT / "MAIN_TABLE_SOURCE_MAP.csv"
    if source_map.exists():
        source_map.unlink()


def ensure_generated_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_capture(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def write_environment_files() -> None:
    version = run_capture(["--version"]).stdout.strip()
    write_text(ROOT / "python_version.txt", f"{version}\n")

    python_version = sys.version.replace("\n", " ").replace("\r", " ")
    platform_lines = [
        f"python_executable: {sys.executable}",
        f"python_version: {python_version}",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"processor: {platform.processor()}",
    ]
    write_text(ROOT / "platform.txt", "\n".join(platform_lines) + "\n")

    pip_freeze = run_capture(["-m", "pip", "freeze"]).stdout
    write_text(ROOT / "pip_freeze.txt", pip_freeze)


def run_logged(log_name: str, description: str, args: list[str], env_updates: dict[str, str]) -> None:
    log_path = LOG_DIR / log_name
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUTF8"] = "1"
    env.pop("FULL_REBUILD", None)
    env.update(env_updates)

    print(description, flush=True)
    write_text(log_path, f"===== RUNNING {description} at {utc_stamp()} =====\n")
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    elapsed = time.perf_counter() - start
    if result.stdout:
        stdout = display_text(result.stdout)
        append_text(log_path, stdout)
        if not result.stdout.endswith("\n"):
            append_text(log_path, "\n")
    append_text(log_path, f"real {elapsed:.2f}\n")
    append_text(log_path, f"===== EXIT {description} status {result.returncode} at {utc_stamp()} =====\n")
    if result.returncode != 0:
        raise SystemExit(f"{description} failed with status {result.returncode}. See {log_path}.")


def write_manifest() -> None:
    command_lines = "\n".join(f"  {description}" for _, description, _, _ in COMMANDS)
    manifest = f"""Replication log manifest for Who Counts as Young? Matching Rules for Social Discounting
Updated by the full Python replication runner, {utc_stamp()}.

Execution environment:
  - python_executable: {sys.executable}
  - python_version.txt, platform.txt, and pip_freeze.txt were regenerated from this environment.
  - Platform reported by Python: {platform.platform()}, {platform.machine()}.

Generated directories and files:
  - outputs/: generated tables, figures, CSV summaries, TeX number files, and source maps.
  - logs/: plain-text command logs for each replication step.
  - MAIN_TABLE_SOURCE_MAP.csv: duplicate of outputs/MAIN_TABLE_SOURCE_MAP.csv at the replication package root.

Run logs and run records:
  - python_compileall.log: successful syntax/bytecode compile check for all Python modules.
  - run_baseline.log: successful fresh baseline regeneration.
  - run_iteration2_extras.log: successful regeneration of add-on tables.
  - run_omitted_support_bounds.log: successful omitted-support bounds regeneration.
  - run_normalization_robustness.log: successful alternative-unit robustness regeneration.
  - run_liquid_illiquid.log: successful liquid/illiquid stress-test regeneration.
  - run_frontier_policy_full_rebuild.log: successful full rebuild of the frontier-policy outputs.
  - run_frontier_policy.log: successful validation of the rebuilt frontier-policy outputs.
  - run_wedge_audit_full_rebuild.log: successful full rebuild of the wedge-audit outputs.
  - run_wedge_audit.log: successful validation of the rebuilt wedge-audit outputs.
  - run_robustness_full_rebuild.log: successful full rebuild of the robustness outputs.
  - run_robustness.log: successful validation of the rebuilt robustness outputs.
  - run_nearby_policy_audit.log: successful support/rematching table regeneration.
  - run_margin_sufficiency_audit.log: successful current-map margin-sufficiency dispersion audit regeneration.
  - run_final_diagnostics.log: successful regeneration of manuscript Tables 3-7 and MAIN_TABLE_SOURCE_MAP.csv.
  - reproduce_main_tables.log: successful synchronization check for Tables 3-7.

Full replication command:
  python run_all.py

Command sequence:
{command_lines}
"""
    write_text(LOG_DIR / MANIFEST_NAME, manifest)


def main() -> int:
    if sys.version_info < (3, 10):
        raise SystemExit("This replication workflow requires Python 3.10 or later.")

    parser = argparse.ArgumentParser(description="Run the complete Python replication workflow.")
    parser.add_argument("--no-clean", action="store_true", help="keep existing generated outputs and logs before running")
    args = parser.parse_args()

    if args.no_clean:
        ensure_generated_dirs()
    else:
        clean_generated_files()

    run_all_log = LOG_DIR / "run_all.log"
    write_text(run_all_log, f"===== RUNNING python run_all.py at {utc_stamp()} =====\n")
    append_text(run_all_log, f"python_executable: {sys.executable}\n")
    append_text(run_all_log, "working_directory: python\n")

    try:
        write_environment_files()
        for log_name, description, command_args, env_updates in COMMANDS:
            run_logged(log_name, description, command_args, env_updates)
            append_text(run_all_log, f"{description}: status 0\n")

        write_manifest()
        append_text(run_all_log, f"===== EXIT python run_all.py status 0 at {utc_stamp()} =====\n")
        print("Full Python replication workflow completed successfully.", flush=True)
        return 0
    except BaseException as exc:
        append_text(run_all_log, f"failure: {exc}\n")
        append_text(run_all_log, f"===== EXIT python run_all.py status 1 at {utc_stamp()} =====\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
