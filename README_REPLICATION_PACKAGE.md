# Replication package for When Does Social Discounting Favor the Young? Welfare Comparisons in Heterogeneous Economies

This repository contains the current paper PDF and the Python replication package for **When Does Social Discounting Favor the Young? Welfare Comparisons in Heterogeneous Economies**. The replication files contain the code, public input files, generated table sources, and run records used in the quantitative application. They reproduce the finite-life household solutions, state matching, comparison-network diagnostics, identified intervals, component-exposure calculations, and robustness exercises reported in the manuscript.

## Reproduction

From this package directory, run:

```bash
python python/run_all.py
```

The code requires Python 3.10 or later and the packages listed in `python/requirements.txt`. The master script rebuilds generated outputs and logs, reruns the manuscript-facing diagnostics, and checks the reported table values. A faster verification of the maintained manuscript values is:

```bash
PYTHONPATH=python python python/run_final_diagnostics.py
PYTHONPATH=python python python/reproduce_main_tables.py
```

## Contents

- `paper/current_cell_matching.pdf`: current paper PDF.
- `python/src/`: model, calibration, household, distribution, and reporting modules.
- `python/data/`: public input data.
- `python/outputs/`: generated tables, CSV summaries, and diagnostic files.
- `python/logs/`: run records for the generated results.
- `python/run_all.py`: complete replication driver.
- `MAIN_TABLE_SOURCE_MAP_COMPACT.csv`: compact map from main-text and appendix table values to scripts and output variables.

No restricted-access microdata are used. The public project repository is https://github.com/sbuhai/current-cell-matching-project.
