# Python companion for the quantitative block

This directory contains the reproducible quantitative companion for **Support, Units, and Current-Cell Matching in Social Discounting**.

The code computes, in a transparent finite-life benchmark, the lower-envelope objects that appear in the quantitative section and the paper appendix under the paper's chosen matching rule, risk-free margin, and terminal normalization:

- the private continuation-value derivative `upsilon`,
- the market benchmark `Q`,
- the implementability wedge `Lambda`,
- the minimal younger-versus-older normalized-weight ratio implied by a candidate social discount factor,
- the fiberwise log-spread `G`, and
- the conditional transport statistic `D`.

## Directory structure

- `src/calibration.py` stores the baseline calibration together with its source notes and alternative calibration blocks used for sensitivity checks.
- `src/demography.py` loads the survival schedule and stationary age weights.
- `src/income.py` defines the earnings process and deterministic efficiency profile.
- `src/household.py` solves the finite-horizon one-asset household problem.
- `src/distribution.py` propagates the stationary cross section.
- `src/theorem_objects.py` computes the theorem objects for each candidate social discount rate.
- `src/reporting.py` writes CSV summaries and the figures and tables used in the manuscript.
- `run_all.py` is the master replication driver. It rebuilds the generated environment records, logs, tables, figures, source maps, and main synchronization checks from a clean generated state.
- `run_baseline.py` executes the full baseline pipeline.
- `run_nearby_policy_audit.py` repackages the frozen nearby-target diagnostics into the compact main-text audit tables used in the current revision.
- `run_normalization_robustness.py` rebuilds the alternative-unit exact comparison graph under the age and earnings reference-slope metric `v_tilde = v / vbar`, reporting support, edge/component counts, graph intervals, and target diagnostics.
- `data/nchs_2021_total_survival.csv` contains the total-population one-year survival schedule used in the paper.
- `outputs/` stores generated artifacts.
- `logs/` stores plain-text run logs and run records for the scripts that generate the main tables and graph diagnostics.

## Comparison discipline implemented in code

The quantitative implementation now follows the paper's comparison discipline directly.

- Pairwise current-younger/current-older matches are constructed by **nearest-neighbor search in `log upsilon` within earnings state, with tolerance-compatible edges retained for graph diagnostics**.
- The within-person compensation factor `Q` is measured on the younger household's one-step risk-free saving margin.
- Asset bins are not used to define the comparison triple.
- Weighted `log upsilon` bins are introduced only after the pairwise objects have been computed, and only for the aggregation of the spread and conditional transport statistics.
- `outputs/baseline_matching_summary.csv` reports the finite-grid matching diagnostics.
- The alternative-unit diagnostic uses `v_tilde = upsilon / vbar_age_state`, recomputes support, the tolerance-retained edge correspondence after the stated search cap, graph envelopes, connected components, and component anchors, and then evaluates the same target pairs on the resulting graph.

## Baseline calibration logic

The baseline follows the paper closely.

- Demographic timing and the replacement ratio follow Conesa and Krueger (1999).
- Survival uses the 2021 total-population U.S. life table from the National Center for Health Statistics.
- The preference block uses the optimal-weighting estimates in Gourinchas and Parker (2002): `beta = 0.9569`, `sigma = 1.3969`, and `r = 0.0344`.
- The earnings block uses the symmetric two-state benchmark process reported in Conesa and Krueger (1999).
- The deterministic age-efficiency profile is normalized to one in the baseline run so that the theorem objects are driven by horizons, survival, retirement, risk, and borrowing constraints rather than by an exogenous hump profile.

The manuscript uses two explicitly labeled one-asset environments. The two-state benchmark produces the current-map and one-step bridge diagnostics in the quantitative audit. The compact graph and nearby PVEU-margin target-pair tables use the five-state hump block: the same preference, demographic, borrowing-limit, and interest-rate block, a 450-point grid, a hump-shaped deterministic efficiency profile, and a five-state Rouwenhorst earnings process with levels `(0.564, 0.738, 0.965, 1.261, 1.649)` and persistence `0.82`. The PVEU target-pair tables use the central row of this five-state process, earnings level `0.965`. Other calibration blocks are documented in `src/calibration.py`.

## Full replication

From this directory, the complete Python replication is:

```bash
python run_all.py
```

The master driver removes and rebuilds the generated `outputs/` and `logs/` directories by default, regenerates `python_version.txt`, `platform.txt`, and `pip_freeze.txt` for the active Python environment, runs the full-rebuild variants of the computational scripts where needed, validates the rebuilt frozen-output checks, and reruns the Tables 3-7 synchronization check. The command writes a step-by-step record to `logs/`, with the summary manifest in `logs/REPLICATION_LOG_MANIFEST.txt`.

To preserve existing generated outputs while rerunning the sequence, use:

```bash
python run_all.py --no-clean
```

The code requires Python 3.10 or later with the packages listed in `requirements.txt`.

## Partial runs

From this directory,

```bash
python run_baseline.py
```

The script writes the following main artifacts into `outputs/`:

- `baseline_calibration_summary.csv`
- `baseline_matching_summary.csv`
- `baseline_comparison_summary.csv`
- `baseline_results_table.tex`
- `baseline_long_horizon_summary.csv`
- `baseline_required_eta_ratio_by_age.pdf`
- `baseline_required_eta_ratio_by_age.png`
- `baseline_spread_distance_by_social_rate.pdf`
- `baseline_spread_distance_by_social_rate.png`

## Reproducibility notes

The code is intentionally compact and uses only `numpy` and `matplotlib`. The aim is interpretability and easy auditing, not computational maximalism. The baseline now uses an 800-point curved asset grid, which removes the isolated retirement-age spike that appeared on a much coarser 250-point grid while leaving the qualitative comparative statics unchanged. A reader can inspect the economic ingredients module by module and map every reported paper object back to a clearly named function.


## Replication package contents

This directory contains the solution scripts, public inputs, generated tables, generated numbers, graph diagnostics, source notes, and run logs/run records used for the manuscript tables and diagnostics. The manuscript reads the compact tables from `outputs/nearby_policy_audit_table.tex`, `outputs/component_exposure_compact_table.tex`, and `outputs/unit_wedge_protocol_table.tex`; the underlying CSV summaries are stored in the same directory. The canonical full replication command is `python run_all.py`. The compact main-text tables can also be regenerated from existing generated CSV files by running `python run_nearby_policy_audit.py` and `python run_final_diagnostics.py`; `python reproduce_main_tables.py` regenerates and checks Tables 3-7; the alternative-unit graph diagnostics can be regenerated with `python run_normalization_robustness.py`.

## Data availability

This directory contains the replication files for the project. The public replication package is available at https://github.com/sbuhai/current-cell-matching-project. No restricted-access microdata are used.
