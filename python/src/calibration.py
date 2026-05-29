"""Calibration objects for the baseline quantitative benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass(frozen=True)
class Calibration:
    """Container for the benchmark and robustness calibration blocks.

    The baseline deliberately keeps the deterministic life-cycle block minimal so
    that the variation in the theorem objects is driven by horizons, survival,
    uninsurable earnings risk, and borrowing constraints rather than by a rich
    deterministic earnings profile. The harder-environment robustness blocks add
    a stylized hump-shaped deterministic efficiency profile and a richer earnings
    process while preserving the paper's direct mapping from calibration inputs to
    theorem objects.
    """

    name: str
    min_age: int = 20
    max_age: int = 85
    retirement_age: int = 65
    beta: float = 0.9569
    sigma: float = 1.3969
    interest_rate: float = 0.0344
    wage: float = 1.0
    pension_replacement: float = 0.50
    borrowing_limit: float = 0.0
    population_growth: float = 0.011
    asset_min: float = 0.0
    asset_max: float = 40.0
    asset_size: int = 800
    asset_curvature: float = 3.0
    income_states: Tuple[float, ...] = (0.73, 1.27)
    income_transition: Tuple[Tuple[float, ...], ...] = ((0.82, 0.18), (0.18, 0.82))
    survival_csv: Path = DATA_DIR / "nchs_2021_total_survival.csv"
    age_efficiency_mode: str = "flat"
    envelope_bin_count: int = 20
    social_discount_rates: Tuple[float, ...] = (0.00, 0.01, 0.02, 0.03, 0.0344)

    @property
    def gross_return(self) -> float:
        """Gross one-period risk-free return."""

        return 1.0 + self.interest_rate

    @property
    def ages(self) -> Tuple[int, ...]:
        """Economic ages in the finite-life benchmark."""

        return tuple(range(self.min_age, self.max_age + 1))

    @property
    def periods(self) -> int:
        """Number of model periods."""

        return len(self.ages)

    @property
    def retirement_index(self) -> int:
        """Index of the first retirement period."""

        return self.retirement_age - self.min_age


SOURCE_NOTES = {
    "entry_retirement_terminal": "Conesa and Krueger (1999)",
    "population_growth": "Conesa and Krueger (1999)",
    "survival": "National Center for Health Statistics 2021 total population life table",
    "beta_sigma_r": "Gourinchas and Parker (2002), optimal-weighting estimates",
    "earnings_two_state": "Conesa and Krueger (1999), symmetric two-state benchmark earnings process",
    "earnings_five_state": "Five-state Rouwenhorst robustness calibrated to preserve mean one, persistence 0.82, and the baseline dispersion in earnings levels",
    "replacement_ratio": "Conesa and Krueger (1999)",
    "borrowing_limit": "Aiyagari (1994); baseline no-unsecured-debt benchmark",
    "flat_efficiency": "Baseline normalization with flat deterministic efficiency profile",
    "hump_efficiency": "Stylized hump-shaped age-efficiency profile normalized to unit mean over working ages and motivated by standard efficiency-units profiles such as Hansen (1993)",
    "envelope_bins": "Finite-grid aggregation of comparison fibers by weighted quantiles of log contv",
    "robustness_pref": "Conesa and Krueger (1999), standard preference benchmark",
    "robustness_risk": "Karahan and Ozkan (2013); see also De Nardi, Fella, and Paz-Pardo (2020) and Sanchez and Wellschmied (2020)",
}


def _rouwenhorst_transition(size: int, persistence: float) -> np.ndarray:
    """Return a symmetric Rouwenhorst transition matrix."""

    if size < 2:
        raise ValueError("size must be at least 2")
    p = (1.0 + persistence) / 2.0
    q = p
    transition = np.array([[p, 1.0 - p], [1.0 - q, q]], dtype=float)
    for n in range(3, size + 1):
        prev = transition
        transition = np.zeros((n, n), dtype=float)
        transition[:-1, :-1] += p * prev
        transition[:-1, 1:] += (1.0 - p) * prev
        transition[1:, :-1] += (1.0 - q) * prev
        transition[1:, 1:] += q * prev
        transition[1:-1, :] /= 2.0
    transition /= transition.sum(axis=1, keepdims=True)
    return transition


def _binomial_weights(size: int) -> np.ndarray:
    """Return the stationary binomial weights for a symmetric Rouwenhorst chain."""

    from math import comb

    coeffs = np.array([comb(size - 1, j) for j in range(size)], dtype=float)
    return coeffs / coeffs.sum()


def _five_state_earnings_block(target_std_levels: float = 0.27, persistence: float = 0.82) -> tuple[Tuple[float, ...], Tuple[Tuple[float, ...], ...]]:
    """Construct a richer five-state earnings block for robustness.

    The states are centered in logs, normalized to mean one in levels, and scaled
    so that the stationary standard deviation of earnings levels approximately
    matches the baseline two-state benchmark.
    """

    size = 5
    weights = _binomial_weights(size)
    grid = np.arange(size, dtype=float) - (size - 1.0) / 2.0

    def std_for_scale(scale: float) -> float:
        levels = np.exp(scale * grid)
        levels /= float(np.dot(weights, levels))
        mean = float(np.dot(weights, levels))
        var = float(np.dot(weights, (levels - mean) ** 2))
        return float(np.sqrt(var))

    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if std_for_scale(mid) < target_std_levels:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    levels = np.exp(scale * grid)
    levels /= float(np.dot(weights, levels))
    transition = _rouwenhorst_transition(size, persistence)
    return tuple(float(x) for x in levels), tuple(tuple(float(y) for y in row) for row in transition)


def baseline_calibration() -> Calibration:
    """Return the baseline calibration used in the paper."""

    return Calibration(name="baseline")


def hump_efficiency_calibration() -> Calibration:
    """Return a harder-environment robustness block with hump-shaped efficiency."""

    return Calibration(
        name="hump_efficiency",
        age_efficiency_mode="hump",
        asset_size=600,
    )


def richer_state_calibration() -> Calibration:
    """Return a harder-environment robustness block with richer earnings states."""

    states, transition = _five_state_earnings_block()
    return Calibration(
        name="five_state_hump",
        age_efficiency_mode="hump",
        asset_size=450,
        income_states=states,
        income_transition=transition,
    )


def macro_robustness_calibration() -> Calibration:
    """Return a classic heterogeneous-agent robustness block."""

    return Calibration(
        name="macro_robustness",
        beta=0.97,
        sigma=2.0,
        interest_rate=0.06,
    )
