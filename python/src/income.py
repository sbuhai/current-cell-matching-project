"""Idiosyncratic earnings and deterministic labor-efficiency blocks."""

from __future__ import annotations

import numpy as np

from .calibration import Calibration


def earnings_states(calibration: Calibration) -> np.ndarray:
    """Return the discrete earnings states."""

    return np.array(calibration.income_states, dtype=float)


def earnings_transition(calibration: Calibration) -> np.ndarray:
    """Return the Markov transition matrix for the earnings process."""

    return np.array(calibration.income_transition, dtype=float)


def stationary_earnings_distribution(transition: np.ndarray) -> np.ndarray:
    """Return the stationary distribution of the earnings chain."""

    values, vectors = np.linalg.eig(transition.T)
    idx = np.argmin(np.abs(values - 1.0))
    stationary = np.real(vectors[:, idx])
    stationary = stationary / stationary.sum()
    stationary = np.maximum(stationary, 0.0)
    return stationary / stationary.sum()


def _stylized_hump_profile(calibration: Calibration) -> np.ndarray:
    """Return a stylized hump-shaped deterministic efficiency profile.

    The profile is normalized to unit mean over working ages. It is intended as a
    transparent robustness device motivated by standard life-cycle efficiency-unit
    estimates such as Hansen (1993), not as a new calibration target.
    """

    ages = np.array(calibration.ages, dtype=float)
    anchor_ages = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 64], dtype=float)
    anchor_values = np.array([0.74, 0.84, 0.95, 1.04, 1.11, 1.16, 1.18, 1.12, 1.01, 0.89], dtype=float)
    profile = np.ones(calibration.periods, dtype=float)
    work_mask = ages < calibration.retirement_age
    profile[work_mask] = np.interp(ages[work_mask], anchor_ages, anchor_values)
    profile[work_mask] /= np.mean(profile[work_mask])
    return profile


def age_efficiency_profile(calibration: Calibration) -> np.ndarray:
    """Return the deterministic age-efficiency profile."""

    if calibration.age_efficiency_mode == "flat":
        return np.ones(calibration.periods, dtype=float)
    if calibration.age_efficiency_mode == "hump":
        return _stylized_hump_profile(calibration)
    raise NotImplementedError(
        f"Unknown age-efficiency mode: {calibration.age_efficiency_mode}"
    )


def labor_income(calibration: Calibration, age_index: int, earnings_state: float, efficiency_profile: np.ndarray) -> float:
    """Return labor or pension income at a given age-state."""

    if age_index < calibration.retirement_index:
        return calibration.wage * efficiency_profile[age_index] * earnings_state
    return calibration.wage * calibration.pension_replacement
