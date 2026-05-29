"""Household problem for the finite-life one-asset benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import Calibration
from .income import age_efficiency_profile, earnings_states, earnings_transition, labor_income


@dataclass
class HouseholdSolution:
    """Numerical solution of the household problem."""

    asset_grid: np.ndarray
    value: np.ndarray
    consumption: np.ndarray
    policy_index: np.ndarray
    marginal_value: np.ndarray
    euler_wedge: np.ndarray
    strict_interior: np.ndarray
    efficiency_profile: np.ndarray
    transition: np.ndarray
    earnings_states: np.ndarray


def utility(consumption: np.ndarray, sigma: float) -> np.ndarray:
    """CRRA utility."""

    if np.any(consumption <= 0.0):
        raise ValueError("Consumption must be strictly positive.")
    if abs(sigma - 1.0) < 1e-12:
        return np.log(consumption)
    return np.power(consumption, 1.0 - sigma) / (1.0 - sigma)


def marginal_utility(consumption: np.ndarray, sigma: float) -> np.ndarray:
    """Marginal utility under CRRA preferences."""

    return np.power(consumption, -sigma)


def build_asset_grid(calibration: Calibration) -> np.ndarray:
    """Construct a curved asset grid concentrated near the borrowing limit."""

    base = np.linspace(0.0, 1.0, calibration.asset_size)
    curved = np.power(base, calibration.asset_curvature)
    grid = calibration.asset_min + (calibration.asset_max - calibration.asset_min) * curved
    grid[0] = calibration.borrowing_limit
    return grid


def solve_household(calibration: Calibration, survival: np.ndarray) -> HouseholdSolution:
    """Solve the finite-horizon household problem by backward induction."""

    grid = build_asset_grid(calibration)
    profile = age_efficiency_profile(calibration)
    eps = earnings_states(calibration)
    transition = earnings_transition(calibration)
    J = calibration.periods
    B = grid.size
    E = eps.size

    value = np.full((J, B, E), -np.inf, dtype=float)
    consumption = np.full((J, B, E), np.nan, dtype=float)
    policy_index = np.zeros((J, B, E), dtype=int)
    marginal_value = np.full((J, B, E), np.nan, dtype=float)
    euler_wedge = np.full((J, B, E), np.nan, dtype=float)
    strict_interior = np.zeros((J, B, E), dtype=bool)

    # Terminal age: consume all available resources.
    j = J - 1
    for e in range(E):
        income = labor_income(calibration, j, eps[e], profile)
        resources = income + calibration.gross_return * grid
        c = np.maximum(resources, 1e-12)
        value[j, :, e] = utility(c, calibration.sigma)
        consumption[j, :, e] = c
        marginal_value[j, :, e] = marginal_utility(c, calibration.sigma)
        policy_index[j, :, e] = 0

    for j in range(J - 2, -1, -1):
        expected_next = transition @ np.swapaxes(value[j + 1, :, :], 0, 1)
        expected_next = expected_next.T
        for e in range(E):
            income = labor_income(calibration, j, eps[e], profile)
            resources = income + calibration.gross_return * grid
            for i in range(B):
                feasible = resources[i] - grid
                utility_vec = np.full(B, -np.inf, dtype=float)
                mask = feasible > 1e-12
                utility_vec[mask] = utility(feasible[mask], calibration.sigma)
                continuation = calibration.beta * survival[j] * expected_next[:, e]
                candidate = utility_vec + continuation
                best = int(np.argmax(candidate))
                value[j, i, e] = candidate[best]
                consumption[j, i, e] = feasible[best]
                policy_index[j, i, e] = best
                strict_interior[j, i, e] = bool(grid[best] > calibration.borrowing_limit + 1e-12)
                marginal_value[j, i, e] = marginal_utility(np.array([feasible[best]]), calibration.sigma)[0]

    for j in range(J - 1):
        for i in range(B):
            for e in range(E):
                next_i = policy_index[j, i, e]
                expected_mu = float(transition[e, :] @ marginal_value[j + 1, next_i, :])
                denom = calibration.beta * survival[j] * calibration.gross_return * expected_mu
                if denom > 0.0:
                    euler_wedge[j, i, e] = marginal_value[j, i, e] / denom

    return HouseholdSolution(
        asset_grid=grid,
        value=value,
        consumption=consumption,
        policy_index=policy_index,
        marginal_value=marginal_value,
        euler_wedge=euler_wedge,
        strict_interior=strict_interior,
        efficiency_profile=profile,
        transition=transition,
        earnings_states=eps,
    )
