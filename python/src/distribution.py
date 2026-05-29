"""Cross-sectional distribution implied by household policies and demographics."""

from __future__ import annotations

import numpy as np

from .calibration import Calibration
from .household import HouseholdSolution
from .income import stationary_earnings_distribution


def stationary_joint_distribution(calibration: Calibration, solution: HouseholdSolution, survival: np.ndarray) -> np.ndarray:
    """Propagate the stationary joint distribution over age, assets, and earnings."""

    J, B, E = solution.value.shape
    dist = np.zeros((J, B, E), dtype=float)
    birth_asset_index = 0
    initial_income = stationary_earnings_distribution(solution.transition)
    dist[0, birth_asset_index, :] = initial_income
    growth = 1.0 + calibration.population_growth

    for j in range(J - 1):
        for i in range(B):
            for e in range(E):
                mass = dist[j, i, e]
                if mass <= 0.0:
                    continue
                next_i = solution.policy_index[j, i, e]
                factor = mass * survival[j] / growth
                dist[j + 1, next_i, :] += factor * solution.transition[e, :]

    total_mass = dist.sum()
    if total_mass <= 0.0:
        raise ValueError("Stationary distribution has zero mass.")
    return dist / total_mass
