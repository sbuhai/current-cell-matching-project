"""Stylized liquid/illiquid life-cycle block used for the richer-state appendix.

The goal of this module is not to deliver a fully calibrated HANK workhorse. It
provides a compact convex two-asset environment in which the theorem-guided
current map must account for current illiquid structure as well as current
earnings. The block is intentionally lightweight so it can be solved quickly in
this revision environment while remaining economically interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LiquidIlliquidCalibration:
    """Calibration for the stylized convex liquid/illiquid block.

    The object mimics the subset of the one-asset ``Calibration`` interface used
    by the theorem-object helpers so the same support, contamination, and policy
    routines can be reused without rewriting them for the appendix block.
    """

    name: str = "liquid_illiquid"
    min_age: int = 20
    max_age: int = 70
    retirement_age: int = 65
    beta: float = 0.9569
    sigma: float = 1.40
    interest_rate: float = 0.0344
    illiquid_return: float = 0.0800
    wage: float = 1.0
    pension_replacement: float = 0.50
    population_growth: float = 0.011
    liquid_asset_grid: tuple[float, ...] = (0.0, 0.05, 0.40, 0.75, 1.10, 1.45, 1.80, 2.15, 2.50, 2.85, 3.20, 3.55, 3.90, 4.25, 4.60, 4.95, 5.30, 5.65, 6.00, 6.35, 6.70, 7.05, 7.40, 7.75, 8.00)
    illiquid_asset_grid: tuple[float, ...] = (0.0, 0.50, 1.00, 1.70, 2.60, 3.80, 5.20)
    earnings_levels: tuple[float, ...] = (0.70, 1.00, 1.30)
    earnings_transition_matrix: tuple[tuple[float, ...], ...] = (
        (0.85, 0.14, 0.01),
        (0.08, 0.84, 0.08),
        (0.01, 0.14, 0.85),
    )
    adjustment_cost_scale: float = 0.04
    envelope_bin_count: int = 10
    age_efficiency_mode: str = "hump"

    @property
    def gross_return(self) -> float:
        return 1.0 + self.interest_rate

    @property
    def gross_illiquid_return(self) -> float:
        return 1.0 + self.illiquid_return

    @property
    def ages(self) -> tuple[int, ...]:
        return tuple(range(self.min_age, self.max_age + 1))

    @property
    def periods(self) -> int:
        return len(self.ages)

    @property
    def retirement_index(self) -> int:
        return self.retirement_age - self.min_age

    @property
    def asset_grid(self) -> np.ndarray:
        return np.asarray(self.liquid_asset_grid, dtype=float)

    @property
    def illiquid_grid(self) -> np.ndarray:
        return np.asarray(self.illiquid_asset_grid, dtype=float)

    @property
    def income_states(self) -> tuple[float, ...]:
        # The theorem-object utilities only use the length of this tuple and
        # natural index groupings over it. We therefore expose the combined
        # (illiquid-state, earnings-state) labels here.
        count = len(self.illiquid_asset_grid) * len(self.earnings_levels)
        return tuple(float(i) for i in range(count))

    @property
    def income_transition(self) -> tuple[tuple[float, ...], ...]:
        count = len(self.income_states)
        return tuple(tuple(1.0 if i == j else 0.0 for j in range(count)) for i in range(count))

    @property
    def asset_min(self) -> float:
        return float(self.asset_grid.min())

    @property
    def asset_max(self) -> float:
        return float(self.asset_grid.max())

    @property
    def asset_size(self) -> int:
        return int(self.asset_grid.size)


@dataclass
class LiquidIlliquidSolution:
    """Numerical solution of the stylized convex two-asset problem."""

    asset_grid: np.ndarray
    value: np.ndarray
    consumption: np.ndarray
    policy_index: np.ndarray
    next_illiquid_index: np.ndarray
    marginal_value: np.ndarray
    euler_wedge: np.ndarray
    strict_interior: np.ndarray
    efficiency_profile: np.ndarray
    transition: np.ndarray
    earnings_states: np.ndarray
    illiquid_grid: np.ndarray
    state_k_index: np.ndarray
    state_e_index: np.ndarray


def age_efficiency_profile(calibration: LiquidIlliquidCalibration) -> np.ndarray:
    """Stylized hump-shaped deterministic efficiency profile."""

    ages = np.asarray(calibration.ages, dtype=float)
    hump = np.exp(-0.5 * ((ages - 45.0) / 14.0) ** 2)
    hump = 0.8 + 0.4 * (hump - hump.min()) / max(hump.max() - hump.min(), 1e-12)
    working = ages < calibration.retirement_age
    hump[~working] = calibration.pension_replacement
    hump[working] /= float(np.mean(hump[working]))
    return hump


def survival_probabilities(calibration: LiquidIlliquidCalibration) -> np.ndarray:
    """Lightweight survival block, chosen only for the appendix illustration."""

    ages = np.asarray(calibration.ages, dtype=int)
    surv = np.ones_like(ages, dtype=float)
    surv[:-1] = 0.995
    surv[ages >= 60] = 0.980
    surv[ages >= 65] = 0.960
    surv[ages >= 68] = 0.940
    surv[-1] = 0.0
    return surv


def _crra(consumption: np.ndarray, sigma: float) -> np.ndarray:
    if np.any(consumption <= 0.0):
        raise ValueError("Consumption must be strictly positive.")
    if abs(sigma - 1.0) < 1e-12:
        return np.log(consumption)
    return np.power(consumption, 1.0 - sigma) / (1.0 - sigma)


def _marginal_utility(consumption: np.ndarray, sigma: float) -> np.ndarray:
    if np.any(consumption <= 0.0):
        raise ValueError("Consumption must be strictly positive.")
    return np.power(consumption, -sigma)


def stationary_earnings_distribution(transition: np.ndarray) -> np.ndarray:
    """Stationary distribution of the exogenous earnings chain."""

    eigvals, eigvecs = np.linalg.eig(transition.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    dist = np.real(eigvecs[:, idx])
    dist = np.abs(dist)
    dist /= dist.sum()
    return dist


def adjustment_cost(next_illiquid: np.ndarray | float, current_illiquid: float, scale: float) -> np.ndarray | float:
    """Differentiable convex adjustment cost."""

    diff = np.asarray(next_illiquid) - float(current_illiquid)
    return 0.5 * scale * diff**2


def _combined_state_arrays(calibration: LiquidIlliquidCalibration) -> tuple[np.ndarray, np.ndarray]:
    k_count = len(calibration.illiquid_asset_grid)
    e_count = len(calibration.earnings_levels)
    state_k = np.repeat(np.arange(k_count, dtype=int), e_count)
    state_e = np.tile(np.arange(e_count, dtype=int), k_count)
    return state_k, state_e


def solve_liquid_illiquid(calibration: LiquidIlliquidCalibration, survival: np.ndarray | None = None) -> LiquidIlliquidSolution:
    """Solve the stylized convex liquid/illiquid problem by backward induction."""

    survival_vec = survival_probabilities(calibration) if survival is None else np.asarray(survival, dtype=float)
    ages = np.asarray(calibration.ages, dtype=int)
    eff = age_efficiency_profile(calibration)
    b_grid = calibration.asset_grid
    k_grid = calibration.illiquid_grid
    eps = np.asarray(calibration.earnings_levels, dtype=float)
    transition = np.asarray(calibration.earnings_transition_matrix, dtype=float)
    state_k, state_e = _combined_state_arrays(calibration)

    J = ages.size
    B = b_grid.size
    K = k_grid.size
    E = eps.size
    S = state_k.size

    value = np.full((J, B, S), -np.inf, dtype=float)
    consumption = np.full((J, B, S), np.nan, dtype=float)
    policy_b_index = np.zeros((J, B, S), dtype=int)
    policy_k_index = np.zeros((J, B, S), dtype=int)
    marginal_value = np.full((J, B, S), np.nan, dtype=float)
    euler_wedge = np.full((J, B, S), np.nan, dtype=float)
    strict_interior = np.zeros((J, B, S), dtype=bool)

    for j in range(J - 1, -1, -1):
        if j < J - 1:
            continuation = np.zeros((B, K, E), dtype=float)
            continuation_mu = np.zeros((B, K, E), dtype=float)
            for e in range(E):
                for bp in range(B):
                    for kp in range(K):
                        next_states = kp * E + np.arange(E, dtype=int)
                        continuation[bp, kp, e] = float(transition[e, :] @ value[j + 1, bp, next_states])
                        continuation_mu[bp, kp, e] = float(transition[e, :] @ marginal_value[j + 1, bp, next_states])

        for bi, liquid in enumerate(b_grid):
            for ki, illiquid in enumerate(k_grid):
                for e in range(E):
                    s = ki * E + e
                    income = eff[j] * eps[e] if ages[j] < calibration.retirement_age else eff[j]
                    resources = income + calibration.gross_return * liquid + calibration.gross_illiquid_return * illiquid

                    if j == J - 1:
                        terminal_cost = float(adjustment_cost(0.0, illiquid, calibration.adjustment_cost_scale))
                        c = max(resources - terminal_cost, 1e-12)
                        value[j, bi, s] = float(_crra(np.array([c]), calibration.sigma)[0])
                        consumption[j, bi, s] = c
                        marginal_value[j, bi, s] = float(_marginal_utility(np.array([c]), calibration.sigma)[0])
                        policy_b_index[j, bi, s] = 0
                        policy_k_index[j, bi, s] = 0
                        continue

                    best_value = -np.inf
                    best_b = 0
                    best_k = 0
                    best_c = np.nan
                    for kp, next_illiquid in enumerate(k_grid):
                        adj = adjustment_cost(next_illiquid, illiquid, calibration.adjustment_cost_scale)
                        feasible_consumption = resources - next_illiquid - adj - b_grid
                        feasible = feasible_consumption > 1e-12
                        if not np.any(feasible):
                            continue
                        candidate = np.full(B, -np.inf, dtype=float)
                        candidate[feasible] = _crra(feasible_consumption[feasible], calibration.sigma) + calibration.beta * survival_vec[j] * continuation[feasible, kp, e]
                        bp = int(np.argmax(candidate))
                        if candidate[bp] > best_value:
                            best_value = float(candidate[bp])
                            best_b = bp
                            best_k = kp
                            best_c = float(feasible_consumption[bp])

                    value[j, bi, s] = best_value
                    consumption[j, bi, s] = best_c
                    policy_b_index[j, bi, s] = best_b
                    policy_k_index[j, bi, s] = best_k
                    marginal_value[j, bi, s] = float(_marginal_utility(np.array([best_c]), calibration.sigma)[0])
                    strict_interior[j, bi, s] = bool(b_grid[best_b] > 1e-12)

        if j < J - 1:
            for bi in range(B):
                for ki in range(K):
                    for e in range(E):
                        s = ki * E + e
                        bp = policy_b_index[j, bi, s]
                        kp = policy_k_index[j, bi, s]
                        denom = calibration.beta * survival_vec[j] * calibration.gross_return * continuation_mu[bp, kp, e]
                        if denom > 0.0:
                            euler_wedge[j, bi, s] = marginal_value[j, bi, s] / denom

    return LiquidIlliquidSolution(
        asset_grid=b_grid,
        value=value,
        consumption=consumption,
        policy_index=policy_b_index,
        next_illiquid_index=policy_k_index,
        marginal_value=marginal_value,
        euler_wedge=euler_wedge,
        strict_interior=strict_interior,
        efficiency_profile=eff,
        transition=np.eye(S, dtype=float),
        earnings_states=np.arange(S, dtype=float),
        illiquid_grid=k_grid,
        state_k_index=state_k,
        state_e_index=state_e,
    )


def stationary_joint_distribution_liquid_illiquid(
    calibration: LiquidIlliquidCalibration,
    solution: LiquidIlliquidSolution,
    survival: np.ndarray | None = None,
) -> np.ndarray:
    """Propagate the stationary joint distribution over age, liquid asset, and combined state."""

    survival_vec = survival_probabilities(calibration) if survival is None else np.asarray(survival, dtype=float)
    b_grid = calibration.asset_grid
    k_count = calibration.illiquid_grid.size
    eps_count = len(calibration.earnings_levels)
    transition = np.asarray(calibration.earnings_transition_matrix, dtype=float)

    J, B, S = solution.value.shape
    dist = np.zeros((J, B, S), dtype=float)
    initial_earnings = stationary_earnings_distribution(transition)
    for e in range(eps_count):
        dist[0, 0, e] = initial_earnings[e]

    growth = 1.0 + calibration.population_growth
    for j in range(J - 1):
        for bi in range(B):
            for s in range(S):
                mass = dist[j, bi, s]
                if mass <= 0.0:
                    continue
                current_e = int(solution.state_e_index[s])
                next_k = int(solution.next_illiquid_index[j, bi, s])
                next_b = int(solution.policy_index[j, bi, s])
                factor = mass * survival_vec[j] / growth
                for ep in range(eps_count):
                    next_s = next_k * eps_count + ep
                    dist[j + 1, next_b, next_s] += factor * transition[current_e, ep]

    total = float(dist.sum())
    if total <= 0.0:
        raise ValueError("Stationary distribution has zero mass.")
    return dist / total


def exact_state_groups(calibration: LiquidIlliquidCalibration) -> np.ndarray:
    """Fine current tuple: exact current illiquid index x current earnings state."""

    return np.arange(len(calibration.income_states), dtype=int)


def illiquid_tercile_groups(calibration: LiquidIlliquidCalibration) -> np.ndarray:
    """Operational theorem-guided coarsening: illiquid tercile x earnings state."""

    state_k, state_e = _combined_state_arrays(calibration)
    k_count = calibration.illiquid_grid.size
    k_group = np.zeros(k_count, dtype=int)
    first_cut = max(1, k_count // 3 + (1 if k_count % 3 else 0))
    second_cut = max(first_cut + 1, k_count - k_count // 3)
    k_group[first_cut:second_cut] = 1
    k_group[second_cut:] = 2
    e_count = len(calibration.earnings_levels)
    return np.array([e_count * k_group[k_idx] + state_e[s] for s, k_idx in enumerate(state_k)], dtype=int)


def earnings_only_groups(calibration: LiquidIlliquidCalibration) -> np.ndarray:
    """Current earnings state only, ignoring current illiquid structure."""

    _state_k, state_e = _combined_state_arrays(calibration)
    return state_e.astype(int)


def age_only_groups(calibration: LiquidIlliquidCalibration) -> np.ndarray:
    """Age-only current matching benchmark."""

    return np.zeros(len(calibration.income_states), dtype=int)


def mid_earnings_combined_state_indices(calibration: LiquidIlliquidCalibration) -> tuple[int, ...]:
    """Indices of combined states in the central current earnings row."""

    _state_k, state_e = _combined_state_arrays(calibration)
    mid = len(calibration.earnings_levels) // 2
    return tuple(int(i) for i in np.where(state_e == mid)[0])


def age_average_illiquid_holdings(
    calibration: LiquidIlliquidCalibration,
    solution: LiquidIlliquidSolution,
    dist: np.ndarray,
    ages_of_interest: Sequence[int],
) -> dict[int, float]:
    """Average current illiquid holdings by age under the stationary cross section."""

    ages = np.asarray(calibration.ages, dtype=int)
    current_illiquid = solution.illiquid_grid[solution.state_k_index]
    out: dict[int, float] = {}
    for age in ages_of_interest:
        idx = int(np.where(ages == age)[0][0])
        mass = float(dist[idx].sum())
        out[int(age)] = float(np.sum(dist[idx] * current_illiquid[None, :]) / mass) if mass > 0.0 else 0.0
    return out
