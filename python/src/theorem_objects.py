"""Computation of theorem objects for the quantitative benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .calibration import Calibration
from .household import HouseholdSolution


@dataclass
class MatchingDiagnostics:
    """Summary diagnostics for current-younger/current-older matching."""

    mean_abs_log_gap: float
    median_abs_log_gap: float
    p95_abs_log_gap: float
    p99_abs_log_gap: float
    max_abs_log_gap: float
    weighted_mean_abs_log_gap: float
    weighted_p95_abs_log_gap: float
    weighted_p99_abs_log_gap: float
    weighted_max_abs_log_gap: float
    overall_support_overlap_share: float
    min_age_support_overlap_share: float


@dataclass
class MatchingStructure:
    """Finite-grid approximation to exact comparison-fiber matching."""

    older_asset_index: np.ndarray
    older_state_index: np.ndarray
    abs_log_gap: np.ndarray
    fiber_bin: np.ndarray
    state_group: np.ndarray
    diagnostics: MatchingDiagnostics
    rule_name: str = "nearest_log"
    group_name: str = "earnings_state"


@dataclass
class WeightedDistributionSummary:
    """Weighted summary of a one-dimensional positive object."""

    mean: float
    median: float
    p05: float
    p95: float
    p99: float
    maximum: float
    subunit_mass_share: float


@dataclass
class WedgeDiagnostics:
    """Diagnostics for raw and implemented wedges on positive-mass younger cells."""

    raw_all: WeightedDistributionSummary
    raw_strict_interior: WeightedDistributionSummary
    implemented: WeightedDistributionSummary
    strict_interior_mass_share: float
    lower_bound_mass_share: float
    raw_strict_interior_mean_abs_deviation: float
    raw_strict_interior_p95_abs_deviation: float
    raw_strict_interior_p99_abs_deviation: float


@dataclass
class ComparisonResults:
    """Theorem objects computed for one candidate social discount rate."""

    social_rate: float
    gross_social_factor: float
    required_eta_ratio: np.ndarray
    eta_profile: np.ndarray
    spread: float
    conditional_wasserstein: float
    mean_required_ratio: float
    peak_age_average_ratio: float
    epsilon_adjusted_spread: float
    epsilon_adjusted_conditional_wasserstein: float


@dataclass
class SensitivityResult:
    """Sensitivity of quantitative summaries to maintained implementation choices."""

    social_rate: float
    specification: str
    mean_required_ratio: float
    peak_age_average_ratio: float
    spread: float
    conditional_wasserstein: float


@dataclass
class ExcludedMassSummary:
    """Mass-share diagnostics for cells outside one-step support."""

    age_block: str
    earnings_group: str
    low_asset_share: float
    mid_asset_share: float
    high_asset_share: float
    total_share_of_excluded: float


@dataclass
class NetTermAgeSummary:
    """Age-block sign diagnostics for the benchmark net term log Lambda minus mismatch."""

    age_block: str
    support_share: float
    positive_share: float
    near_zero_share: float
    negative_share: float
    mean_net_term: float


@dataclass
class ApproximateSufficiencySummary:
    """Support and contamination diagnostics for a coarsened comparison map.

    The theorem-relevant scalar ``max_kappa`` is a radius, not an oscillation.
    The corresponding oscillation diagnostics are recorded separately so the
    implementation can report both the theoremic object and the descriptive
    within-cell spread from which it is constructed.
    """

    label: str
    state_groups: np.ndarray
    overall_support_overlap_share: float
    partition_label: str
    bin_count: int
    weighted_mean_oscillation: float
    weighted_p95_oscillation: float
    weighted_p99_oscillation: float
    max_oscillation: float
    weighted_mean_kappa: float
    weighted_p95_kappa: float
    weighted_p99_kappa: float
    max_kappa: float


@dataclass
class GraphEnvelopeSummary:
    """Maintained graph-robust envelope over admissible one-step correspondences."""

    label: str
    ages: np.ndarray
    point_age_average: np.ndarray
    lower_age_average: np.ndarray
    upper_age_average: np.ndarray


@dataclass
class PolicyRankingSummary:
    """Scale-free policy-ranking summary for a local budget-neutral comparison."""

    label: str
    support_label: str
    support_share: float
    target_a_support_share: float
    target_b_support_share: float
    kappa_mean: float
    kappa_max: float
    target_a_score: float
    target_b_score: float
    log_ratio: float
    lower_log_ratio: float
    upper_log_ratio: float


@dataclass
class PolicyDecompositionSummary:
    """Additive decomposition of the shift in the policy log-ranking statistic."""

    support_exclusion: float
    within_support_rematching: float
    total_change: float


@dataclass
class GraphTopologySummary:
    """Finite edge/component footprint for an exact-comparison graph."""

    edge_count: int
    node_count: int
    component_count: int
    target_component_count: int


@dataclass
class AlternativeNormalizationSummary:
    """Policy rankings after recomputing the graph under another private unit."""

    label: str
    support_label: str
    support_share: float
    target_a_support_share: float
    target_b_support_share: float
    alternative_support_label: str
    alternative_support_share: float
    alternative_target_a_support_share: float
    alternative_target_b_support_share: float
    baseline_edge_count: int
    baseline_component_count: int
    alternative_edge_count: int
    alternative_component_count: int
    baseline_log_ratio: float
    baseline_lower_log_ratio: float
    baseline_upper_log_ratio: float
    alternative_log_ratio: float
    alternative_lower_log_ratio: float
    alternative_upper_log_ratio: float


@dataclass
class KappaSensitivitySummary:
    """Sensitivity of contamination diagnostics to the number of log-contv bins."""

    label: str
    bin_count: int
    weighted_mean_kappa: float
    max_kappa: float


@dataclass
class EnvelopeWidthSensitivitySummary:
    """Sensitivity of graph-envelope width to the candidate-count parameter K."""

    label: str
    candidate_count: int
    mean_width: float
    max_width: float



def one_step_market_term(calibration: Calibration) -> float:
    """Gross market compensation factor for the one-step risk-free margin."""

    return calibration.gross_return



def natural_state_groups(calibration: Calibration) -> np.ndarray:
    """Natural comparison groups: each earnings state is its own group."""

    return np.arange(len(calibration.income_states), dtype=int)



def standardized_state_groups(calibration: Calibration) -> np.ndarray:
    """Standardized comparison groups used for cross-specification comparisons.

    We classify states into two common bins: below-mean current earnings and
    at-or-above-mean current earnings. This keeps the comparison map fixed when
    the earnings partition changes across robustness environments.
    """

    states = np.asarray(calibration.income_states, dtype=float)
    return np.where(states < 1.0, 0, 1).astype(int)



def required_eta_ratio(calibration: Calibration, wedge: np.ndarray, social_rate: float) -> np.ndarray:
    """Compute the minimal younger-versus-older normalized-weight ratio."""

    gross_social = 1.0 + social_rate
    q_term = one_step_market_term(calibration)
    ratio = q_term * wedge / gross_social
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    return ratio



def adjust_for_matching_error(required_ratio: np.ndarray, abs_log_gap: np.ndarray) -> np.ndarray:
    """Propagate the realized log-mismatch through the approximate-matching bound."""

    adjusted = required_ratio * np.exp(-abs_log_gap)
    adjusted = np.where(np.isfinite(required_ratio) & np.isfinite(abs_log_gap), adjusted, np.nan)
    return adjusted



def _log_unit_values(solution: HouseholdSolution, unit_values: np.ndarray | None = None) -> np.ndarray:
    """Log of the private metric used to define exact fibers."""

    values = solution.marginal_value if unit_values is None else np.asarray(unit_values, dtype=float)
    return np.log(np.maximum(values, 1e-300))



def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Return weighted quantiles of a one-dimensional sample."""

    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError("values and weights must be one-dimensional")
    if values.size == 0:
        raise ValueError("cannot compute weighted quantiles of an empty sample")
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = np.maximum(weights[sorter], 0.0)
    total = float(weights_sorted.sum())
    if total <= 0.0:
        return np.quantile(values_sorted, probs)
    cdf = np.cumsum(weights_sorted) / total
    return np.interp(probs, cdf, values_sorted, left=values_sorted[0], right=values_sorted[-1])



def _weighted_summary(values: np.ndarray, weights: np.ndarray) -> WeightedDistributionSummary:
    """Return weighted summary statistics for finite values."""

    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return WeightedDistributionSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values_f = np.asarray(values[finite], dtype=float)
    weights_f = np.asarray(weights[finite], dtype=float)
    quantiles = _weighted_quantiles(values_f, weights_f, np.array([0.05, 0.50, 0.95, 0.99]))
    total_weight = float(np.sum(weights_f))
    return WeightedDistributionSummary(
        mean=float(np.sum(values_f * weights_f) / total_weight),
        median=float(quantiles[1]),
        p05=float(quantiles[0]),
        p95=float(quantiles[2]),
        p99=float(quantiles[3]),
        maximum=float(np.max(values_f)),
        subunit_mass_share=float(np.sum(weights_f[values_f < 1.0]) / total_weight),
    )



def _support_overlap_mask(log_upsilon: np.ndarray, mu: np.ndarray, state_groups: np.ndarray) -> np.ndarray:
    """Return a cell-by-cell indicator for one-step support overlap."""

    J, B, E = log_upsilon.shape
    mask = np.zeros((J - 1, B, E), dtype=bool)
    for j in range(J - 1):
        for g in np.unique(state_groups):
            states = np.where(state_groups == g)[0]
            older_values = []
            for e in states:
                vals = log_upsilon[j + 1, :, e]
                finite = np.isfinite(vals)
                if np.any(finite):
                    older_values.append(vals[finite])
            if not older_values:
                continue
            older_concat = np.concatenate(older_values)
            lower = float(np.min(older_concat))
            upper = float(np.max(older_concat))
            for e in states:
                younger_mask = (mu[j, :, e] > 0.0) & np.isfinite(log_upsilon[j, :, e])
                mask[j, :, e] = younger_mask & (log_upsilon[j, :, e] >= lower) & (log_upsilon[j, :, e] <= upper)
    return mask



def _support_overlap_statistics(log_upsilon: np.ndarray, mu: np.ndarray, state_groups: np.ndarray) -> tuple[float, float]:
    """Mass share of younger cells that lie inside next-age log-upsilon support."""

    overlap_mask = _support_overlap_mask(log_upsilon, mu, state_groups)
    Jm1, _B, _E = overlap_mask.shape
    overall_overlap_mass = 0.0
    overall_mass = 0.0
    age_shares: list[float] = []

    for j in range(Jm1):
        age_mass = float(np.sum(mu[j, :, :]))
        if age_mass <= 0.0:
            continue
        overlap_mass = float(np.sum(mu[j, :, :][overlap_mask[j]]))
        overall_overlap_mass += overlap_mass
        overall_mass += age_mass
        age_shares.append(overlap_mass / age_mass)

    overall_share = overall_overlap_mass / overall_mass if overall_mass > 0.0 else 0.0
    min_age_share = float(min(age_shares)) if age_shares else 0.0
    return float(overall_share), min_age_share



def support_overlap_mask(
    solution: HouseholdSolution,
    mu: np.ndarray,
    state_groups: np.ndarray | None = None,
    unit_values: np.ndarray | None = None,
) -> np.ndarray:
    """Public wrapper returning the one-step support-overlap mask."""

    log_upsilon = _log_unit_values(solution, unit_values)
    if state_groups is None:
        state_groups = np.arange(log_upsilon.shape[2], dtype=int)
    return _support_overlap_mask(log_upsilon, mu, np.asarray(state_groups, dtype=int))



def _match_indices(values_y: np.ndarray, values_o: np.ndarray, rule: str) -> np.ndarray:
    """Return matched older flat indices for one age-group block."""

    if rule == "nearest_log":
        diff = np.abs(values_y[:, None] - values_o[None, :])
        return np.argmin(diff, axis=1)

    if rule == "nearest_level":
        diff = np.abs(np.exp(values_y)[:, None] - np.exp(values_o)[None, :])
        return np.argmin(diff, axis=1)

    if rule == "floor_log":
        idx = np.zeros(values_y.size, dtype=int)
        for i, value in enumerate(values_y):
            candidates = np.where(values_o <= value)[0]
            if candidates.size == 0:
                idx[i] = int(np.argmin(np.abs(values_o - value)))
            else:
                idx[i] = int(candidates[np.argmin(np.abs(values_o[candidates] - value))])
        return idx

    if rule == "ceil_log":
        idx = np.zeros(values_y.size, dtype=int)
        for i, value in enumerate(values_y):
            candidates = np.where(values_o >= value)[0]
            if candidates.size == 0:
                idx[i] = int(np.argmin(np.abs(values_o - value)))
            else:
                idx[i] = int(candidates[np.argmin(np.abs(values_o[candidates] - value))])
        return idx

    raise ValueError(f"Unknown matching rule: {rule}")



def build_matching_index(
    solution: HouseholdSolution,
    mu: np.ndarray,
    rule: str = "nearest_log",
    state_groups: np.ndarray | None = None,
    unit_values: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MatchingDiagnostics]:
    """Match each younger cell to an older cell under a maintained rule."""

    log_upsilon = _log_unit_values(solution, unit_values)
    J, B, E = log_upsilon.shape
    if state_groups is None:
        state_groups = np.arange(E, dtype=int)
    state_groups = np.asarray(state_groups, dtype=int)
    older_asset_index = np.zeros((J - 1, B, E), dtype=int)
    older_state_index = np.zeros((J - 1, B, E), dtype=int)
    abs_log_gap = np.full((J - 1, B, E), np.nan, dtype=float)

    for j in range(J - 1):
        for g in np.unique(state_groups):
            states = np.where(state_groups == g)[0]
            older_values_list = []
            older_assets_list = []
            older_states_list = []
            for e_old in states:
                older_values_list.append(log_upsilon[j + 1, :, e_old])
                older_assets_list.append(np.arange(B, dtype=int))
                older_states_list.append(np.full(B, e_old, dtype=int))
            older_values = np.concatenate(older_values_list)
            older_assets = np.concatenate(older_assets_list)
            older_states = np.concatenate(older_states_list)

            for e_y in states:
                flat_idx = _match_indices(log_upsilon[j, :, e_y], older_values, rule)
                older_asset_index[j, :, e_y] = older_assets[flat_idx]
                older_state_index[j, :, e_y] = older_states[flat_idx]
                abs_log_gap[j, :, e_y] = np.abs(log_upsilon[j, :, e_y] - older_values[flat_idx])

    relevant = (mu[:-1, :, :] > 0.0) & np.isfinite(abs_log_gap)
    values = abs_log_gap[relevant]
    weights = mu[:-1, :, :][relevant]
    overlap_share, min_age_overlap_share = _support_overlap_statistics(log_upsilon, mu, state_groups)
    if values.size == 0:
        diagnostics = MatchingDiagnostics(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            overlap_share,
            min_age_overlap_share,
        )
    else:
        weighted_quantiles = _weighted_quantiles(values, weights, np.array([0.95, 0.99]))
        diagnostics = MatchingDiagnostics(
            mean_abs_log_gap=float(np.mean(values)),
            median_abs_log_gap=float(np.quantile(values, 0.5)),
            p95_abs_log_gap=float(np.quantile(values, 0.95)),
            p99_abs_log_gap=float(np.quantile(values, 0.99)),
            max_abs_log_gap=float(np.max(values)),
            weighted_mean_abs_log_gap=float(np.sum(values * weights) / np.sum(weights)),
            weighted_p95_abs_log_gap=float(weighted_quantiles[0]),
            weighted_p99_abs_log_gap=float(weighted_quantiles[1]),
            weighted_max_abs_log_gap=float(np.max(values)),
            overall_support_overlap_share=overlap_share,
            min_age_support_overlap_share=min_age_overlap_share,
        )
    return older_asset_index, older_state_index, abs_log_gap, diagnostics



def _fiber_bins(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    state_groups: np.ndarray,
    bin_count: int | None = None,
    unit_values: np.ndarray | None = None,
) -> np.ndarray:
    """Assign each cell to a discrete log-upsilon bin within comparison group."""

    log_upsilon = _log_unit_values(solution, unit_values)
    J, B, E = log_upsilon.shape
    bin_count = calibration.envelope_bin_count if bin_count is None else int(bin_count)
    bin_labels = np.full((J, B, E), -1, dtype=int)
    probs = np.linspace(0.0, 1.0, bin_count + 1)

    for g in np.unique(state_groups):
        states = np.where(state_groups == g)[0]
        finite = np.zeros((J, B), dtype=bool)
        values_list = []
        weights_list = []
        for e in states:
            finite |= np.isfinite(log_upsilon[:, :, e])
            positive_mass = np.isfinite(log_upsilon[:, :, e]) & (mu[:, :, e] > 0.0)
            if np.any(positive_mass):
                values_list.append(log_upsilon[:, :, e][positive_mass])
                weights_list.append(mu[:, :, e][positive_mass])
        if not values_list:
            continue
        values = np.concatenate(values_list)
        weights = np.concatenate(weights_list)
        if np.allclose(values, values[0]):
            for e in states:
                bin_labels[:, :, e] = np.where(np.isfinite(log_upsilon[:, :, e]), 0, -1)
            continue
        edges = _weighted_quantiles(values, weights, probs)
        interior = edges[1:-1]
        for e in states:
            labels = np.searchsorted(interior, log_upsilon[:, :, e], side="right")
            bin_labels[:, :, e] = np.where(np.isfinite(log_upsilon[:, :, e]), labels, -1)
    return bin_labels



def build_matching_structure(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    rule: str = "nearest_log",
    state_groups: np.ndarray | None = None,
    group_name: str = "earnings_state",
    bin_count: int | None = None,
    unit_values: np.ndarray | None = None,
) -> MatchingStructure:
    """Construct the finite-grid comparison-fiber approximation."""

    if state_groups is None:
        state_groups = natural_state_groups(calibration)
    state_groups = np.asarray(state_groups, dtype=int)
    older_asset_index, older_state_index, abs_log_gap, diagnostics = build_matching_index(
        solution,
        mu,
        rule=rule,
        state_groups=state_groups,
        unit_values=unit_values,
    )
    fiber_bin = _fiber_bins(calibration, solution, mu, state_groups, bin_count=bin_count, unit_values=unit_values)
    return MatchingStructure(
        older_asset_index=older_asset_index,
        older_state_index=older_state_index,
        abs_log_gap=abs_log_gap,
        fiber_bin=fiber_bin,
        state_group=state_groups,
        diagnostics=diagnostics,
        rule_name=rule,
        group_name=group_name,
    )



def implemented_wedge(calibration: Calibration, solution: HouseholdSolution) -> np.ndarray:
    """Return the implemented wedge used in the paper's baseline numerics."""

    raw = np.asarray(solution.euler_wedge[:-1, :, :], dtype=float)
    strict_interior = np.asarray(solution.strict_interior[:-1, :, :], dtype=bool)
    wedge = np.where(strict_interior, 1.0, np.maximum(raw, 1.0))
    wedge = np.where(np.isfinite(raw) | strict_interior, wedge, np.nan)
    return wedge




def age_state_reference_upsilon(solution: HouseholdSolution, mu: np.ndarray) -> np.ndarray:
    """Age-state reference private scale based on a weighted geometric mean.

    The reference unit keeps age and current earnings state fixed but replaces the
    cell-specific private envelope derivative with the cross-sectional weighted
    geometric mean within that age-state cell. The alternative-unit diagnostic
    then rebuilds support, exact edges, graph envelopes, and components under
    the corresponding rescaled private metric.
    """

    base = np.asarray(solution.marginal_value, dtype=float)
    weights = np.asarray(mu, dtype=float)
    ref = np.empty_like(base)
    J, _B, E = base.shape
    for j in range(J):
        for e in range(E):
            mask = np.isfinite(base[j, :, e]) & (base[j, :, e] > 0.0) & (weights[j, :, e] > 0.0)
            if np.any(mask):
                value = float(np.exp(np.sum(np.log(base[j, mask, e]) * weights[j, mask, e]) / np.sum(weights[j, mask, e])))
            else:
                finite = np.isfinite(base[j, :, e]) & (base[j, :, e] > 0.0)
                value = float(np.exp(np.mean(np.log(base[j, finite, e])))) if np.any(finite) else 1.0
            ref[j, :, e] = value
    return ref



def age_state_rescaled_upsilon(solution: HouseholdSolution, mu: np.ndarray) -> np.ndarray:
    """Private metric after rescaling by age-state reference slopes.

    The alternative exact graph uses the metric v_tilde = v / v_bar_{a,k},
    where v_bar_{a,k} is the positive-mass weighted geometric mean of the
    baseline envelope within an age--earnings cell.
    """

    base = np.asarray(solution.marginal_value, dtype=float)
    reference = age_state_reference_upsilon(solution, mu)
    return base / np.maximum(reference, 1e-300)



def renormalize_eta(
    eta: np.ndarray,
    base_upsilon: np.ndarray,
    alternative_upsilon: np.ndarray,
) -> np.ndarray:
    """Re-express a maintained raw welfare-weight array in alternative units."""

    base = np.asarray(base_upsilon, dtype=float)
    alt = np.asarray(alternative_upsilon, dtype=float)
    raw = np.asarray(eta, dtype=float) * base
    return raw / alt



def wedge_diagnostics(calibration: Calibration, solution: HouseholdSolution, mu: np.ndarray) -> WedgeDiagnostics:
    """Return raw-versus-implemented wedge diagnostics on positive-mass younger cells."""

    mass = np.asarray(mu[:-1, :, :], dtype=float)
    raw = np.asarray(solution.euler_wedge[:-1, :, :], dtype=float)
    implemented = implemented_wedge(calibration, solution)
    strict_interior = np.asarray(solution.strict_interior[:-1, :, :], dtype=bool)
    relevant = (mass > 0.0) & np.isfinite(raw)
    interior_relevant = relevant & strict_interior
    lower_bound_relevant = relevant & (~strict_interior)

    raw_all = _weighted_summary(raw[relevant], mass[relevant])
    raw_strict_interior = _weighted_summary(raw[interior_relevant], mass[interior_relevant])
    implemented_summary = _weighted_summary(implemented[relevant], mass[relevant])

    total_mass = float(np.sum(mass[relevant])) if np.any(relevant) else 0.0
    strict_mass = float(np.sum(mass[interior_relevant])) if np.any(interior_relevant) else 0.0
    lower_bound_mass = float(np.sum(mass[lower_bound_relevant])) if np.any(lower_bound_relevant) else 0.0

    if strict_mass > 0.0:
        abs_dev = np.abs(raw[interior_relevant] - 1.0)
        abs_dev_quantiles = _weighted_quantiles(abs_dev, mass[interior_relevant], np.array([0.95, 0.99]))
        mean_abs_dev = float(np.sum(abs_dev * mass[interior_relevant]) / strict_mass)
        p95_abs_dev = float(abs_dev_quantiles[0])
        p99_abs_dev = float(abs_dev_quantiles[1])
    else:
        mean_abs_dev = 0.0
        p95_abs_dev = 0.0
        p99_abs_dev = 0.0

    return WedgeDiagnostics(
        raw_all=raw_all,
        raw_strict_interior=raw_strict_interior,
        implemented=implemented_summary,
        strict_interior_mass_share=(strict_mass / total_mass if total_mass > 0.0 else 0.0),
        lower_bound_mass_share=(lower_bound_mass / total_mass if total_mass > 0.0 else 0.0),
        raw_strict_interior_mean_abs_deviation=mean_abs_dev,
        raw_strict_interior_p95_abs_deviation=p95_abs_dev,
        raw_strict_interior_p99_abs_deviation=p99_abs_dev,
    )



def reconstruct_eta_profile(required_ratio: np.ndarray, older_asset_index: np.ndarray, older_state_index: np.ndarray) -> np.ndarray:
    """Reconstruct a pointwise lower-envelope normalized-weight profile recursively."""

    Jm1, B, E = required_ratio.shape
    J = Jm1 + 1
    eta = np.ones((J, B, E), dtype=float)
    for j in range(J - 2, -1, -1):
        safe_ratio = np.where(np.isfinite(required_ratio[j]), np.maximum(required_ratio[j], 1e-12), 1.0)
        for e in range(E):
            eta[j, :, e] = safe_ratio[:, e] * eta[j + 1, older_asset_index[j, :, e], older_state_index[j, :, e]]
    return eta



def fiberwise_log_spread(eta: np.ndarray, mu: np.ndarray, fiber_bin: np.ndarray, state_groups: np.ndarray, bin_count: int) -> float:
    """Return the maximal discrete fiberwise log-spread on positive-mass cells."""

    log_eta = np.log(np.maximum(eta, 1e-300))
    spread = 0.0
    for g in np.unique(state_groups):
        states = np.where(state_groups == g)[0]
        for q in range(bin_count):
            mask = np.zeros_like(mu, dtype=bool)
            for e in states:
                mask[:, :, e] = (fiber_bin[:, :, e] == q) & (mu[:, :, e] > 0.0)
            if not np.any(mask):
                continue
            values = log_eta[mask]
            spread = max(spread, float(np.max(values) - np.min(values)))
    return spread



def wasserstein_1d(weights_a: np.ndarray, weights_b: np.ndarray, positions: np.ndarray) -> float:
    """One-dimensional Wasserstein-1 distance on a fixed age grid."""

    wa = np.array(weights_a, dtype=float)
    wb = np.array(weights_b, dtype=float)
    wa = wa / wa.sum() if wa.sum() > 0 else wa
    wb = wb / wb.sum() if wb.sum() > 0 else wb
    cdfa = np.cumsum(wa)
    cdfb = np.cumsum(wb)
    deltas = np.diff(positions)
    return float(np.sum(np.abs(cdfa[:-1] - cdfb[:-1]) * deltas))



def conditional_wasserstein_age(
    eta: np.ndarray,
    mu: np.ndarray,
    ages: np.ndarray,
    fiber_bin: np.ndarray,
    state_groups: np.ndarray,
    bin_count: int,
) -> float:
    """Aggregate the within-fiber age-transport distance across discrete fibers."""

    J = eta.shape[0]
    total = 0.0
    mass_total = 0.0
    for g in np.unique(state_groups):
        states = np.where(state_groups == g)[0]
        for q in range(bin_count):
            neutral_by_age = np.zeros(J, dtype=float)
            weighted_by_age = np.zeros(J, dtype=float)
            for e in states:
                for j in range(J):
                    mask = fiber_bin[j, :, e] == q
                    if not np.any(mask):
                        continue
                    neutral_by_age[j] += float(np.sum(mu[j, mask, e]))
                    weighted_by_age[j] += float(np.sum(eta[j, mask, e] * mu[j, mask, e]))
            weighted_fiber_mass = float(np.sum(weighted_by_age))
            if weighted_fiber_mass <= 0.0:
                continue
            total += weighted_fiber_mass * wasserstein_1d(weighted_by_age, neutral_by_age, ages)
            mass_total += weighted_fiber_mass
    return total / mass_total if mass_total > 0.0 else 0.0



def age_averaged_required_ratio(required_ratio: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Age profile of the required ratio, weighted by younger-cell mass."""

    mass = mu[:-1, :, :]
    weight = mass / np.maximum(mass.sum(axis=(1, 2), keepdims=True), 1e-12)
    return np.nansum(required_ratio * weight, axis=(1, 2))



def age_quantile_required_ratio(required_ratio: np.ndarray, mu: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Weighted age-by-age quantiles of the required ratio across current fibers."""

    probs = np.asarray(probs, dtype=float)
    Jm1 = required_ratio.shape[0]
    out = np.full((probs.size, Jm1), np.nan, dtype=float)
    mass = mu[:-1, :, :]
    for j in range(Jm1):
        relevant = np.isfinite(required_ratio[j]) & np.isfinite(mass[j]) & (mass[j] > 0.0)
        if not np.any(relevant):
            continue
        values = required_ratio[j][relevant]
        weights = mass[j][relevant]
        out[:, j] = _weighted_quantiles(values, weights, probs)
    return out



def comparison_results(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    state_groups: np.ndarray | None = None,
    group_name: str = "earnings_state",
    rule: str = "nearest_log",
    bin_count: int | None = None,
) -> Tuple[List[ComparisonResults], MatchingStructure, WedgeDiagnostics]:
    """Compute theorem objects for each candidate social discount rate."""

    matching = build_matching_structure(
        calibration,
        solution,
        mu,
        rule=rule,
        state_groups=state_groups,
        group_name=group_name,
        bin_count=bin_count,
    )
    wedge = implemented_wedge(calibration, solution)
    wedge_diag = wedge_diagnostics(calibration, solution, mu)
    ages = np.array(calibration.ages, dtype=float)
    mass = mu[:-1, :, :]
    current_mass_total = float(np.sum(mass))
    actual_bin_count = calibration.envelope_bin_count if bin_count is None else int(bin_count)
    results: List[ComparisonResults] = []
    for social_rate in calibration.social_discount_rates:
        ratio = required_eta_ratio(calibration, wedge, social_rate)
        eta = reconstruct_eta_profile(ratio, matching.older_asset_index, matching.older_state_index)
        spread = fiberwise_log_spread(eta, mu, matching.fiber_bin, matching.state_group, actual_bin_count)
        distance = conditional_wasserstein_age(eta, mu, ages, matching.fiber_bin, matching.state_group, actual_bin_count)

        adjusted_ratio = adjust_for_matching_error(ratio, matching.abs_log_gap)
        adjusted_eta = reconstruct_eta_profile(adjusted_ratio, matching.older_asset_index, matching.older_state_index)
        adjusted_spread = fiberwise_log_spread(adjusted_eta, mu, matching.fiber_bin, matching.state_group, actual_bin_count)
        adjusted_distance = conditional_wasserstein_age(
            adjusted_eta,
            mu,
            ages,
            matching.fiber_bin,
            matching.state_group,
            actual_bin_count,
        )

        mean_ratio = float(np.nansum(ratio * mass) / current_mass_total)
        age_profile = age_averaged_required_ratio(ratio, mu)
        peak_ratio = float(np.nanmax(age_profile))
        results.append(
            ComparisonResults(
                social_rate=social_rate,
                gross_social_factor=1.0 + social_rate,
                required_eta_ratio=ratio,
                eta_profile=eta,
                spread=spread,
                conditional_wasserstein=distance,
                mean_required_ratio=mean_ratio,
                peak_age_average_ratio=peak_ratio,
                epsilon_adjusted_spread=adjusted_spread,
                epsilon_adjusted_conditional_wasserstein=adjusted_distance,
            )
        )
    return results, matching, wedge_diag



def sensitivity_results(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    social_rates: tuple[float, ...] | list[float] | None = None,
    state_groups: np.ndarray | None = None,
    bin_count: int | None = None,
) -> List[SensitivityResult]:
    """Return sensitivity of geometric summaries to maintained matching choices.

    By default we report a small grid away from the special case $\rho^S=r$.
    """

    if social_rates is None:
        social_rates = (0.00, 0.02, calibration.interest_rate)
    actual_bin_count = calibration.envelope_bin_count if bin_count is None else int(bin_count)
    state_groups = natural_state_groups(calibration) if state_groups is None else np.asarray(state_groups, dtype=int)
    wedge = implemented_wedge(calibration, solution)
    ages = np.array(calibration.ages, dtype=float)
    rows: list[SensitivityResult] = []

    for social_rate in social_rates:
        ratio = required_eta_ratio(calibration, wedge, float(social_rate))
        for rule_name, label in [
            ("nearest_log", r"Nearest $\log \contv$"),
            ("nearest_level", r"Nearest level $\contv$"),
            ("floor_log", r"Lower-sided $\log \contv$"),
            ("ceil_log", r"Upper-sided $\log \contv$"),
        ]:
            matching = build_matching_structure(
                calibration,
                solution,
                mu,
                rule=rule_name,
                state_groups=state_groups,
                group_name="custom",
                bin_count=actual_bin_count,
            )
            eta = reconstruct_eta_profile(ratio, matching.older_asset_index, matching.older_state_index)
            age_profile = age_averaged_required_ratio(ratio, mu)
            rows.append(
                SensitivityResult(
                    social_rate=float(social_rate),
                    specification=label,
                    mean_required_ratio=float(np.nansum(ratio * mu[:-1]) / np.sum(mu[:-1])),
                    peak_age_average_ratio=float(np.nanmax(age_profile)),
                    spread=fiberwise_log_spread(eta, mu, matching.fiber_bin, matching.state_group, actual_bin_count),
                    conditional_wasserstein=conditional_wasserstein_age(
                        eta,
                        mu,
                        ages,
                        matching.fiber_bin,
                        matching.state_group,
                        actual_bin_count,
                    ),
                )
            )
    return rows



def excluded_mass_summaries(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    state_groups: np.ndarray | None = None,
) -> list[ExcludedMassSummary]:
    """Summarize where younger cells fall outside one-step overlap support."""

    log_upsilon = np.log(np.maximum(solution.marginal_value, 1e-300))
    state_groups = natural_state_groups(calibration) if state_groups is None else np.asarray(state_groups, dtype=int)
    overlap = _support_overlap_mask(log_upsilon, mu, state_groups)
    excluded = (~overlap) & (mu[:-1] > 0.0)
    excluded_mass = mu[:-1][excluded]
    total_excluded_mass = float(np.sum(excluded_mass))
    if total_excluded_mass <= 0.0:
        return []

    ages = np.array(calibration.ages[:-1], dtype=int)
    age_block_bounds = [(20, 29), (30, 39), (40, 49), (50, 59), (60, calibration.retirement_age - 1)]
    age_block_labels = [f"{lo}-{hi}" for lo, hi in age_block_bounds]
    asset_grid = np.asarray(solution.asset_grid, dtype=float)
    asset_edges = np.quantile(asset_grid, [1 / 3, 2 / 3])
    states = np.asarray(calibration.income_states, dtype=float)
    if states.size >= 5:
        earnings_panel_names = {
            0: "Current earnings: low",
            1: "Current earnings: middle",
            2: "Current earnings: high",
        }
        earnings_panel_index = np.zeros(states.size, dtype=int)
        earnings_panel_index[:2] = 0
        earnings_panel_index[2:3] = 1
        earnings_panel_index[3:] = 2
    else:
        earnings_panel_names = {0: "Current earnings: low", 1: "Current earnings: high"}
        earnings_panel_index = np.arange(states.size, dtype=int)

    summaries: list[ExcludedMassSummary] = []
    for panel in np.unique(earnings_panel_index):
        states_in_panel = np.where(earnings_panel_index == panel)[0]
        panel_mask = np.zeros_like(excluded, dtype=bool)
        panel_mask[:, :, states_in_panel] = True
        panel_total = float(np.sum(mu[:-1][excluded & panel_mask]))
        if panel_total <= 0.0:
            continue
        for (lo, hi), label in zip(age_block_bounds, age_block_labels):
            age_mask = (ages >= lo) & (ages <= hi)
            block_mask = np.zeros_like(excluded, dtype=bool)
            block_mask[age_mask, :, :] = True
            block_mask &= panel_mask & excluded
            block_mass = float(np.sum(mu[:-1][block_mask]))
            if block_mass <= 0.0:
                summaries.append(
                    ExcludedMassSummary(label, earnings_panel_names[int(panel)], 0.0, 0.0, 0.0, 0.0)
                )
                continue
            low_mask = np.zeros_like(block_mask, dtype=bool)
            mid_mask = np.zeros_like(block_mask, dtype=bool)
            high_mask = np.zeros_like(block_mask, dtype=bool)
            low_idx = np.where(asset_grid <= asset_edges[0])[0]
            mid_idx = np.where((asset_grid > asset_edges[0]) & (asset_grid <= asset_edges[1]))[0]
            high_idx = np.where(asset_grid > asset_edges[1])[0]
            low_mask[:, low_idx, :] = True
            mid_mask[:, mid_idx, :] = True
            high_mask[:, high_idx, :] = True
            summaries.append(
                ExcludedMassSummary(
                    age_block=label,
                    earnings_group=earnings_panel_names[int(panel)],
                    low_asset_share=float(np.sum(mu[:-1][block_mask & low_mask]) / block_mass),
                    mid_asset_share=float(np.sum(mu[:-1][block_mask & mid_mask]) / block_mass),
                    high_asset_share=float(np.sum(mu[:-1][block_mask & high_mask]) / block_mass),
                    total_share_of_excluded=float(block_mass / total_excluded_mass),
                )
            )
    return summaries


def net_term_age_summaries(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    state_groups: np.ndarray | None = None,
    *,
    tolerance: float = 1e-10,
) -> list[NetTermAgeSummary]:
    """Age-block sign diagnostics for the net term log Lambda minus mismatch."""

    state_groups = natural_state_groups(calibration) if state_groups is None else np.asarray(state_groups, dtype=int)
    matching = build_matching_structure(calibration, solution, mu, state_groups=state_groups)
    overlap = support_overlap_mask(solution, mu, state_groups)
    log_wedge = np.log(np.maximum(implemented_wedge(calibration, solution), 1e-300))
    net_term = log_wedge - np.asarray(matching.abs_log_gap, dtype=float)
    younger_mass = np.asarray(mu[:-1], dtype=float)
    ages = np.array(calibration.ages[:-1], dtype=int)
    age_block_bounds = [
        (20, 29),
        (30, 39),
        (40, 49),
        (50, 59),
        (60, 69),
        (70, 79),
        (80, int(calibration.max_age - 1)),
    ]

    summaries: list[NetTermAgeSummary] = []
    for lo, hi in age_block_bounds:
        age_mask = (ages >= lo) & (ages <= hi)
        if not np.any(age_mask):
            continue
        selector = np.zeros_like(overlap, dtype=bool)
        selector[age_mask, :, :] = True
        selector &= overlap & np.isfinite(net_term) & (younger_mass > 0.0)
        block_mass = float(np.sum(younger_mass[selector]))
        age_total_mass = float(np.sum(younger_mass[age_mask, :, :]))
        if block_mass <= 0.0 or age_total_mass <= 0.0:
            summaries.append(
                NetTermAgeSummary(
                    age_block=f"{lo}-{hi}",
                    support_share=0.0,
                    positive_share=0.0,
                    near_zero_share=0.0,
                    negative_share=0.0,
                    mean_net_term=0.0,
                )
            )
            continue
        block_values = net_term[selector]
        block_weights = younger_mass[selector]
        positive = float(np.sum(block_weights[block_values > tolerance]) / block_mass)
        near_zero = float(np.sum(block_weights[np.abs(block_values) <= tolerance]) / block_mass)
        negative = float(np.sum(block_weights[block_values < -tolerance]) / block_mass)
        mean_net = float(np.sum(block_values * block_weights) / block_mass)
        summaries.append(
            NetTermAgeSummary(
                age_block=f"{lo}-{hi}",
                support_share=float(block_mass / age_total_mass),
                positive_share=positive,
                near_zero_share=near_zero,
                negative_share=negative,
                mean_net_term=mean_net,
            )
        )
    return summaries


def three_block_state_groups(calibration: Calibration) -> np.ndarray:
    """A contiguous three-block coarsening used for the support-sufficiency frontier."""

    state_count = len(calibration.income_states)
    if state_count <= 3:
        return natural_state_groups(calibration)
    if state_count == 5:
        return np.array([0, 0, 1, 2, 2], dtype=int)
    cut_low = max(1, state_count // 3)
    cut_high = max(cut_low + 1, state_count - state_count // 3)
    groups = np.zeros(state_count, dtype=int)
    groups[cut_low:cut_high] = 1
    groups[cut_high:] = 2
    return groups


def pooled_state_groups(calibration: Calibration) -> np.ndarray:
    """Age-only current matching benchmark: no non-age state partition is held fixed."""

    return np.zeros(len(calibration.income_states), dtype=int)


def support_sufficiency_group_specs(calibration: Calibration) -> list[tuple[str, np.ndarray]]:
    """Nested coarsenings used for the support-sufficiency frontier."""

    specs: list[tuple[str, np.ndarray]] = [
        ("Verified current-cell map", natural_state_groups(calibration)),
    ]
    state_count = len(calibration.income_states)
    if state_count >= 5:
        specs.append(("Three-block coarsening", three_block_state_groups(calibration)))
    specs.append(("Standardized heuristic map", standardized_state_groups(calibration)))
    specs.append(("Age-only current matching", pooled_state_groups(calibration)))
    return specs


def age_averaged_eta(eta: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Age profile of normalized multipliers, weighted by realized cross-sectional mass."""

    age_mass = np.maximum(mu.sum(axis=(1, 2)), 1e-300)
    return (eta * mu).sum(axis=(1, 2)) / age_mass


def _weighted_spread_summary(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """Mass-weighted mean, p95, and p99 of a nonnegative summary object."""

    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return 0.0, 0.0, 0.0
    values_f = np.asarray(values[finite], dtype=float)
    weights_f = np.asarray(weights[finite], dtype=float)
    total_weight = float(np.sum(weights_f))
    mean = float(np.sum(values_f * weights_f) / total_weight)
    p95, p99 = _weighted_quantiles(values_f, weights_f, np.array([0.95, 0.99]))
    return mean, float(p95), float(p99)


def _partition_label(partition_mode: str, bin_count: int) -> str:
    """Human-readable description of the operative contamination partition."""

    if partition_mode == "baseline":
        return f"Age x coarse weighted quantiles (N_q={bin_count})"
    if partition_mode == "shifted":
        return f"Shifted weighted quantiles (N_q={bin_count})"
    if partition_mode == "fixed_age":
        return f"Age-specific fixed cutoffs (N_q={bin_count})"
    if partition_mode == "pooled":
        return f"Pooled cutoffs (N_q={bin_count})"
    raise ValueError(f"Unknown partition_mode: {partition_mode}")



def _weighted_bin_cutoffs(
    values: np.ndarray,
    weights: np.ndarray,
    bin_count: int,
    *,
    shifted: bool = False,
) -> np.ndarray:
    """Weighted cutoffs used to partition log-upsilon cells."""

    actual_bin_count = max(1, int(bin_count))
    if actual_bin_count <= 1 or values.size == 0 or np.allclose(values, values[0]):
        return np.array([], dtype=float)
    if shifted:
        probs = (np.arange(actual_bin_count - 1, dtype=float) + 0.5) / actual_bin_count
    else:
        probs = np.arange(1, actual_bin_count, dtype=float) / actual_bin_count
    cutoffs = np.asarray(_weighted_quantiles(values, weights, probs), dtype=float)
    return np.unique(cutoffs)



def _assign_partition_bins(
    log_upsilon_slice: np.ndarray,
    merged_mask: np.ndarray,
    cutoffs: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Assign cells to an operative contamination partition."""

    bin_index = np.full(log_upsilon_slice.shape, -1, dtype=int)
    if cutoffs.size == 0:
        bin_index[merged_mask] = 0
        return bin_index, 1
    bin_index[merged_mask] = np.searchsorted(cutoffs, log_upsilon_slice[merged_mask], side="right")
    return bin_index, int(cutoffs.size + 1)



def approximate_sufficiency_summary(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    state_groups: np.ndarray,
    label: str,
    social_rate: float | None = None,
    bin_count: int | None = None,
    partition_mode: str = "baseline",
) -> ApproximateSufficiencySummary:
    """Support-versus-sufficiency diagnostics for a coarsened comparison map.

    The theorem-relevant scalar ``max_kappa`` is the worst-case contamination
    radius induced by the operative partition. It is computed as one half of the
    within-bin oscillation of the theorem-relevant benchmark term across the
    verified current-cell fine groups merged by the coarsening, corresponding to the
    measurable midpoint benchmark used in the theorem.
    """

    groups = np.asarray(state_groups, dtype=int)
    matching = build_matching_structure(calibration, solution, mu, state_groups=groups)
    overlap_share = matching.diagnostics.overall_support_overlap_share

    actual_bin_count = calibration.envelope_bin_count if bin_count is None else max(1, int(bin_count))
    partition_label = _partition_label(partition_mode, actual_bin_count)

    fine_groups = natural_state_groups(calibration)
    if np.array_equal(groups, fine_groups):
        return ApproximateSufficiencySummary(
            label=label,
            state_groups=groups,
            overall_support_overlap_share=overlap_share,
            partition_label=partition_label,
            bin_count=actual_bin_count,
            weighted_mean_oscillation=0.0,
            weighted_p95_oscillation=0.0,
            weighted_p99_oscillation=0.0,
            max_oscillation=0.0,
            weighted_mean_kappa=0.0,
            weighted_p95_kappa=0.0,
            weighted_p99_kappa=0.0,
            max_kappa=0.0,
        )

    rate = calibration.interest_rate if social_rate is None else float(social_rate)
    benchmark = required_eta_ratio(calibration, implemented_wedge(calibration, solution), rate)
    log_upsilon = np.log(np.maximum(solution.marginal_value, 1e-300))
    younger_mass = np.asarray(mu[:-1, :, :], dtype=float)
    Jm1, B, E = benchmark.shape
    oscillation_values: list[float] = []
    kappa_values: list[float] = []
    summary_weights: list[float] = []

    age_fixed_cutoffs: dict[int, np.ndarray] = {}
    pooled_cutoffs = np.array([], dtype=float)
    if partition_mode == "fixed_age":
        for j in range(Jm1):
            age_mask = np.isfinite(log_upsilon[j]) & np.isfinite(benchmark[j]) & (younger_mass[j] > 0.0)
            if np.any(age_mask):
                age_fixed_cutoffs[j] = _weighted_bin_cutoffs(
                    log_upsilon[j][age_mask],
                    younger_mass[j][age_mask],
                    actual_bin_count,
                    shifted=False,
                )
            else:
                age_fixed_cutoffs[j] = np.array([], dtype=float)
    elif partition_mode == "pooled":
        pooled_mask = np.isfinite(log_upsilon[:-1]) & np.isfinite(benchmark) & (younger_mass > 0.0)
        if np.any(pooled_mask):
            pooled_cutoffs = _weighted_bin_cutoffs(
                log_upsilon[:-1][pooled_mask],
                younger_mass[pooled_mask],
                actual_bin_count,
                shifted=False,
            )

    for j in range(Jm1):
        for coarse_group in np.unique(groups):
            merged_states = np.where(groups == coarse_group)[0]
            merged_mask = np.zeros((B, E), dtype=bool)
            merged_mask[:, merged_states] = True
            merged_mask &= np.isfinite(log_upsilon[j]) & np.isfinite(benchmark[j]) & (younger_mass[j] > 0.0)
            if not np.any(merged_mask):
                continue

            values = log_upsilon[j][merged_mask]
            weights = younger_mass[j][merged_mask]
            if partition_mode == "baseline":
                cutoffs = _weighted_bin_cutoffs(values, weights, actual_bin_count, shifted=False)
            elif partition_mode == "shifted":
                cutoffs = _weighted_bin_cutoffs(values, weights, actual_bin_count, shifted=True)
            elif partition_mode == "fixed_age":
                cutoffs = age_fixed_cutoffs[j]
            elif partition_mode == "pooled":
                cutoffs = pooled_cutoffs
            else:
                raise ValueError(f"Unknown partition_mode: {partition_mode}")

            bin_index, bin_total = _assign_partition_bins(log_upsilon[j], merged_mask, cutoffs)
            for q in range(bin_total):
                fine_means: list[float] = []
                bin_mass = 0.0
                for e in merged_states:
                    cell_mask = merged_mask[:, e] & (bin_index[:, e] == q)
                    if not np.any(cell_mask):
                        continue
                    cell_weights = younger_mass[j][:, e][cell_mask]
                    log_h = np.log(np.maximum(benchmark[j][:, e][cell_mask], 1e-300))
                    fine_means.append(float(np.sum(log_h * cell_weights) / np.sum(cell_weights)))
                    bin_mass += float(np.sum(cell_weights))
                if len(fine_means) <= 1 or bin_mass <= 0.0:
                    continue
                oscillation = float(max(fine_means) - min(fine_means))
                oscillation_values.append(oscillation)
                kappa_values.append(0.5 * oscillation)
                summary_weights.append(bin_mass)

    if not kappa_values:
        return ApproximateSufficiencySummary(
            label=label,
            state_groups=groups,
            overall_support_overlap_share=overlap_share,
            partition_label=partition_label,
            bin_count=actual_bin_count,
            weighted_mean_oscillation=0.0,
            weighted_p95_oscillation=0.0,
            weighted_p99_oscillation=0.0,
            max_oscillation=0.0,
            weighted_mean_kappa=0.0,
            weighted_p95_kappa=0.0,
            weighted_p99_kappa=0.0,
            max_kappa=0.0,
        )

    oscillation_array = np.asarray(oscillation_values, dtype=float)
    kappa_array = np.asarray(kappa_values, dtype=float)
    weight_array = np.asarray(summary_weights, dtype=float)
    mean_oscillation, p95_oscillation, p99_oscillation = _weighted_spread_summary(oscillation_array, weight_array)
    mean_kappa, p95_kappa, p99_kappa = _weighted_spread_summary(kappa_array, weight_array)
    return ApproximateSufficiencySummary(
        label=label,
        state_groups=groups,
        overall_support_overlap_share=overlap_share,
        partition_label=partition_label,
        bin_count=actual_bin_count,
        weighted_mean_oscillation=mean_oscillation,
        weighted_p95_oscillation=p95_oscillation,
        weighted_p99_oscillation=p99_oscillation,
        max_oscillation=float(np.max(oscillation_array)),
        weighted_mean_kappa=mean_kappa,
        weighted_p95_kappa=p95_kappa,
        weighted_p99_kappa=p99_kappa,
        max_kappa=float(np.max(kappa_array)),
    )



def partition_audit_summaries(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    *,
    social_rate: float | None = None,
    baseline_bin_count: int | None = None,
) -> list[ApproximateSufficiencySummary]:
    """Partition-robust contamination audit for coarsened comparison maps."""

    actual_bin_count = calibration.envelope_bin_count if baseline_bin_count is None else max(1, int(baseline_bin_count))
    lower_nested = max(5, actual_bin_count // 2)
    upper_nested = max(actual_bin_count + 1, 2 * actual_bin_count)
    audit_specs = [
        ("baseline", actual_bin_count),
        ("baseline", lower_nested),
        ("baseline", upper_nested),
        ("shifted", actual_bin_count),
        ("fixed_age", actual_bin_count),
        ("pooled", actual_bin_count),
    ]
    coarsened_specs = [
        (group_label, state_groups)
        for group_label, state_groups in support_sufficiency_group_specs(calibration)
        if not np.array_equal(np.asarray(state_groups, dtype=int), natural_state_groups(calibration))
    ]

    rows: list[ApproximateSufficiencySummary] = []
    for group_label, state_groups in coarsened_specs:
        for partition_mode, this_bin_count in audit_specs:
            summary = approximate_sufficiency_summary(
                calibration,
                solution,
                mu,
                state_groups=np.asarray(state_groups, dtype=int),
                label=group_label,
                social_rate=social_rate,
                bin_count=this_bin_count,
                partition_mode=partition_mode,
            )
            if partition_mode == "baseline" and this_bin_count == actual_bin_count:
                summary.partition_label = f"Baseline operative partition ({summary.partition_label})"
            elif partition_mode == "baseline":
                summary.partition_label = f"Nested-bin robustness ({summary.partition_label})"
            elif partition_mode == "shifted":
                summary.partition_label = f"Shifted-bin robustness ({summary.partition_label})"
            elif partition_mode == "fixed_age":
                summary.partition_label = f"Fixed-cutoff robustness ({summary.partition_label})"
            elif partition_mode == "pooled":
                summary.partition_label = f"Pooled-cutoff robustness ({summary.partition_label})"
            rows.append(summary)
    return rows


def k_nearest_candidate_correspondence(
    solution: HouseholdSolution,
    state_groups: np.ndarray,
    candidate_count: int = 5,
    unit_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the K nearest older candidates within each comparison class in the private metric."""

    log_upsilon = _log_unit_values(solution, unit_values)
    J, B, E = log_upsilon.shape
    groups = np.asarray(state_groups, dtype=int)
    K = int(candidate_count)
    older_asset = np.full((J - 1, B, E, K), -1, dtype=int)
    older_state = np.full((J - 1, B, E, K), -1, dtype=int)
    abs_gap = np.full((J - 1, B, E, K), np.nan, dtype=float)

    for j in range(J - 1):
        for g in np.unique(groups):
            states = np.where(groups == g)[0]
            older_values = []
            older_assets = []
            older_states = []
            for e_old in states:
                older_values.append(log_upsilon[j + 1, :, e_old])
                older_assets.append(np.arange(B, dtype=int))
                older_states.append(np.full(B, e_old, dtype=int))
            older_values_arr = np.concatenate(older_values)
            older_assets_arr = np.concatenate(older_assets)
            older_states_arr = np.concatenate(older_states)

            for e_y in states:
                diffs = np.abs(log_upsilon[j, :, e_y][:, None] - older_values_arr[None, :])
                if diffs.shape[1] <= K:
                    idx = np.argsort(diffs, axis=1)
                else:
                    idx = np.argpartition(diffs, kth=K - 1, axis=1)[:, :K]
                    idx = np.take_along_axis(idx, np.argsort(np.take_along_axis(diffs, idx, axis=1), axis=1), axis=1)
                take_k = idx.shape[1]
                older_asset[j, :, e_y, :take_k] = older_assets_arr[idx]
                older_state[j, :, e_y, :take_k] = older_states_arr[idx]
                abs_gap[j, :, e_y, :take_k] = np.take_along_axis(diffs, idx, axis=1)

    return older_asset, older_state, abs_gap


def graph_robust_eta_envelope(
    required_ratio: np.ndarray,
    older_asset_candidates: np.ndarray,
    older_state_candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper maintained envelopes over an admissible one-step correspondence."""

    Jm1, B, E = required_ratio.shape
    candidate_count = older_asset_candidates.shape[-1]
    J = Jm1 + 1
    lower = np.ones((J, B, E), dtype=float)
    upper = np.ones((J, B, E), dtype=float)

    for j in range(J - 2, -1, -1):
        safe_ratio = np.where(np.isfinite(required_ratio[j]), np.maximum(required_ratio[j], 1e-12), 1.0)
        for e in range(E):
            for i in range(B):
                future_lower = []
                future_upper = []
                for k in range(candidate_count):
                    asset_idx = int(older_asset_candidates[j, i, e, k])
                    state_idx = int(older_state_candidates[j, i, e, k])
                    if asset_idx < 0 or state_idx < 0:
                        continue
                    future_lower.append(lower[j + 1, asset_idx, state_idx])
                    future_upper.append(upper[j + 1, asset_idx, state_idx])
                if future_lower:
                    next_lower = float(np.min(future_lower))
                    next_upper = float(np.max(future_upper))
                else:
                    next_lower = 1.0
                    next_upper = 1.0
                lower[j, i, e] = safe_ratio[i, e] * next_lower
                upper[j, i, e] = safe_ratio[i, e] * next_upper
    return lower, upper


def graph_envelope_summary(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    state_groups: np.ndarray,
    label: str,
    social_rate: float | None = None,
    candidate_count: int = 5,
) -> GraphEnvelopeSummary:
    """Summarize the maintained graph-robust envelope by age."""

    rate = calibration.interest_rate if social_rate is None else float(social_rate)
    ratio = required_eta_ratio(calibration, implemented_wedge(calibration, solution), rate)
    matching = build_matching_structure(calibration, solution, mu, state_groups=np.asarray(state_groups, dtype=int))
    point_eta = reconstruct_eta_profile(ratio, matching.older_asset_index, matching.older_state_index)
    candidate_assets, candidate_states, _ = k_nearest_candidate_correspondence(
        solution,
        np.asarray(state_groups, dtype=int),
        candidate_count=candidate_count,
    )
    lower_eta, upper_eta = graph_robust_eta_envelope(ratio, candidate_assets, candidate_states)
    return GraphEnvelopeSummary(
        label=label,
        ages=np.asarray(calibration.ages, dtype=int),
        point_age_average=age_averaged_eta(point_eta, mu),
        lower_age_average=age_averaged_eta(lower_eta, mu),
        upper_age_average=age_averaged_eta(upper_eta, mu),
    )



def _extend_support_mask(mask: np.ndarray | None, horizon: int) -> np.ndarray | None:
    """Extend a younger-cell support mask to the full age horizon if needed."""

    if mask is None:
        return None
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape[0] == horizon:
        return mask_arr
    if mask_arr.shape[0] == horizon - 1:
        extension = np.zeros((1,) + mask_arr.shape[1:], dtype=bool)
        return np.concatenate([mask_arr, extension], axis=0)
    raise ValueError("support mask has incompatible age dimension")


def _policy_selector(
    shape: tuple[int, int, int],
    calibration: Calibration,
    solution: HouseholdSolution,
    age_bounds: tuple[int, int],
    state_indices: list[int] | tuple[int, ...] | None = None,
    asset_quantile: str = "all",
) -> np.ndarray:
    """Boolean selector for a policy target group."""

    selector = np.zeros(shape, dtype=bool)
    ages = np.asarray(calibration.ages, dtype=int)
    age_mask = (ages >= age_bounds[0]) & (ages <= age_bounds[1])
    selector[age_mask, :, :] = True

    if state_indices is not None:
        state_mask = np.zeros(shape[2], dtype=bool)
        state_mask[np.asarray(state_indices, dtype=int)] = True
        selector &= state_mask[None, None, :]

    if asset_quantile != "all":
        q1, q2 = np.quantile(solution.asset_grid, [1.0 / 3.0, 2.0 / 3.0])
        if asset_quantile == "low":
            asset_idx = np.where(solution.asset_grid <= q1)[0]
        elif asset_quantile == "mid":
            asset_idx = np.where((solution.asset_grid > q1) & (solution.asset_grid <= q2))[0]
        elif asset_quantile == "high":
            asset_idx = np.where(solution.asset_grid > q2)[0]
        else:
            raise ValueError(f"Unknown asset quantile: {asset_quantile}")
        asset_mask = np.zeros(shape[1], dtype=bool)
        asset_mask[asset_idx] = True
        selector &= asset_mask[None, :, None]

    return selector


def policy_target_union_support_share(
    mu: np.ndarray,
    calibration: Calibration,
    solution: HouseholdSolution,
    support_mask: np.ndarray | None,
    target_a_age_bounds: tuple[int, int] = (20, 29),
    target_a_state_indices: tuple[int, ...] = (2,),
    target_a_asset_quantile: str = "all",
    target_b_age_bounds: tuple[int, int] = (40, 49),
    target_b_state_indices: tuple[int, ...] | None = None,
    target_b_asset_quantile: str = "all",
) -> float:
    """Mass share of the union of policy targets retained by a support restriction."""

    selector_a = _policy_selector(
        mu.shape,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
    )
    selector_b = _policy_selector(
        mu.shape,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
    )
    union_selector = selector_a | selector_b
    total_mass = float(np.sum(np.where(union_selector, mu, 0.0)))
    if total_mass <= 0.0:
        return 0.0
    support = _extend_support_mask(support_mask, mu.shape[0])
    if support is None:
        return 1.0
    retained_mass = float(np.sum(np.where(union_selector & support, mu, 0.0)))
    return retained_mass / total_mass


def policy_target_support_share(
    mu: np.ndarray,
    calibration: Calibration,
    solution: HouseholdSolution,
    support_mask: np.ndarray | None,
    age_bounds: tuple[int, int],
    state_indices: tuple[int, ...] | None = None,
    asset_quantile: str = "all",
) -> float:
    """Mass share of one target group retained by a support restriction."""

    selector = _policy_selector(
        mu.shape,
        calibration,
        solution,
        age_bounds,
        state_indices,
        asset_quantile,
    )
    total_mass = float(np.sum(np.where(selector, mu, 0.0)))
    if total_mass <= 0.0:
        return 0.0
    support = _extend_support_mask(support_mask, mu.shape[0])
    if support is None:
        return 1.0
    retained_mass = float(np.sum(np.where(selector & support, mu, 0.0)))
    return retained_mass / total_mass


def policy_log_ratio(score_a: float, score_b: float) -> float:
    """Scale-free target comparison statistic based on a log ratio."""

    if not np.isfinite(score_a) or not np.isfinite(score_b) or score_a <= 0.0 or score_b <= 0.0:
        return float("nan")
    return float(np.log(score_a / score_b))


def policy_target_score(
    eta: np.ndarray,
    mu: np.ndarray,
    calibration: Calibration,
    solution: HouseholdSolution,
    age_bounds: tuple[int, int],
    state_indices: list[int] | tuple[int, ...] | None = None,
    asset_quantile: str = "all",
    support_mask: np.ndarray | None = None,
) -> float:
    """Mass-weighted average normalized multiplier for a target group."""

    selector = _policy_selector(
        eta.shape,
        calibration,
        solution,
        age_bounds,
        state_indices,
        asset_quantile,
    )
    support = _extend_support_mask(support_mask, eta.shape[0])
    if support is not None:
        selector &= support

    weights = np.where(selector, mu, 0.0)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return float("nan")
    return float(np.sum(eta * weights) / total_weight)


def graph_topology_summary(
    mu: np.ndarray,
    calibration: Calibration,
    solution: HouseholdSolution,
    support_mask: np.ndarray | None,
    older_asset_candidates: np.ndarray,
    older_state_candidates: np.ndarray,
    target_a_age_bounds: tuple[int, int] = (20, 29),
    target_a_state_indices: tuple[int, ...] = (2,),
    target_a_asset_quantile: str = "all",
    target_b_age_bounds: tuple[int, int] = (40, 49),
    target_b_state_indices: tuple[int, ...] | None = None,
    target_b_asset_quantile: str = "all",
) -> GraphTopologySummary:
    """Count finite edges and connected components for a maintained K-candidate graph."""

    J, B, E = mu.shape
    support = _extend_support_mask(support_mask, J)
    if support is None:
        support = np.ones((J, B, E), dtype=bool)
    active_y = support[:-1] & (mu[:-1] > 0.0)
    shape = (J, B, E)

    parent: dict[int, int] = {}

    def add(x: int) -> None:
        if x not in parent:
            parent[x] = x

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        add(x)
        add(y)
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    edge_count = 0
    K = older_asset_candidates.shape[-1]
    ys = np.argwhere(active_y)
    for j, i, e in ys:
        src = int(np.ravel_multi_index((int(j), int(i), int(e)), shape))
        add(src)
        for k in range(K):
            asset_idx = int(older_asset_candidates[j, i, e, k])
            state_idx = int(older_state_candidates[j, i, e, k])
            if asset_idx < 0 or state_idx < 0:
                continue
            dst = int(np.ravel_multi_index((int(j) + 1, asset_idx, state_idx), shape))
            union(src, dst)
            edge_count += 1

    selector_a = _policy_selector(
        shape,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
    )
    selector_b = _policy_selector(
        shape,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
    )
    target_selector = (selector_a | selector_b) & support & (mu > 0.0)
    target_nodes = []
    for j, i, e in np.argwhere(target_selector):
        node = int(np.ravel_multi_index((int(j), int(i), int(e)), shape))
        add(node)
        target_nodes.append(node)

    components = {find(node) for node in parent}
    target_components = {find(node) for node in target_nodes}
    return GraphTopologySummary(
        edge_count=int(edge_count),
        node_count=int(len(parent)),
        component_count=int(len(components)),
        target_component_count=int(len(target_components)),
    )




def policy_ranking_summary(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    state_groups: np.ndarray,
    label: str,
    support_label: str,
    support_share: float,
    kappa_mean: float,
    kappa_max: float,
    social_rate: float | None = None,
    candidate_count: int = 5,
    support_mask: np.ndarray | None = None,
    target_a_age_bounds: tuple[int, int] = (20, 29),
    target_a_state_indices: tuple[int, ...] = (2,),
    target_a_asset_quantile: str = "all",
    target_b_age_bounds: tuple[int, int] = (40, 49),
    target_b_state_indices: tuple[int, ...] | None = None,
    target_b_asset_quantile: str = "all",
) -> PolicyRankingSummary:
    """Scale-free point and graph-robust ranking for a PVEU-balanced local target comparison."""

    rate = calibration.interest_rate if social_rate is None else float(social_rate)
    groups = np.asarray(state_groups, dtype=int)
    ratio = required_eta_ratio(calibration, implemented_wedge(calibration, solution), rate)
    matching = build_matching_structure(calibration, solution, mu, state_groups=groups)
    point_eta = reconstruct_eta_profile(ratio, matching.older_asset_index, matching.older_state_index)
    candidate_assets, candidate_states, _ = k_nearest_candidate_correspondence(solution, groups, candidate_count=candidate_count)
    lower_eta, upper_eta = graph_robust_eta_envelope(ratio, candidate_assets, candidate_states)

    target_a_support_share = policy_target_support_share(
        mu,
        calibration,
        solution,
        support_mask,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
    )
    target_b_support_share = policy_target_support_share(
        mu,
        calibration,
        solution,
        support_mask,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
    )

    point_a = policy_target_score(
        point_eta,
        mu,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
        support_mask=support_mask,
    )
    point_b = policy_target_score(
        point_eta,
        mu,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
        support_mask=support_mask,
    )
    lower_a = policy_target_score(
        lower_eta,
        mu,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
        support_mask=support_mask,
    )
    upper_a = policy_target_score(
        upper_eta,
        mu,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
        support_mask=support_mask,
    )
    lower_b = policy_target_score(
        lower_eta,
        mu,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
        support_mask=support_mask,
    )
    upper_b = policy_target_score(
        upper_eta,
        mu,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
        support_mask=support_mask,
    )

    return PolicyRankingSummary(
        label=label,
        support_label=support_label,
        support_share=support_share,
        target_a_support_share=target_a_support_share,
        target_b_support_share=target_b_support_share,
        kappa_mean=kappa_mean,
        kappa_max=kappa_max,
        target_a_score=point_a,
        target_b_score=point_b,
        log_ratio=policy_log_ratio(point_a, point_b),
        lower_log_ratio=policy_log_ratio(lower_a, upper_b),
        upper_log_ratio=policy_log_ratio(upper_a, lower_b),
    )



def alternative_normalization_summary(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    state_groups: np.ndarray,
    label: str,
    support_label: str,
    support_share: float,
    social_rate: float | None = None,
    candidate_count: int = 5,
    support_mask: np.ndarray | None = None,
    target_a_age_bounds: tuple[int, int] = (20, 29),
    target_a_state_indices: tuple[int, ...] = (2,),
    target_a_asset_quantile: str = "all",
    target_b_age_bounds: tuple[int, int] = (40, 49),
    target_b_state_indices: tuple[int, ...] | None = None,
    target_b_asset_quantile: str = "all",
) -> AlternativeNormalizationSummary:
    """Recompute policy rankings under a reference-slope private unit.

    Baseline columns use the maintained current-goods private metric.  The
    alternative columns use v_tilde = v / v_bar_{a,k}, rebuild one-step support,
    nearest-neighbor correspondences, graph envelopes, and component counts, and
    then evaluate the same target pair on the resulting support.
    """

    groups = np.asarray(state_groups, dtype=int)
    rate = calibration.interest_rate if social_rate is None else float(social_rate)
    ratio = required_eta_ratio(calibration, implemented_wedge(calibration, solution), rate)

    baseline_matching = build_matching_structure(calibration, solution, mu, state_groups=groups)
    baseline_point_eta = reconstruct_eta_profile(
        ratio,
        baseline_matching.older_asset_index,
        baseline_matching.older_state_index,
    )
    baseline_candidate_assets, baseline_candidate_states, _ = k_nearest_candidate_correspondence(
        solution,
        groups,
        candidate_count=candidate_count,
    )
    baseline_lower_eta, baseline_upper_eta = graph_robust_eta_envelope(
        ratio,
        baseline_candidate_assets,
        baseline_candidate_states,
    )

    alternative_unit_values = age_state_rescaled_upsilon(solution, mu)
    alternative_support_mask = support_overlap_mask(
        solution,
        mu,
        groups,
        unit_values=alternative_unit_values,
    )
    alternative_support_share = policy_target_union_support_share(
        mu,
        calibration,
        solution,
        alternative_support_mask,
        target_a_age_bounds=target_a_age_bounds,
        target_a_state_indices=target_a_state_indices,
        target_a_asset_quantile=target_a_asset_quantile,
        target_b_age_bounds=target_b_age_bounds,
        target_b_state_indices=target_b_state_indices,
        target_b_asset_quantile=target_b_asset_quantile,
    )
    alternative_matching = build_matching_structure(
        calibration,
        solution,
        mu,
        state_groups=groups,
        unit_values=alternative_unit_values,
    )
    alternative_point_eta = reconstruct_eta_profile(
        ratio,
        alternative_matching.older_asset_index,
        alternative_matching.older_state_index,
    )
    alternative_candidate_assets, alternative_candidate_states, _ = k_nearest_candidate_correspondence(
        solution,
        groups,
        candidate_count=candidate_count,
        unit_values=alternative_unit_values,
    )
    alternative_lower_eta, alternative_upper_eta = graph_robust_eta_envelope(
        ratio,
        alternative_candidate_assets,
        alternative_candidate_states,
    )

    target_a_support_share = policy_target_support_share(
        mu,
        calibration,
        solution,
        support_mask,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
    )
    target_b_support_share = policy_target_support_share(
        mu,
        calibration,
        solution,
        support_mask,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
    )
    alternative_target_a_support_share = policy_target_support_share(
        mu,
        calibration,
        solution,
        alternative_support_mask,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
    )
    alternative_target_b_support_share = policy_target_support_share(
        mu,
        calibration,
        solution,
        alternative_support_mask,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
    )

    baseline_a = policy_target_score(
        baseline_point_eta,
        mu,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
        support_mask=support_mask,
    )
    baseline_b = policy_target_score(
        baseline_point_eta,
        mu,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
        support_mask=support_mask,
    )
    baseline_lower_a = policy_target_score(
        baseline_lower_eta,
        mu,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
        support_mask=support_mask,
    )
    baseline_upper_a = policy_target_score(
        baseline_upper_eta,
        mu,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
        support_mask=support_mask,
    )
    baseline_lower_b = policy_target_score(
        baseline_lower_eta,
        mu,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
        support_mask=support_mask,
    )
    baseline_upper_b = policy_target_score(
        baseline_upper_eta,
        mu,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
        support_mask=support_mask,
    )

    alternative_a = policy_target_score(
        alternative_point_eta,
        mu,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
        support_mask=alternative_support_mask,
    )
    alternative_b = policy_target_score(
        alternative_point_eta,
        mu,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
        support_mask=alternative_support_mask,
    )
    alternative_lower_a = policy_target_score(
        alternative_lower_eta,
        mu,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
        support_mask=alternative_support_mask,
    )
    alternative_upper_a = policy_target_score(
        alternative_upper_eta,
        mu,
        calibration,
        solution,
        target_a_age_bounds,
        target_a_state_indices,
        target_a_asset_quantile,
        support_mask=alternative_support_mask,
    )
    alternative_lower_b = policy_target_score(
        alternative_lower_eta,
        mu,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
        support_mask=alternative_support_mask,
    )
    alternative_upper_b = policy_target_score(
        alternative_upper_eta,
        mu,
        calibration,
        solution,
        target_b_age_bounds,
        target_b_state_indices,
        target_b_asset_quantile,
        support_mask=alternative_support_mask,
    )

    baseline_topology = graph_topology_summary(
        mu,
        calibration,
        solution,
        support_mask,
        baseline_candidate_assets,
        baseline_candidate_states,
        target_a_age_bounds=target_a_age_bounds,
        target_a_state_indices=target_a_state_indices,
        target_a_asset_quantile=target_a_asset_quantile,
        target_b_age_bounds=target_b_age_bounds,
        target_b_state_indices=target_b_state_indices,
        target_b_asset_quantile=target_b_asset_quantile,
    )
    alternative_topology = graph_topology_summary(
        mu,
        calibration,
        solution,
        alternative_support_mask,
        alternative_candidate_assets,
        alternative_candidate_states,
        target_a_age_bounds=target_a_age_bounds,
        target_a_state_indices=target_a_state_indices,
        target_a_asset_quantile=target_a_asset_quantile,
        target_b_age_bounds=target_b_age_bounds,
        target_b_state_indices=target_b_state_indices,
        target_b_asset_quantile=target_b_asset_quantile,
    )

    return AlternativeNormalizationSummary(
        label=label,
        support_label=support_label,
        support_share=support_share,
        target_a_support_share=target_a_support_share,
        target_b_support_share=target_b_support_share,
        alternative_support_label="Recomputed support",
        alternative_support_share=alternative_support_share,
        alternative_target_a_support_share=alternative_target_a_support_share,
        alternative_target_b_support_share=alternative_target_b_support_share,
        baseline_edge_count=baseline_topology.edge_count,
        baseline_component_count=baseline_topology.component_count,
        alternative_edge_count=alternative_topology.edge_count,
        alternative_component_count=alternative_topology.component_count,
        baseline_log_ratio=policy_log_ratio(baseline_a, baseline_b),
        baseline_lower_log_ratio=policy_log_ratio(baseline_lower_a, baseline_upper_b),
        baseline_upper_log_ratio=policy_log_ratio(baseline_upper_a, baseline_lower_b),
        alternative_log_ratio=policy_log_ratio(alternative_a, alternative_b),
        alternative_lower_log_ratio=policy_log_ratio(alternative_lower_a, alternative_upper_b),
        alternative_upper_log_ratio=policy_log_ratio(alternative_upper_a, alternative_lower_b),
    )



def policy_decomposition_summary(
    age_only_full: PolicyRankingSummary,
    age_only_common: PolicyRankingSummary,
    natural_common: PolicyRankingSummary,
) -> PolicyDecompositionSummary:
    """Decompose the shift from age-only full support to natural matching on common support."""

    support_exclusion = age_only_common.log_ratio - age_only_full.log_ratio
    within_support_rematching = natural_common.log_ratio - age_only_common.log_ratio
    total_change = natural_common.log_ratio - age_only_full.log_ratio
    return PolicyDecompositionSummary(
        support_exclusion=support_exclusion,
        within_support_rematching=within_support_rematching,
        total_change=total_change,
    )


def kappa_sensitivity_summaries(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    bin_counts: tuple[int, ...] = (10, 20, 40),
) -> list[KappaSensitivitySummary]:
    """Sensitivity of contamination diagnostics to the number of aggregation bins."""

    rows: list[KappaSensitivitySummary] = []
    for bin_count in bin_counts:
        for label, state_groups in support_sufficiency_group_specs(calibration):
            summary = approximate_sufficiency_summary(
                calibration,
                solution,
                mu,
                state_groups=state_groups,
                label=label,
                bin_count=bin_count,
            )
            rows.append(
                KappaSensitivitySummary(
                    label=label,
                    bin_count=bin_count,
                    weighted_mean_kappa=summary.weighted_mean_kappa,
                    max_kappa=summary.max_kappa,
                )
            )
    return rows


def envelope_width_sensitivity_summaries(
    calibration: Calibration,
    solution: HouseholdSolution,
    mu: np.ndarray,
    candidate_counts: tuple[int, ...] = (1, 5, 10),
    social_rate: float | None = None,
) -> list[EnvelopeWidthSensitivitySummary]:
    """Sensitivity of age-profile envelope width to the selector-count parameter K."""

    rows: list[EnvelopeWidthSensitivitySummary] = []
    for label, state_groups in [
        ("Verified current-cell map", natural_state_groups(calibration)),
        ("Age-only current matching", pooled_state_groups(calibration)),
    ]:
        for candidate_count in candidate_counts:
            summary = graph_envelope_summary(
                calibration,
                solution,
                mu,
                state_groups=state_groups,
                label=label,
                social_rate=social_rate,
                candidate_count=candidate_count,
            )
            width = np.maximum(summary.upper_age_average - summary.lower_age_average, 0.0)
            rows.append(
                EnvelopeWidthSensitivitySummary(
                    label=label,
                    candidate_count=candidate_count,
                    mean_width=float(np.mean(width)),
                    max_width=float(np.max(width)),
                )
            )
    return rows
