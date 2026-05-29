"""Demographic inputs for the finite-life benchmark."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from .calibration import Calibration


def load_survival_table(path: Path) -> List[Dict[str, float]]:
    """Load the one-year survival schedule from CSV."""

    rows: List[Dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({k: float(v) if k != "age" else int(v) for k, v in row.items()})
    return rows


def survival_probabilities(calibration: Calibration) -> np.ndarray:
    """Return the survival probability from each age to the next."""

    table = load_survival_table(calibration.survival_csv)
    mapping = {int(row["age"]): row["p_survive_to_next"] for row in table}
    probs = np.array([mapping[age] for age in calibration.ages], dtype=float)
    probs[-1] = 0.0
    return probs


def stationary_age_weights(calibration: Calibration, survival: np.ndarray) -> np.ndarray:
    """Construct stationary age weights implied by survival and population growth."""

    weights = np.zeros(calibration.periods, dtype=float)
    weights[0] = 1.0
    growth = 1.0 + calibration.population_growth
    for j in range(1, calibration.periods):
        weights[j] = weights[j - 1] * survival[j - 1] / growth
    weights_sum = weights.sum()
    if weights_sum <= 0.0:
        raise ValueError("Age weights did not sum to a positive number.")
    return weights / weights_sum
