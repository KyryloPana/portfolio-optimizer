from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def sum_to_one_constraint() -> Dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def hhi_max_constraint(hhi_max: float) -> Dict:
    """
    Inequality constraint for SLSQP:
        fun(w) >= 0  =>  hhi_max - sum(w^2) >= 0  =>  sum(w^2) <= hhi_max
    """
    if not (0.0 < hhi_max <= 1.0):
        raise ValueError("hhi_max must be in (0, 1].")
    return {"type": "ineq", "fun": lambda w: float(hhi_max - np.sum(np.square(w)))}