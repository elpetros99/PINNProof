"""High-level verification report helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .residuals import swing_equation_residual


@dataclass(frozen=True)
class VerificationReport:
    max_abs_residual: float
    mean_abs_residual: float
    rms_residual: float
    passed: bool
    tolerance: float


def verify_swing_trajectories(
    delta: np.ndarray,
    omega: np.ndarray,
    times: np.ndarray,
    *,
    inertia: float,
    damping: float,
    coupling: float,
    mechanical_power: float,
    tolerance: float = 1e-2,
) -> VerificationReport:
    """Verify that trajectories satisfy the swing-equation residual budget."""
    residual = swing_equation_residual(
        delta,
        omega,
        times,
        inertia=inertia,
        damping=damping,
        coupling=coupling,
        mechanical_power=mechanical_power,
    )
    abs_res = np.abs(residual)
    max_abs = float(np.max(abs_res))
    mean_abs = float(np.mean(abs_res))
    rms = float(np.sqrt(np.mean(residual**2)))
    return VerificationReport(
        max_abs_residual=max_abs,
        mean_abs_residual=mean_abs,
        rms_residual=rms,
        passed=max_abs <= tolerance,
        tolerance=tolerance,
    )
