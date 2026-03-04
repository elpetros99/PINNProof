"""Physics verification utilities (residual-based checks)."""

from __future__ import annotations

import numpy as np


ArrayLike = np.ndarray


def finite_difference(values: ArrayLike, times: ArrayLike) -> np.ndarray:
    """Compute d(values)/dt with second-order central differences."""
    values_np = np.asarray(values, dtype=float)
    times_np = np.asarray(times, dtype=float)
    return np.gradient(values_np, times_np, axis=-1, edge_order=2)


def swing_equation_residual(
    delta: ArrayLike,
    omega: ArrayLike,
    times: ArrayLike,
    *,
    inertia: float,
    damping: float,
    coupling: float,
    mechanical_power: float,
) -> np.ndarray:
    """Residual ``m*domega/dt + D*omega + B*sin(delta) - Pm``.

    Inputs can be shape ``(n_time,)`` or ``(n_traj, n_time)``.
    """
    delta_np = np.asarray(delta, dtype=float)
    omega_np = np.asarray(omega, dtype=float)
    domega_dt = finite_difference(omega_np, times)
    return inertia * domega_dt + damping * omega_np + coupling * np.sin(delta_np) - mechanical_power
