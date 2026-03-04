"""Validation metrics for power-system surrogate trajectories."""

from __future__ import annotations

import numpy as np


ArrayLike = np.ndarray


def _as_numpy(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=float)


def rmse(y_true: ArrayLike, y_pred: ArrayLike, axis=None) -> np.ndarray:
    """Root mean squared error."""
    y_true_np = _as_numpy(y_true)
    y_pred_np = _as_numpy(y_pred)
    return np.sqrt(np.mean((y_pred_np - y_true_np) ** 2, axis=axis))


def mae(y_true: ArrayLike, y_pred: ArrayLike, axis=None) -> np.ndarray:
    """Mean absolute error."""
    y_true_np = _as_numpy(y_true)
    y_pred_np = _as_numpy(y_pred)
    return np.mean(np.abs(y_pred_np - y_true_np), axis=axis)


def nrmse(y_true: ArrayLike, y_pred: ArrayLike, axis=None, eps: float = 1e-12) -> np.ndarray:
    """Normalized RMSE using the true-signal range as denominator."""
    y_true_np = _as_numpy(y_true)
    denom = np.ptp(y_true_np, axis=axis) + eps
    return rmse(y_true_np, y_pred, axis=axis) / denom


def trajectory_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, np.ndarray]:
    """Per-state and aggregate error metrics for trajectory tensors.

    Parameters
    ----------
    y_true, y_pred:
        Arrays with shape ``(n_traj, n_time, n_states)``.
    """
    y_true_np = _as_numpy(y_true)
    y_pred_np = _as_numpy(y_pred)
    if y_true_np.shape != y_pred_np.shape:
        raise ValueError(f"Shape mismatch: {y_true_np.shape=} vs {y_pred_np.shape=}")

    per_state_rmse = rmse(y_true_np, y_pred_np, axis=(0, 1))
    per_state_mae = mae(y_true_np, y_pred_np, axis=(0, 1))
    per_traj_rmse = rmse(y_true_np, y_pred_np, axis=(1, 2))

    return {
        "global_rmse": rmse(y_true_np, y_pred_np),
        "global_mae": mae(y_true_np, y_pred_np),
        "per_state_rmse": per_state_rmse,
        "per_state_mae": per_state_mae,
        "per_trajectory_rmse": per_traj_rmse,
    }
