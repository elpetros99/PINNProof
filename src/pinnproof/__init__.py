"""PINNProof: validation and verification toolbox for power-system surrogate models."""

from .validation.metrics import (
    mae,
    nrmse,
    rmse,
    trajectory_metrics,
)
from .verification.report import VerificationReport, verify_swing_trajectories

__all__ = [
    "rmse",
    "mae",
    "nrmse",
    "trajectory_metrics",
    "VerificationReport",
    "verify_swing_trajectories",
]
