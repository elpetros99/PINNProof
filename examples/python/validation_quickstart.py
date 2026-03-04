"""Quickstart: validate surrogate trajectories against reference trajectories."""

import numpy as np

from pinnproof.validation import trajectory_metrics

rng = np.random.default_rng(7)
n_traj, n_time, n_states = 5, 200, 2

reference = rng.normal(size=(n_traj, n_time, n_states))
surrogate = reference + 0.03 * rng.normal(size=(n_traj, n_time, n_states))

metrics = trajectory_metrics(reference, surrogate)
print("Global RMSE:", metrics["global_rmse"])
print("Per-state RMSE:", metrics["per_state_rmse"])
print("Per-trajectory RMSE:", metrics["per_trajectory_rmse"])
