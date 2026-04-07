"""Quickstart: validate surrogate trajectories against reference trajectories."""

from pathlib import Path
import sys

import numpy as np


def _add_src_to_path() -> None:
    for base in Path(__file__).resolve().parents:
        src_dir = base / "src"
        if (src_dir / "pinnproof").is_dir():
            sys.path.insert(0, str(src_dir))
            return
    raise ModuleNotFoundError("Could not locate src/pinnproof for local example execution.")


_add_src_to_path()

from pinnproof.validation import trajectory_metrics

rng = np.random.default_rng(7)
n_traj, n_time, n_states = 5, 200, 2

reference = rng.normal(size=(n_traj, n_time, n_states))
surrogate = reference + 0.03 * rng.normal(size=(n_traj, n_time, n_states))

metrics = trajectory_metrics(reference, surrogate)
print("Global RMSE:", metrics["global_rmse"])
print("Per-state RMSE:", metrics["per_state_rmse"])
print("Per-trajectory RMSE:", metrics["per_trajectory_rmse"])
