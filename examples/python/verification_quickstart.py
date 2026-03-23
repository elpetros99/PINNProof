"""Quickstart: verify surrogate trajectories with swing-equation residual checks."""

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

from pinnproof.verification import verify_swing_trajectories

t = np.linspace(0.0, 2.0, 400)
delta = 0.4 * np.sin(2 * np.pi * t)
omega = np.gradient(delta, t)

report = verify_swing_trajectories(
    delta,
    omega,
    t,
    inertia=0.4,
    damping=0.1,
    coupling=0.2,
    mechanical_power=0.105,
    tolerance=0.6,
)

print(report)
