"""Quickstart: verify surrogate trajectories with swing-equation residual checks."""

import numpy as np

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
