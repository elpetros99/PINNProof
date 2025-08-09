# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 14:48:54 2025

@author: INDRAJIT
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import odeint  # not used directly but kept for consistency
import torchode
import src.solvers.solver as solver

class GeneralTorchODESolver(solver):
    """
    A general-purpose wrapper around torchode's ODE solver.
    
    You supply:
      • func       : a callable f(t, y, *args, **kwargs) returning dy/dt as a torch.Tensor
      • ini_cond   : list or array of initial values [y1(0), y2(0), ..., yN(0)]
      • t_final    : final time up to which you want the solution
      • num_points : number of time points between 0 and t_final
      • args       : any extra positional arguments needed by func (e.g. physical constants)
      • kwargs     : any extra keyword arguments needed by func
    
    After initialization, calling solve() returns (t, Y):
      • t is a NumPy array of length num_points
      • Y is a NumPy array of shape (num_points, len(ini_cond)),
        where each column corresponds to one dependent variable over time.
    """

    def __init__(self, func, ini_cond, t_final, num_points, *args, **kwargs):
        # Save the user-provided right‐hand side function
        self.func = func

        # Initial conditions (convert to float)
        self.ini_cond = torch.tensor(ini_cond, dtype=torch.float32).unsqueeze(0)  
        #   shape: (1, N) for torchode (batch_dim=1, state_dim=N)

        # Time‐grid parameters
        self.t_final = float(t_final)
        self.num_points = int(num_points)

        # Store any extra parameters/args for func
        # We will convert these to torch.Tensor before solving
        self.raw_args = args
        self.raw_kwargs = kwargs

    def solve(self):
        """
        Solve dy/dt = func(t, y, *args, **kwargs)
        from t = 0 to t = t_final, with y(0) = ini_cond.
        Returns:
          t : np.ndarray, shape (num_points,)
          Y : np.ndarray, shape (num_points, len(ini_cond))
        """
        # 1) Convert all raw_args to torch.Tensor (float32) if they are scalars or arrays
        #    If an argument is already a torch.Tensor, leave it unchanged.
        torch_args = []
        for a in self.raw_args:
            if torch.is_tensor(a):
                torch_args.append(a)
            else:
                # assume numeric or array-like: convert to torch.Tensor
                torch_args.append(torch.tensor(a, dtype=torch.float32))
        torch_kwargs = {}
        for k, v in self.raw_kwargs.items():
            if torch.is_tensor(v):
                torch_kwargs[k] = v
            else:
                torch_kwargs[k] = torch.tensor(v, dtype=torch.float32)

        # 2) Prepare the time‐evaluation grid as a (1, num_points) Tensor
        t_eval = torch.linspace(0.0, self.t_final, self.num_points).unsqueeze(0)  
        #   shape: (batch=1, num_points)

        # 3) Define a wrapper f(t, y) that calls the user func with the converted args/kwargs
        def system_f(t, y):
            # y shape: (batch=1, state_dim=N)
            # t is a (batch=1,)-shaped Tensor for current time
            return self.func(t, y, *torch_args, **torch_kwargs)

        # 4) Build the ODETerm and choose integrator components
        ode_term = torchode.ODETerm(system_f)
        stepper = torchode.Tsit5(term=ode_term)
        controller = torchode.IntegralController(atol=1e-6, rtol=1e-3, term=ode_term)
        solver = torchode.AutoDiffAdjoint(stepper, controller)

        # Optionally JIT‐compile the solver for speed
        jit_solver = torch.compile(solver)

        # 5) Package the initial value problem
        ivp = torchode.InitialValueProblem(y0=self.ini_cond, t_eval=t_eval)

        # 6) Solve
        sol = jit_solver.solve(ivp)

        # Extract results
        # sol.ts has shape (1, num_points), sol.ys has shape (1, num_points, state_dim)
        t_tensor = sol.ts[0, :].detach().cpu()
        y_tensor = sol.ys[0, :, :].detach().cpu()  # shape: (num_points, N)

        # Convert to NumPy arrays
        t_np = t_tensor.numpy()
        y_np = y_tensor.numpy()

        return t_np, y_np


# ------------------------------------------------------------
# Example usage of GeneralTorchODESolver for the “swing equation”
# ------------------------------------------------------------

def swing_equation_torch(t, y, D1, B12_V1_V2, m1, P_gen):
    """
    Right‐hand side of the 2‐state swing equation in torch:
      y = [δ, ω], both shape (batch=1,) in torch
      Returns dy/dt = [dδ/dt, dω/dt] as shape (batch=1, 2)
    """
    # Unpack state variables
    δ = y[..., 0]
    ω = y[..., 1]

    # Compute derivatives
    dδ_dt = ω
    dω_dt = ( -D1 * ω - B12_V1_V2 * torch.sin(δ) + P_gen ) / m1

    return torch.stack([dδ_dt, dω_dt], dim=-1)


if __name__ == "__main__":
    # 1) Define initial conditions and parameters
    δ0 = 0.05          # initial rotor angle [rad]
    ω0 = 0.0           # initial rotor speed [rad/s]
    ini_cond = [δ0, ω0]

    t_final = 15.0     # simulate up to 15 seconds
    num_points = 500   # 500 time points

    # Swing‐equation parameters (scalars or arrays)
    D1        = 0.7
    B12_V1_V2 = 1.5
    m1        = 1.0
    P_gen     = 0.4

    # 2) Instantiate the solver, passing the extra parameters in the same order
    solver_torch = GeneralTorchODESolver(
        func       = swing_equation_torch,
        ini_cond   = ini_cond,
        t_final    = t_final,
        num_points = num_points,
        D1, B12_V1_V2, m1, P_gen
    )

    # 3) Solve
    t, y = solver_torch.solve()
    δ_vals = y[:, 0]  # shape (num_points,)
    ω_vals = y[:, 1]  # shape (num_points,)

    # 4) Plot the results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t, δ_vals, label='δ(t)')
    plt.xlabel('Time [s]')
    plt.ylabel('δ [rad]')
    plt.title('Rotor Angle vs. Time')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, ω_vals, color='orange', label='ω(t)')
    plt.xlabel('Time [s]')
    plt.ylabel('ω [rad/s]')
    plt.title('Rotor Speed vs. Time')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
