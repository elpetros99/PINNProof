
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from scipy.integrate import odeint
from torchdiffeq import odeint
import random
from pyDOE import lhs
import os

class solver():
    """
    Base class for ODE solvers.
    This class defines the interface for ODE solvers.
    Subclasses must implement the solve() method.
    """
    # def __init__(self, func, ini_cond, t_final, num_points, *args, **kwargs):

    def __init__(self, func, *args, **kwargs):

        """
        Initialize the ODE solver with the function, initial conditions,
        final time, number of points, and any additional arguments.
        
        Parameters:
            func       : callable f(t, y, *args, **kwargs) returning dy/dt

            ini_cond   : list or array of initial values [y1(0), y2(0), ..., yN(0)]
            t_final    : final time up to which you want the solution
            num_points : number of time points between 0 and t_final
=======
            args       : any extra positional arguments needed by func
            kwargs     : any extra keyword arguments needed by func
        """
        self.func = func
        # self.ini_cond = ini_cond
        # self.t_final = t_final
        # self.num_points = num_points
        self.args = args
        self.kwargs = kwargs
    
    def generate_dataset(
        self,
        ic_ranges: dict,
        num_ic: int,
        t_final: float,
        num_points: int,
        sampling: str = "random",
        device: str = "cpu",
        save_path: str = None
    ):
        """
        Builds a dataset by solving your ODE for multiple initial conditions.

        Args:
        ic_ranges  : dict mapping state‐names → (low, high) tuples, e.g.
                    {"delta": (0.0, 0.2), "omega": (-0.1, 0.1)}
        num_ic     : how many initial‐condition samples to draw
        t_final    : end time for each trajectory
        num_points : number of time‐steps per trajectory
        sampling   : one of {"grid", "random", "lhs"}:
                    - grid: full Cartesian grid if num_states small and num_ic = ∏n_i
                    - random: uniform random
                    - lhs: Latin‐hypercube 
        device     : torch device for integration
        save_path  : if not None, path where dataset is saved (.pt or .npz)
                    
        Returns:
        t     : (num_points,) torch.Tensor
        data  : (num_ic, num_points, num_states) torch.Tensor
        ics   : (num_ic, num_states) torch.Tensor
        """
        # 1) time grid
        t = torch.linspace(0.0, t_final, num_points, dtype=torch.float32, device=device)

        # 2) assemble ranges and state‐order
        state_names = list(ic_ranges.keys())
        lows  = np.array([ic_ranges[k][0] for k in state_names], dtype=float)
        highs = np.array([ic_ranges[k][1] for k in state_names], dtype=float)
        n_states = len(state_names)

        # 3) draw samples
        if sampling.lower() == "grid":
            # assume num_ic == product of desired points per dim
            grids = [np.linspace(lows[i], highs[i], int(round(num_ic**(1/n_states)))) 
                    for i in range(n_states)]
            mesh = np.meshgrid(*grids, indexing="ij")
            ics_np = np.stack([m.flatten() for m in mesh], axis=-1)
            ics_np = ics_np[:num_ic]  # trim if over‐abundant
        elif sampling.lower() == "lhs":
            unit = lhs(n_states, samples=num_ic)
            ics_np = lows + unit * (highs - lows)
        else:  # random
            ics_np = np.random.uniform(lows, highs, size=(num_ic, n_states))

        ics = torch.tensor(ics_np, dtype=torch.float32, device=device)

        # 4) solve each trajectory
        data = torch.zeros((num_ic, num_points, n_states), dtype=torch.float32, device=device)
        for i in range(num_ic):
            t_i, sol_i = self.solve(ics[i].cpu().numpy(), t_final, num_points)
            # sol_i: (num_points, n_states) torch.Tensor
            data[i] = sol_i.to(device)

        # 5) optionally save
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if save_path.endswith(".pt"):
                torch.save({"t": t, "ics": ics, "data": data}, save_path)
            else:
                np.savez(save_path, t=t.cpu().numpy(), ics=ics.cpu().numpy(), data=data.cpu().numpy())

        return t, data, ics

    def solve(self):
        """
        Solve the ODE from t=0 to t=t_final with initial conditions ini_cond.
    
        self.args = args
        self.kwargs = kwargs
        """
        raise NotImplementedError("Subclasses must implement this method.")
