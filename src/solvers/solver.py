import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import random
from pyDOE import lhs
import os
import copy

class Solver:
    """
    Base class for ODE solvers. Subclasses must implement solve() and can use
    generate_dataset() and active_sample_initial() for data generation.
    """
    def __init__(self, func, interface_func, *args, **kwargs):
        """
        Initialize the ODE solver with:
          • func: the ODE function f(t, y, *args)
          • interface_func: post-processing on full solver output
          • args, kwargs: extra parameters passed to func
        """
        self.func = func
        self.interface_func = interface_func
        self.args = args
        self.kwargs = kwargs

    def generate_dataset(
        self,
        ic_ranges: torch.Tensor,
        num_ic: int,
        t_final: float,
        num_points: int,
        sampling: str = "random",
        device: str = "cpu",
        save_path: str = None
    ):
        """Build a dataset by solving the ODE for multiple ICs.

        Args:
            ic_ranges: Tensor[D,2] of [min,max] per state variable.
            num_ic:    total ICs to generate.
            t_final:   end time.
            num_points:points per trajectory.
            sampling:  'random' or 'active'.
            device:    'cpu' or 'cuda'.
            save_path: optional directory to save .pt files.

        Returns:
            ic_tensor:   [num_ic, D]
            traj_tensor: [num_ic, num_points, D]
        """
        D = ic_ranges.shape[0]
        ic_list, traj_list = [], []
        t_grid = torch.linspace(0.0, t_final, num_points, device=device)

        def solve_ic(x0: torch.Tensor):
            t, sol = self.solve(x0.to(device), t_final, num_points)
            return sol.detach().cpu()

        if sampling == "random":
            samples = lhs(D, samples=num_ic)
            mins, maxs = ic_ranges[:,0], ic_ranges[:,1]
            X0 = torch.from_numpy(samples).float() * (maxs - mins) + mins
            for x0 in X0:
                traj = solve_ic(x0)
                ic_list.append(x0.cpu())
                traj_list.append(traj)

        elif sampling == "active":
            bounds = ic_ranges.clone().float()
            for k in range(num_ic):
                if k == 0:
                    # first IC random
                    rand = torch.rand(D)
                    x0 = bounds[:,0] + (bounds[:,1]-bounds[:,0]) * rand
                    traj = solve_ic(x0)
                else:
                    collected = torch.stack(traj_list, dim=0)
                    x0, _, sol = self.active_sample_initial(
                        collected, bounds, t_final, num_points, device=device
                    )
                    traj = sol.cpu()
                ic_list.append(x0.cpu())
                traj_list.append(traj)
        else:
            raise ValueError(f"Unknown sampling: {sampling}")

        ic_tensor = torch.stack(ic_list, dim=0)
        traj_tensor = torch.stack(traj_list, dim=0)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(ic_tensor, os.path.join(save_path, 'initial_conditions.pt'))
            torch.save(traj_tensor, os.path.join(save_path, 'trajectories.pt'))

        return ic_tensor, traj_tensor

    def solve(self, ini_cond, t_final, num_points):
        """Solve the ODE via torchdiffeq, returning (t, sol).
        sol shape: [num_points, D]"""
        t = torch.linspace(0.0, t_final, num_points, dtype=torch.float32)
        if not isinstance(ini_cond, torch.Tensor):
            y0 = torch.tensor(ini_cond, dtype=torch.float32, requires_grad=True)
        else:
            y0 = ini_cond.clone().detach().requires_grad_(True)
        f_wrapped = lambda t, y: self.func(t, y, *self.args)
        sol = odeint(f_wrapped, y0, t)
        return t, sol

    @staticmethod
    def _soft_min_squared_dist(traj_flat, collected_flat, alpha=50.0):
        diffs = collected_flat - traj_flat.unsqueeze(0)
        sqd = torch.sum(diffs * diffs, dim=1)
        return - (1.0/alpha) * torch.logsumexp(-alpha * sqd, dim=0)

    def active_sample_initial(
        self, collected_data: torch.Tensor, bounds: torch.Tensor,
        t_final: float, num_points: int, alpha: float=50.0,
        lr: float=1e-2, steps: int=10, device: str='cpu'
    ):
        """Gradient-based search for most novel IC. See docs above."""
        N, T, D = collected_data.shape
        collected_flat = collected_data.reshape(N, -1).to(device)
        mins, maxs = bounds[:,0].to(device), bounds[:,1].to(device)
        x0 = ((mins + maxs)/2).clone().detach().to(device)
        x0.requires_grad_(True)

        opt = torch.optim.Adam([x0], lr=lr)
        t_grid = torch.linspace(0.0, t_final, num_points, device=device)

        for step in range(steps):
            opt.zero_grad()
            with torch.no_grad(): x0.clamp_(mins, maxs)
            sol = odeint(lambda t,y: self.func(t,y,*self.args), x0, t_grid)
            score = self._soft_min_squared_dist(sol.reshape(-1), collected_flat, alpha)
            (-score).backward()
            opt.step()
            print(step)
        with torch.no_grad(): x0.clamp_(mins, maxs)
        sol_best = odeint(lambda t,y: self.func(t,y,*self.args), x0, t_grid)
        return x0.detach(), t_grid, sol_best.detach()
