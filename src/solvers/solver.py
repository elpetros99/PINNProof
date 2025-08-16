import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import random
from pyDOE import lhs
import os
import copy

class Solver():
    """
    Base class for ODE solvers. Subclasses must implement solve() and can use
    generate_dataset() and active_sample_initial() for data generation.
    """
    def __init__(self, func, interface_func, control_variables=2, *args, **kwargs):
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
        self.control_variables= control_variables

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
        """Build a dataset by solving the ODE for multiple ICs.

        Args:
            ic_ranges: Dictionary of ranges, which will be converted to Tensor[D,2] of [min,max] per state variable.
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

        ic_ranges = torch.tensor(list(ic_ranges.values()))
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
            ic_tensor_so_far = []
            for k in range(num_ic):
                print("Creating trajectory:", k)
                if k == 0:
                    rand = torch.rand(D)
                    x0 = bounds[:,0] + (bounds[:,1]-bounds[:,0]) * rand
                    traj = solve_ic(x0)
                else:
                    collected = torch.stack(traj_list, dim=0)  # [N, T, D]
                    ic_so_far = torch.stack(ic_tensor_so_far, dim=0) if ic_tensor_so_far else None
                    x0, _, sol = self.active_sample_initial(
                        collected, bounds, t_final, num_points,
                        alpha=10.0, steps=10, restarts=8, device=device,
                        ic_list=ic_so_far, ic_lambda=0.1, transient_weight=1.0, dup_tol=1e-3
                    )
                    traj = sol.cpu()
                ic_list.append(x0.cpu())
                ic_tensor_so_far.append(x0.cpu())
                traj_list.append(traj)

        # elif sampling == "active":
        #     bounds = ic_ranges.clone().float() 
        #     for k in range(num_ic):
        #         if k == 0:
        #             # first IC random
        #             rand = torch.rand(D)
        #             x0 = bounds[:,0] + (bounds[:,1]-bounds[:,0]) * rand
        #             traj = solve_ic(x0)
        #         else:
        #             collected = torch.stack(traj_list, dim=0)
        #             x0, _, sol = self.active_sample_initial(
        #                 collected, bounds, t_final, num_points, device=device
        #             )
        #             traj = sol.cpu()
        #         ic_list.append(x0.cpu())
        #         traj_list.append(traj)
        else:
            raise ValueError(f"Unknown sampling: {sampling}")

        ic_tensor = torch.stack(ic_list, dim=0)
        traj_tensor = torch.stack(traj_list, dim=0)
        traj_tensor = traj_tensor.permute(0, 2, 1)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(ic_tensor, os.path.join(save_path, 'initial_conditions.pt'))
            torch.save(traj_tensor, os.path.join(save_path, 'trajectories.pt'))

        return t_grid, traj_tensor, ic_tensor

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
    
    def split_data(self, dataset):
        """
        Splits a raw dataset into input features and targets for training.

        Args:
            dataset: Array-like of samples, each of shape (M, N), where
                    M = number of features per time step (including initial),
                    N = number of time steps.

        Returns:
            x_train: Tensor of shape (total_time_steps, M) with gradients enabled.
            y_train: Tensor of shape (total_time_steps, M-1) with gradients enabled.
        """
        # Convert the entire dataset into a single Tensor
        data = torch.Tensor(dataset)
        x_list, y_list = [], []

        # Process each sample individually
        for sample in data:
            # sample shape: (M, N)
            
            # === Target tensor (y) ===
            # Drop the first feature row (e.g., initial condition) and transpose
            # to get shape (N, M-1)
            y = sample[1:].T

            # === Input tensor (x) ===
            # Transpose sample to shape (N, M)
            x = sample.T
            
            # For each time step, overwrite columns 1..end with the values
            # from time step 0 (broadcasting initial features across time)
            x[:, 1:] = x[0, 1:]
            
            # Enable gradient tracking for both inputs and targets
            x.requires_grad_(True)
            y.requires_grad_(True)

            # Collect for later concatenation
            x_list.append(x)
            y_list.append(y)

        # Concatenate all samples along the first (time/batch) dimension
        # Final shapes:
        #   x_train: (sum of N over samples, M)
        #   y_train: (sum of N over samples, M-1)
        x_train = torch.cat(x_list, dim=0)
        y_train = torch.cat(y_list, dim=0)

        return x_train, y_train

    
    def get_trajectories(y_train: torch.Tensor, num_traj: int) -> torch.Tensor:
        """
        Re‐shapes a concatenated y_train of shape (num_traj * T, state_dim)
        into trajectories of shape (num_traj, state_dim, T).

        Args:
            y_train   (torch.Tensor): concatenated tensor, shape (num_traj * T, state_dim)
            num_traj  (int):           number of trajectories that were concatenated

        Returns:
            torch.Tensor: shape (num_traj, state_dim, T)
        """
        # total time‐points per trajectory
        total_time = y_train.size(0)
        state_dim  = y_train.size(1)
        T = total_time // num_traj

        # first reshape into (num_traj, T, state_dim), then swap axes 1<->2
        return y_train.view(num_traj, T, state_dim).permute(0, 2, 1)

    def active_sample_initial(
        self,
        collected_data: torch.Tensor,
        bounds: torch.Tensor,
        t_final: float,
        num_points: int,
        alpha: float = 400.0,
        lr: float = 0.1,
        steps: int = 15,
        restarts: int = 8,
        device: str = "cpu",
        ic_list: torch.Tensor = None,   # previously chosen ICs [N,D] (optional)
        ic_lambda: float = 0.1,         # strength of IC spacing bonus
        transient_weight: float = 1.0,  # 0=uniform, >0 emphasizes later times
        dup_tol: float = 1e-3,          # reject if trajectory MSE < dup_tol vs any existing
        optimizer: str = "lbfgs",       # "lbfgs" | "adam" | "hybrid"
        adam_warmup: int = 40,          # only used when optimizer="hybrid"
        ode_method: str = "dopri5",
    ):
        """
        Gradient search for a novel initial condition x0.

        optimizer:
        - "lbfgs": pure LBFGS (fast convergence but each step costs multiple ODE solves)
        - "adam": pure Adam
        - "hybrid": Adam warm-up then short LBFGS polish

        Returns:
        x0_best [D], t_grid [T], sol_best [T,D]
        """
        N, T, D = collected_data.shape
        mins, maxs = bounds[:, 0].to(device), bounds[:, 1].to(device)
        t_grid = torch.linspace(0.0, t_final, num_points, device=device)

        # per-dimension scale to neutralize ranges/units
        scale = (maxs - mins).clamp_min(1e-8)                   # [D]
        collected_norm = (collected_data.to(device) / scale)    # [N,T,D]
        collected_flat = collected_norm.reshape(N, -1)          # [N, T*D]

        # emphasize transients if requested
        if transient_weight != 0.0:
            w = torch.linspace(1.0, 1.0 + transient_weight, num_points, device=device)
        else:
            w = torch.ones(num_points, device=device)

        def flatten_weighted(sol):  # sol: [T,D]
            soln = (sol / scale) * w.view(-1, 1)
            return soln.reshape(-1)  # [T*D]

        def softmin_dist(traj_flat):
            # far from nearest existing trajectory (soft-min of squared distances)
            diffs = collected_flat - traj_flat.unsqueeze(0)     # [N, L]
            sqd = torch.sum(diffs * diffs, dim=1)               # [N]
            return -(1.0 / alpha) * torch.logsumexp(-alpha * sqd, dim=0)

        best_score, best_x0, best_sol = None, None, None

        for r in range(restarts):
            # ----- random restart on unconstrained variable u (mapped to box via sigmoid) -----
            with torch.no_grad():
                z0 = torch.rand(D, device=device).clamp(1e-4, 1 - 1e-4)
                u = torch.log(z0) - torch.log1p(-z0)            # logit
            u.requires_grad_(True)

            # ----- define a closure factory so we can reuse for LBFGS or Adam -----
            def compute_loss_and_score():
                x0 = mins + (maxs - mins) * torch.sigmoid(u)    # [D]
                sol = odeint(lambda t, y: self.func(t, y, *self.args),
                            x0, t_grid, method=ode_method)
                traj_flat = flatten_weighted(sol)

                score = softmin_dist(traj_flat)                 # novelty in trajectory space

                # IC dispersion term (optional)
                if ic_list is not None and ic_list.numel() > 0:
                    diffs_ic = ic_list.to(device) - x0.unsqueeze(0)  # [N,D]
                    ic_min = (diffs_ic * diffs_ic).sum(1).min()
                    score = score + ic_lambda * ic_min

                loss = -score                                    # we minimize in optimizers
                return loss, score, x0, sol

            # ----- optimize u -----
            if optimizer == "adam":
                opt = torch.optim.Adam([u], lr=lr)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(steps, 10), eta_min=lr * 0.05
                )
                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    loss, _, _, _ = compute_loss_and_score()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([u], max_norm=5.0)
                    opt.step()
                    sched.step()

            elif optimizer == "lbfgs":
                opt = torch.optim.LBFGS(
                    [u], lr=lr, max_iter=steps, max_eval=steps * 2,
                    history_size=10, line_search_fn="strong_wolfe"
                )

                def closure():
                    opt.zero_grad(set_to_none=True)
                    loss, _, _, _ = compute_loss_and_score()
                    loss.backward()
                    return loss

                try:
                    opt.step(closure)
                except RuntimeError:
                    # line search sometimes fails; evaluate once so we don't crash
                    with torch.enable_grad():
                        loss, _, _, _ = compute_loss_and_score()
                        loss.backward()

            elif optimizer == "hybrid":
                # Adam warm-up
                optA = torch.optim.Adam([u], lr=lr)
                for _ in range(adam_warmup):
                    optA.zero_grad(set_to_none=True)
                    loss, _, _, _ = compute_loss_and_score()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([u], max_norm=5.0)
                    optA.step()
                # short LBFGS polish
                optB = torch.optim.LBFGS(
                    [u], lr=lr, max_iter=max(5, steps // 2), max_eval=steps,
                    history_size=10, line_search_fn="strong_wolfe"
                )

                def closureB():
                    optB.zero_grad(set_to_none=True)
                    loss, _, _, _ = compute_loss_and_score()
                    loss.backward()
                    return loss

                try:
                    optB.step(closureB)
                except RuntimeError:
                    with torch.enable_grad():
                        loss, _, _, _ = compute_loss_and_score()
                        loss.backward()
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

            # ----- evaluate this restart -----
            with torch.no_grad():
                loss, score, x0_cand, sol_cand = compute_loss_and_score()

            if (best_score is None) or (score.item() > best_score):
                best_score = score.item()
                best_x0, best_sol = x0_cand.detach(), sol_cand.detach()

        # ----- final duplicate guard vs. collected_data -----
        with torch.no_grad():
            cand = (best_sol / scale).reshape(-1)
            if N > 0:
                diffs = collected_flat - cand.unsqueeze(0)      # [N,L]
                mse_min = (diffs * diffs).mean(dim=1).min().item()
            else:
                mse_min = float("inf")

            if mse_min < dup_tol:
                # fallback to a random point to ensure diversity
                z = torch.rand(D, device=device)
                x0_fb = mins + (maxs - mins) * z
                sol_fb = odeint(lambda t, y: self.func(t, y, *self.args),
                                x0_fb, t_grid, method=ode_method)
                best_x0, best_sol = x0_fb.detach(), sol_fb.detach()

        return best_x0, t_grid, best_sol

    # def active_sample_initial(
    #     self, collected_data: torch.Tensor, bounds: torch.Tensor,
    #     t_final: float, num_points: int, alpha: float=400.0,
    #     lr: float=1e-1, steps: int=10, restarts: int=8, device: str='cpu',
    #     ic_list: torch.Tensor=None,  # pass previously chosen ICs if available [N,D]
    #     ic_lambda: float=0.1,        # strength of IC spacing
    #     transient_weight: float=1.0, # 0 = uniform, >0 emphasizes later times
    #     dup_tol: float=1e-3          # reject if trajectory MSE < dup_tol vs any existing
    # ):
    #     """
    #     Gradient search for a novel IC that avoids boundary collapse and duplicates.
    #     """
    #     N, T, D = collected_data.shape
    #     mins, maxs = bounds[:,0].to(device), bounds[:,1].to(device)
    #     t_grid = torch.linspace(0.0, t_final, num_points, device=device)

    #     # Scale per state to neutralize units/ranges
    #     scale = (maxs - mins).clamp_min(1e-8)                 # [D]
    #     collected_norm = (collected_data / scale)             # [N,T,D]
    #     collected_flat = collected_norm.reshape(N, -1).to(device)

    #     # Time weights to emphasize transients (helps avoid identical attractor tails)
    #     if transient_weight != 0.0:
    #         w = torch.linspace(1.0, 1.0 + transient_weight, num_points, device=device)  # [T]
    #     else:
    #         w = torch.ones(num_points, device=device)

    #     def flatten_weighted(sol):  # sol: [T,D]
    #         soln = (sol / scale) * w.view(-1,1)               # [T,D]
    #         return soln.reshape(-1)

    #     def softmin_dist(traj_flat):
    #         diffs = collected_flat - traj_flat.unsqueeze(0)   # [N,L]
    #         sqd = torch.sum(diffs * diffs, dim=1)
    #         return - (1.0/alpha) * torch.logsumexp(-alpha * sqd, dim=0)

    #     # Keep best over several random restarts
    #     best_score, best_x0, best_sol = None, None, None

    #     for r in range(restarts):
    #         # random init in (0,1) → unconstrained u via logit
    #         with torch.no_grad():
    #             z0 = torch.rand(D, device=device).clamp(1e-4, 1-1e-4)
    #             u  = torch.log(z0) - torch.log1p(-z0)
    #         u.requires_grad_(True)

    #         opt = torch.optim.Adam([u], lr=lr)
    #         sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 10), eta_min=lr*0.05)

    #         for s in range(steps):
    #             opt.zero_grad()
    #             # smooth box projection
    #             x0 = mins + (maxs - mins) * torch.sigmoid(u)     # [D]
    #             sol = odeint(lambda t,y: self.func(t,y,*self.args), x0, t_grid, method="dopri5")
    #             traj_flat = flatten_weighted(sol)

    #             score = softmin_dist(traj_flat)                  # novelty in trajectory space

    #             # encourage IC dispersion if we have previous ICs
    #             if ic_list is not None and ic_list.numel() > 0:
    #                 diffs_ic = ic_list.to(device) - x0.unsqueeze(0)          # [N,D]
    #                 ic_min = (diffs_ic*diffs_ic).sum(1).min()
    #                 score = score + ic_lambda * ic_min

    #             (-score).backward()
    #             torch.nn.utils.clip_grad_norm_([u], max_norm=5.0)
    #             opt.step()
    #             sched.step()

    #         # Evaluate this restart
    #         with torch.no_grad():
    #             x0 = mins + (maxs - mins) * torch.sigmoid(u)
    #             sol = odeint(lambda t,y: self.func(t,y,*self.args), x0, t_grid, method="dopri5")
    #             traj_flat = flatten_weighted(sol)
    #             score = softmin_dist(traj_flat)

    #         if (best_score is None) or (score.item() > best_score):
    #             best_score = score.item()
    #             best_x0, best_sol = x0.detach(), sol.detach()

    #     # Final duplicate check against existing trajectories
    #     with torch.no_grad():
    #         cand = (best_sol/scale).reshape(-1)
    #         diffs = collected_flat - cand.unsqueeze(0)             # [N,L]
    #         mse = (diffs*diffs).mean(dim=1).min().item() if N>0 else float('inf')
    #         if mse < dup_tol:
    #             # fall back to a random LHS sample to ensure diversity
    #             z = torch.rand(D, device=device)
    #             x0_fallback = mins + (maxs - mins) * z
    #             sol_fb = odeint(lambda t,y: self.func(t,y,*self.args), x0_fallback, t_grid, method="dopri5")
    #             best_x0, best_sol = x0_fallback.detach(), sol_fb.detach()

    #     return best_x0, t_grid, best_sol

    # def active_sample_initial(
    #     self, collected_data: torch.Tensor, bounds: torch.Tensor,
    #     t_final: float, num_points: int, alpha: float=50.0,
    #     lr: float=1e-1, steps: int=20, device: str='cpu'
    # ):
    #     """Gradient-based search for most novel IC. See docs above."""
    #     N, T, D = collected_data.shape
    #     collected_flat = collected_data.reshape(N, -1).to(device)
    #     mins, maxs = bounds[:,0].to(device), bounds[:,1].to(device)
    #     x0 = ((mins + maxs)/2).clone().detach().to(device)
    #     x0.requires_grad_(True)

    #     opt = torch.optim.Adam([x0], lr=lr)
    #     t_grid = torch.linspace(0.0, t_final, num_points, device=device)

    #     for step in range(steps):
    #         opt.zero_grad()
    #         with torch.no_grad(): x0.clamp_(mins, maxs)
    #         sol = odeint(lambda t,y: self.func(t,y,*self.args), x0, t_grid)
    #         score = self._soft_min_squared_dist(sol.reshape(-1), collected_flat, alpha)
    #         (-score).backward()
    #         opt.step()
    #         print(step)
    #     with torch.no_grad(): x0.clamp_(mins, maxs)
    #     sol_best = odeint(lambda t,y: self.func(t,y,*self.args), x0, t_grid)
    #     return x0.detach(), t_grid, sol_best.detach()
