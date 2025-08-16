
import torch
import torch.nn as nn
import numpy as np
import random
from src.solvers.solver import Solver
from src.verification.utils import *

class Solver_NN(Solver):
    """
    A neural network-based ODE solver with optional training capability.
    This class uses a neural network to approximate the solution of an ODE.
    It inherits from the base Solver class and implements the solve method.
    
    Parameters:
        func       : callable f(t, y, *args, **kwargs) returning dy/dt as a torch.Tensor
        ini_cond   : list or array of initial values [y1(0), y2(0), ..., yN(0)]
        t_final    : final time up to which you want the solution
        num_points : number of time points between 0 and t_final
        args       : any extra positional arguments needed by func
        kwargs     : any extra keyword arguments needed by func
        model      : pre-trained model if any
    """
    
    def __init__(self, func, n_control, n_states, model=None, *args, **kwargs):
        
        #super().__init__(func, ini_cond, t_final, num_points, *args, **kwargs)

        # self.config = {
        #     'network': kwargs.get('network_params', {
        #         'N_INPUT': 3, 
        #         'N_OUTPUT': 2,
        #         'N_HIDDEN': 64,
        #         'N_LAYERS': 3
        #     }),
        #     'training': kwargs.get('training_params', {
        #         'epochs': 10000,
        #         'lr': 1e-3,
        #         'gamma': 0.9996,
        #         'recording_step': 500
        #     }),
        #     'domain': kwargs.get('domain_params', {
        #         'num_collocation_points': 11,
        #         'range_training_time': 2,
        #         'delta_range': [0, 1],
        #         'omega_range': [-0.5, 0.5],
        #         'num_initial_deltas': 10,
        #         'num_initial_omegas': 10
        #     }),
        #     'loss_weights': kwargs.get('loss_weights', [1, 3])
        # }
        self.network_params = kwargs.get('network_params', {
            'N_INPUT': 3, 
            'N_OUTPUT': 2,
            'N_HIDDEN': 64,
            'N_LAYERS': 3
        })
        self.training_params = kwargs.get('training_params', {
            'epochs': 10000,
            'lr': 1e-3,
            'gamma': 0.9996,
            'recording_step': 500
        })
        self.domain_params = kwargs.get('domain_params', {
            'num_collocation_points': 11,
            'range_training_time': 2,
            'delta_range': [0, 1],
            'omega_range': [-0.5, 0.5],
            'num_initial_deltas': 10,
            'num_initial_omegas': 10
        })
        
        self.loss_weights = kwargs.get('loss_weights', [1, 3])
        
        # self.physical_params = {
        #     'P_gen': self.args[0] if len(self.args) > 0 else 0.1,
        #     'D1': self.args[1] if len(self.args) > 1 else 0.1,
        #     'B12_V1_V2': self.args[2] if len(self.args) > 2 else 0.2,
        #     'm1': self.args[3] if len(self.args) > 3 else 0.4
        # }
        self.physical_params = kwargs.get('physical_params', {
                'P_gen': 0.10525,
                'D_1': 0.1,
                'B12_V1_V2': 0.2,
                'm1': 0.4
                })
        
        # Model handling
        self.model = model
        self.is_trained = model is not None
        self.requires_training = not self.is_trained
        # self.autograd+compatible
        # Training-related attributes
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        
        # Build model if needed
        if self.requires_training:
            self.model = self._build_model()

        self.func = func
        self.n_control = n_control
        self.n_states = n_states
        
    def _build_model(self):
        """Construct the neural network model"""
        # Calculate input normalization range
        delta_vals = np.linspace(*self.domain_params['delta_range'], 
                                self.domain_params['num_initial_deltas'])
        omega_vals = np.linspace(*self.domain_params['omega_range'], 
                                self.domain_params['num_initial_omegas'])
        t_vals = np.linspace(0, self.domain_params['range_training_time'], 
                            self.domain_params['num_collocation_points'])
        
        # Create sample input tensor for range calculation
        sample_input = torch.tensor(np.array(np.meshgrid(
            delta_vals, omega_vals, t_vals)).T.reshape(-1, 3), dtype=torch.float32)
        
        # Calculate range for normalization
        input_range = torch.max(sample_input, dim=0)[0] - torch.min(sample_input, dim=0)[0]
        
        # Build and return the model
        return FCN(
            N_INPUT=self.network_params['N_INPUT'],
            N_OUTPUT=self.network_params['N_OUTPUT'],
            N_HIDDEN=self.network_params['N_HIDDEN'],
            N_LAYERS=self.network_params['N_LAYERS'],
            range_input=input_range
        )
    
    def _generate_training_data(self):
        """Generate collocation points for physics-informed training in case of new training"""
        params = self.domain_params
        
        # Calculate grid steps, not necessary but for information only
        step_col_points = params['range_training_time'] / (params['num_collocation_points'] - 1)
        step_delta = (params['delta_range'][1] - params['delta_range'][0]) / (params['num_initial_deltas'] - 1)
        step_omega = (params['omega_range'][1] - params['omega_range'][0]) / (params['num_initial_omegas'] - 1)
        
        # Create value arrays
        delta_vals = np.linspace(*params['delta_range'], params['num_initial_deltas'])
        omega_vals = np.linspace(*params['omega_range'], params['num_initial_omegas'])
        t_vals = np.linspace(0, params['range_training_time'], params['num_collocation_points'])
        
        # Create full grid
        delta_rep = torch.tensor(np.repeat(delta_vals, len(omega_vals) * len(t_vals))).float().view(-1, 1)
        omega_rep = torch.tensor(np.tile(np.repeat(omega_vals, len(t_vals)), len(delta_vals))).float().view(-1, 1)
        t_rep = torch.tensor(np.tile(t_vals, len(delta_vals) * len(omega_vals))).float().view(-1, 1)
        t_rep.requires_grad = True
        
        return torch.cat([delta_rep, omega_rep, t_rep], dim=1)
        
        #self.model = None  # Placeholder for the neural network model
        
    def train(self):
        """Train the neural network using physics-informed loss, no labelled data points"""
        if not self.requires_training:
            print("Using pre-trained model - training skipped")
            return []        
        
        # Set random seeds for reproducibility
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)
        
        # Generate training data
        phy_input = self._generate_training_data()
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_params['lr'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.training_params['gamma'])
        
        # Extract physical parameters
        P_gen = self.physical_params['P_gen']
        D1 = self.physical_params['D1']
        B12_V1_V2 = self.physical_params['B12_V1_V2']
        m1 = self.physical_params['m1']
        lambda_l1, lambda_l2 = self.loss_weights
        
        # Training loop
        epochs = self.training_params['epochs']
        recording_step = self.training_params['recording_step']
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            preds = self.model(phy_input)
            delta_0 = phy_input[:, 0:1]
            omega_0 = phy_input[:, 1:2]
            t = phy_input[:, 2:3]
            
            # Reconstruct states
            delta_hat = delta_0 + preds[:, 0:1] * t
            omega_hat = omega_0 + preds[:, 1:2] * t
            
            # Compute gradients for physics loss
            ddelta_dt = torch.autograd.grad(
                delta_hat, t, torch.ones_like(delta_hat), 
                create_graph=True, retain_graph=True)[0]
            
            domega_dt = torch.autograd.grad(
                omega_hat, t, torch.ones_like(omega_hat), 
                create_graph=True)[0]
            
            # Physics-informed loss components
            physics_loss = torch.mean(
                (m1 * domega_dt + D1 * omega_hat + B12_V1_V2 * torch.sin(delta_hat) - P_gen)**2)
            consistency_loss = torch.mean((omega_hat - ddelta_dt)**2)
            
            # Total loss
            total_loss = lambda_l1 * physics_loss + lambda_l2 * consistency_loss
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Record loss
            self.loss_history.append(total_loss.item())
            
            # Print progress
            if (epoch % recording_step == 0) or (epoch == epochs - 1):
                print(f'Epoch {epoch}/{epochs}: Loss = {total_loss.item():.6f}')
        
        print(f'Training complete. Final loss: {self.loss_history[-1]:.6f}')
        return self.loss_history

    def solve(self, ini_cond=None, t_final=None, num_points=None):
        """
        Solve the ODE using the neural network model in a one-shot manner.

        ini_cond is a list, as given
        
        Returns:
            t : np.ndarray of time points
            Y : np.ndarray of shape (num_points, len(ini_cond)) with the solution
        """
        # Use instance values if not provided
        ini_cond = ini_cond if ini_cond is not None else self.ini_cond
        t_final = t_final if t_final is not None else self.t_final
        num_points = num_points if num_points is not None else self.num_points
        
        n_vars = len(ini_cond)

        # Prepare inputs
        ini_cond_tensor = torch.tensor(ini_cond, dtype=torch.float32).view(1, -1)  # reshaping to (1, n_vars)
        t_tensor = torch.linspace(0, t_final, num_points, dtype=torch.float32).view(-1, 1)  # time tensor (num_points, 1)

        # The final input tensor shape will be (num_points, n_vars + 1).
        ini_cond_repeated = ini_cond_tensor.repeat(num_points, 1)
        inputs = torch.cat([ini_cond_repeated, t_tensor], dim=1)
        
        # Run prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)  # expected shape is (num_points, n_vars)
        
        # Reconstruct states by indirect prediction
        # Y = ini_cond_tensor + outputs * t_tensor

        # Reconstruct states by direct prediction
        Y = outputs
        
        return t_tensor.numpy().squeeze(), Y.detach().numpy()
        # Implement the logic to use the neural network model to solve the ODE
        #raise NotImplementedError("Neural network-based ODE solving not implemented yet.")
        

    def solve_recurrent(self, ini_cond=None, t_final=None, num_points=None):
        """
        Solve the ODE using the neural network model in a recurrent fashion.
        
        Returns:
            t : np.ndarray of time points
            Y : np.ndarray of shape (num_points, len(ini_cond)) with the solution
        """
        # Use instance values if not provided
        ini_cond = ini_cond if ini_cond is not None else self.ini_cond
        t_final = t_final if t_final is not None else self.t_final
        num_points = num_points if num_points is not None else self.num_points

        n_vars = len(ini_cond)
        dt = t_final / (num_points - 1)  # time step

        t_array = np.linspace(0, t_final, num_points, dtype=np.float32)
        Y = np.zeros((num_points, n_vars), dtype=np.float32)
        Y[0] = np.array(ini_cond, dtype=np.float32)
 
        self.model.eval()
        with torch.no_grad():
            for i in range(1, num_points):
                current_state = Y[i-1]
                t_curr = t_array[i-1]

                input_tensor = torch.cat([
                    torch.tensor(current_state, dtype=torch.float32).view(1, -1),
                    torch.tensor([[t_curr]], dtype=torch.float32)], dim=1)   # shape should be (1, n_vars+1)
                
                output = self.model(input_tensor)

                # Insdirect prediction
                derivatives = output.numpy().flatten()
                next_state = current_state + dt * derivatives

                # Direct prediction
                # next_state = output.numpy().flatten()

                Y[i] = next_state

        return t_array, Y    
        
    
    def compute_residuals(self, bounds, num_trajectories, num_points):
        """
        Compute residuals dy/dt - f(y, t) using autograd and the known equations which we have inside func.

        Args:
            bounds: for now, assuming this to be a dictionary with keys 'u', 'v', 't', each mapping to (lower, upper)
            num_trajectories: int, number of different initial conditions
            num_points: int, number of time points per trajectory

        Returns:
            residuals: torch.Tensor of shape (num_trajectories, num_timepoints, 2)
        """
        var_names = [k for k in bounds.keys() if k != 't']
        n_vars = len(var_names)

        # Generate batched samples
        y0s, t_grid = self.generate_batched_samples(bounds, var_names, num_trajectories, num_points)
        t_grid.requires_grad_(True)

        t_vec = t_grid.reshape(-1, 1)
        y0s_vec = y0s.repeat_interleave(num_points, dim=0)   # Repeat initial conditions for each time point: (N, n_vars) -> (N * T, n_vars)

        # Concatenate to form the model input of shape (Num_trajectories * num_points, n_vars + 1)
        model_input = torch.cat([t_vec, y0s_vec], dim=1)

        self.model.eval()
        y_pred = self.model(model_input)

        # y_pred = y0s_vec + model_output * t_vec  # if model is to give indirect output, not considered now

        # Compute dy/dt for all points at once.
        dy_dt_vec = torch.autograd.grad(
            outputs=y_pred,
            inputs=t_vec,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True, retain_graph=True,
        )[0]
        # dy_dt_vec will have shape (N * T, n_vars)

        # Get the initial conditions for the last 3 variables.
        y0s_control = y0s_vec[:, self.n_states:] # Shape: (N * T, 3)
        y_full = torch.cat([y_pred, y0s_control], dim=1) # Shape: (N * T, 7) for the actual func function
        
        f_val_list = []
        for i in range(y_full.shape[0]):
            # Get the i-th state vector (shape 7,) and i-th time value
            y = y_full[i]
            t = t_vec[i]
        
            f = self.func(t, y) # Returns a tensor of shape (7,)
            f_val_list.append(f)
        
        f_val_full = torch.stack(f_val_list, dim=0)   

        # f_val_full = self.func(t_vec, y_full)  # this can be used if func is vectorised
        f_val_pred = f_val_full[:, :self.n_states]

        residuals_vec = dy_dt_vec - f_val_pred

        residuals = residuals_vec.view(num_trajectories, num_points, self.n_states)   # reshaping as required
    
        return residuals
    # def compute_residuals(self, bounds, num_trajectories, num_points):
    #     """
    #     Compute PINN residuals r(t) = dy/dt - f(t, y) for a 6th-order SM with 2 controls (V_t, theta_vs).
    #     Returns: (num_trajectories, num_points, self.n_states)
    #     """
    #     import torch

    #     # ---- explicit variable order (match training) ----
    #     var_order = ['delta','omega','E_d_dash','E_q_dash','E_q_dd','E_d_dd','V_t','theta_vs']

    #     # Samples: y0s shape (N, 8), t_grid shape (N, T, 1)
    #     y0s, t_grid = self.generate_batched_samples(bounds, var_order, num_trajectories, num_points)

    #     device = next(self.model.parameters()).device
    #     dtype  = next(self.model.parameters()).dtype
    #     y0s, t_grid = y0s.to(device=device, dtype=dtype), t_grid.to(device=device, dtype=dtype)

    #     # Flatten time and repeat ICs
    #     t_vec     = t_grid.reshape(-1, 1).detach().requires_grad_(True)        # (N*T, 1)
    #     y0s_vec   = y0s.repeat_interleave(num_points, dim=0)                   # (N*T, 8)
    #     y0_ctrl   = y0s_vec[:, self.n_states:]                                 # (N*T, 2)

    #     # Model input: [y0(8), t]
    #     model_in  = torch.cat([y0s_vec, t_vec], dim=1)                          # (N*T, 9)

    #     # Predict states y_hat(t)
    #     self.model.eval()
    #     y_hat = self.model(model_in)                                            # (N*T, 6)

    #     # Time derivatives dy/dt via autograd wrt t
    #     dy_dt = torch.cat([
    #         torch.autograd.grad(
    #             y_hat[:, k:k+1], t_vec,
    #             grad_outputs=torch.ones_like(y_hat[:, k:k+1]),
    #             create_graph=True,
    #             retain_graph=(k < self.n_states - 1)
    #         )[0]
    #         for k in range(self.n_states)
    #     ], dim=1)                                                               # (N*T, 6)

    #     # Full state for RHS f(t, y)
    #     y_full = torch.cat([y_hat, y0_ctrl], dim=1)                             # (N*T, 8)

    #     # Evaluate RHS; prefer batched func(t, y) -> (N*T, 8)
    #     try:
    #         f_full = self.func(t_vec.squeeze(1), y_full)                        # (N*T, 8)
    #     except TypeError:
    #         # fallback: unbatched loop if func isn't vectorized
    #         f_full = torch.stack([self.func(t_vec[i, 0], y_full[i]) for i in range(y_full.shape[0])], dim=0)

    #     f_val = f_full[:, :self.n_states]                                       # (N*T, 6)

    #     # Residuals
    #     residuals = (dy_dt - f_val).view(num_trajectories, num_points, self.n_states)
    #     return residuals

    # def compute_residuals(self, bounds, num_trajectories, num_points):
    #     """
    #     Residuals: dy/dt - f(t, y) for the 6th-order SM with 2 exogenous inputs (V_t, theta_vs).
    #     Returns: (num_trajectories, num_points, self.n_states)
    #     """
    #     # --- order of variables must match the model's training order ---
    #     var_names = [k for k in bounds.keys() if k != 't']  # ['delta','omega','E_d_dash','E_q_dash','E_q_dd','E_d_dd','V_t','theta_vs']
    #     y0s, t_grid = self.generate_batched_samples(bounds, var_names, num_trajectories, num_points)

    #     # device
    #     device = next(self.model.parameters()).device
    #     y0s   = y0s.to(device)
    #     t_grid= t_grid.to(device)

    #     # flatten time and enable grads wrt t
    #     t_vec = t_grid.reshape(-1, 1).clone().detach().requires_grad_(True)          # (N*T,1)
    #     # repeat ICs for each time point
    #     y0s_vec = y0s.repeat_interleave(num_points, dim=0)                            # (N*T, 8)

    #     # split into states (first 6) and controls (last 2)
    #     y0_states_vec = y0s_vec[:, :self.n_states]                                    # (N*T, 6)
    #     y0_ctrl_vec   = y0s_vec[:, self.n_states:]                                    # (N*T, 2)

    #     # model input: SAME as in your solve(): [y0(8), t]
    #     model_input = torch.cat([y0s_vec, t_vec], dim=1)                              # (N*T, 9)

    #     # forward pass (keep graph! we need grads wrt t)
    #     self.model.eval()
    #     net_out = self.model(model_input)                                             # (N*T, 6)

    #     # reconstruct the trajectory y_hat used in physics
    #     # if getattr(self, "outputs_are_slopes", False):
    #     #     # y(t) = y0 + slope * t
    #     #     y_hat = y0_states_vec + net_out * t_vec
    #     # else:
    #         # network outputs states directly
    #     y_hat = net_out                                                           # (N*T, 6)

    #     # per-state time derivatives dy/dt
    #     grads = []
    #     for k in range(self.n_states):
    #         gk = torch.autograd.grad(
    #             outputs=y_hat[:, k:k+1],
    #             inputs=t_vec,
    #             grad_outputs=torch.ones_like(t_vec),
    #             create_graph=True,
    #             retain_graph=(k < self.n_states - 1)
    #         )[0]                                                                      # (N*T,1)
    #         grads.append(gk)
    #     dy_dt = torch.cat(grads, dim=1)                                               # (N*T, 6)

    #     # build full 8-dim state to evaluate RHS: [y_hat(6), controls(2)]
    #     # we don't need grads through f, so detach to save memory
    #     y_hat_det = y_hat.detach()
    #     y_full = torch.cat([y_hat_det, y0_ctrl_vec.detach()], dim=1)                  # (N*T, 8)

    #     # evaluate RHS f(t, y) using your machine.forward signature
    #     f_list = []
    #     with torch.no_grad():
    #         for i in range(y_full.shape[0]):
    #             ti = t_vec[i, 0]
    #             fi = self.func(ti, y_full[i])                                         # (8,)
    #             f_list.append(fi)
    #     f_full = torch.stack(f_list, dim=0).to(dy_dt.dtype).to(device)                # (N*T, 8)

    #     # keep only the first 6 derivatives (the last 2 are zeros for V_t, theta_vs)
    #     f_val = f_full[:, :self.n_states]                                             # (N*T, 6)

    #     # residuals and reshape
    #     residuals_vec = dy_dt - f_val                                                 # (N*T, 6)
    #     residuals = residuals_vec.view(num_trajectories, num_points, self.n_states)
    #     return residuals

    def pinn_residual_heatmaps_by_state_y(
        solver,                    # your Solver_NN (needs .model, .func, .n_states, .generate_batched_samples)
        bounds,                    # dict with keys for 8 states + 't'
        num_trajectories: int,
        num_points: int,
        *,
        outputs_are_slopes: bool = False,
        state_labels = (r"$\delta$", r"$\omega$", r"$E'_d$", r"$E'_q$", r"$E''_q$", r"$E''_d$"),
        log10: bool = True,
        eps: float = 1e-12,
        save_dir: str | None = None,
    ):
        """
        Computes residuals r = dy/dt - f(t,y) and plots one heatmap per state:
        x-axis: time, y-axis: IC value of the SAME state, color: residual magnitude.
        Returns (R, y0s): residual tensor (N,T,S) and the IC matrix (N,8).
        """
        def _edges_from_centers(x):
            x = np.asarray(x)
            if x.size == 1:
                return np.array([x[0] - 0.5, x[0] + 0.5])
            mid = 0.5 * (x[:-1] + x[1:])
            first = x[0] - (mid[0] - x[0])
            last  = x[-1] + (x[-1] - mid[-1])
            return np.concatenate(([first], mid, [last]))

        def _time_edges(t0, t1, T):
            t = np.linspace(t0, t1, T)
            if T == 1:
                return np.array([t0 - 0.5, t1 + 0.5])
            mids = 0.5 * (t[:-1] + t[1:])
            left  = t[0]  - (mids[0] - t[0])
            right = t[-1] + (t[-1] - mids[-1])
            return np.concatenate(([left], mids, [right]))

        # ------------------ sample ICs/time exactly once ------------------
        var_names = [k for k in bounds.keys() if k != 't']  # keep your order
        y0s, t_grid = solver.generate_batched_samples(bounds, var_names, num_trajectories, num_points)
        # Model device for autograd; func (machine.forward) will be run on CPU
        model_device = next(solver.model.parameters()).device

        y0s = y0s.to(model_device)
        t_grid = t_grid.to(model_device)

        # Flatten time (enable grad)
        t_vec = t_grid.reshape(-1, 1).clone().detach().requires_grad_(True)     # (N*T,1)
        # Repeat ICs per time step
        y0s_vec = y0s.repeat_interleave(num_points, dim=0)                       # (N*T, 8)

        # Split states/controls
        S = solver.n_states
        y0_states_vec = y0s_vec[:, :S]                                           # (N*T, 6)
        y0_ctrl_vec   = y0s_vec[:, S:] if y0s_vec.shape[1] > S else None         # (N*T, 2) or None

        # ------------------ forward PINN ------------------
        # Model input matches your solve(): [y0(8), t]
        model_input = torch.cat([t_vec,y0s_vec], dim=1)                         # (N*T, 9)
        solver.model.eval()
        net_out = solver.model(model_input)                                      # (N*T, 6)

        # Reconstruct y(t)
        if outputs_are_slopes:
            y_hat = y0_states_vec + net_out * t_vec                              # slopes -> states
        else:
            y_hat = net_out                                                      # states directly

        # ------------------ dy/dt (per state) ------------------
        grads = []
        for k in range(S):
            gk = torch.autograd.grad(
                outputs=y_hat[:, k:k+1],
                inputs=t_vec,
                grad_outputs=torch.ones_like(y_hat[:, k:k+1]),
                create_graph=False,
                retain_graph=(k < S - 1)
            )[0]                                                                 # (N*T,1)
            grads.append(gk)
        dy_dt = torch.cat(grads, dim=1)                                          # (N*T,6)

        # ------------------ RHS f(t,y) on CPU (robust to device mix) ------------------
        # Detach y and move to CPU for machine.forward(t,y)
        y_hat_cpu   = y_hat.detach().cpu()
        y0_ctrl_cpu = y0_ctrl_vec.detach().cpu() if y0_ctrl_vec is not None else None
        y_full_cpu  = torch.cat([y_hat_cpu, y0_ctrl_cpu], dim=1) if y0_ctrl_cpu is not None else y_hat_cpu
        t_cpu       = t_vec.detach().cpu().squeeze(-1)

        f_list = []
        with torch.no_grad():
            for i in range(y_full_cpu.shape[0]):
                fi = solver.func(t_cpu[i], y_full_cpu[i])                        # (8,)
                f_list.append(fi)
        f_full = torch.stack(f_list, dim=0).to(dy_dt.dtype)                      # (N*T,8) on CPU
        f_val  = f_full[:, :S].to(model_device)                                  # back to model device

        # ------------------ residuals and reshape ------------------
        R = (dy_dt - f_val).view(num_trajectories, num_points, S)                # (N,T,6)
        R_np   = R.detach().cpu().numpy()
        y0s_np = y0s.detach().cpu().numpy()

        # ------------------ plot heatmaps ------------------
        t0, t1 = bounds['t']
        t_edges = _time_edges(t0, t1, num_points)

        if log10:
            Z_title = r"$\log_{10}(|\mathrm{residual}|)$"
        else:
            Z_title = "residual"

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for s in range(S):
            # y-axis = IC of state s
            yvals   = y0s_np[:, s]                       # (N,)
            order   = np.argsort(yvals)
            y_sorted= yvals[order]
            Z       = R_np[order, :, s]
            Zplot   = np.log10(np.abs(Z) + eps) if log10 else Z
            y_edges = _edges_from_centers(y_sorted)

            plt.figure(figsize=(8, 4))
            plt.pcolormesh(t_edges, y_edges, Zplot, shading='auto')
            plt.colorbar(label=Z_title)
            plt.xlabel("Time [s]")
            plt.ylabel(f"IC of {state_labels[s]}")
            plt.title(f"Residual heatmap vs time — {state_labels[s]}")
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"residual_heatmap_{s}.png"), dpi=200)
            plt.show()

        return R, y0s_np

    def generate_batched_samples(self, bounds, var_names, num_trajectories, num_points):
        # create a tensor for the lower bounds and a tensor for the upper bounds, then sample uniformly in the space between them
        n_vars = len(var_names)

        lower_bounds = torch.tensor([bounds[v][0] for v in var_names]).view(1, -1)
        upper_bounds = torch.tensor([bounds[v][1] for v in var_names]).view(1, -1)

        y0s = lower_bounds + (upper_bounds - lower_bounds) * torch.rand(num_trajectories, n_vars)
        t_lower, t_upper = bounds['t']
        t_row = torch.linspace(t_lower, t_upper, num_points).view(1, -1)
        t_grid = t_row.repeat(num_trajectories, 1)

        return y0s, t_grid



class Normalization_strat(nn.Module):
    def __init__(self, tensor_range):
        super().__init__()
        self.register_buffer('tensor_range', tensor_range)  # Ensures proper device placement   
    def forward(self, x):
        return x / (self.tensor_range + 1e-5)
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, range_input):
        super().__init__()
        activation = nn.Tanh
        # Input normalization
        self.norm = Normalization_strat(range_input)
        # Input layer
        self.input_layer = nn.Sequential(
            self.norm,
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        )
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(N_LAYERS - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(N_HIDDEN, N_HIDDEN),
                    activation()))
        # Output layer
        self.output_layer = nn.Linear(N_HIDDEN, N_OUTPUT)
        # Initialize weights
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_normal_(self.input_layer[1].weight)
        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer[0].weight)
        nn.init.xavier_normal_(self.output_layer.weight)
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


# num_points_solver = 201
# num_collocation_points = 11
# range_training_time =  2


# number_initial_deltas = 10   
# range_initial_deltas = [0, 1]
# number_initial_omegas = 10   
# range_initial_omegas = [-0.5, 0.5]
# step_col_points = range_training_time/(num_collocation_points-1)    
# step_initial_deltas = (range_initial_deltas[1] - range_initial_deltas[0]) / (number_initial_deltas - 1)
# step_initial_omegas = (range_initial_omegas[1] - range_initial_omegas[0]) / (number_initial_omegas - 1)

# num_neurons = 64
# num_layers = 3
# input_neurons = 3   
# output_neurons = 2
# num_training_epochs = 12000  
# recording_step = 500
# learning_rate = 1e-3 
# gamma = 0.9996

# delta_values = np.arange(range_initial_deltas[0], range_initial_deltas[1]+step_initial_deltas, step_initial_deltas)
# omega_values = np.arange(range_initial_omegas[0], range_initial_omegas[1]+step_initial_omegas, step_initial_omegas)
# t_values = np.arange(0, range_training_time+step_col_points, step_col_points)
# delta_rep = torch.tensor(np.repeat(delta_values, len(omega_values)*len(t_values))).float().view(-1,1)
# omega_rep = torch.tensor(np.tile(np.repeat(omega_values, len(t_values)), len(delta_values))).float().view(-1,1)
# t_rep = torch.tensor(np.tile(t_values, len(delta_values) * len(omega_values))).float().view(-1,1).requires_grad_(True)
# phy_input = torch.cat([delta_rep, omega_rep, t_rep], dim=1)  
# range_input = torch.max(phy_input, dim=0)[0] - torch.min(phy_input, dim=0)[0]  

# lambda_l1 = 1 
# lambda_l2 = 3
# PINN = FCN(input_neurons, output_neurons, num_neurons, num_layers, range_input)  

# sample_input = torch.tensor([0.1, 0.3, 0.5])
# sample_output = PINN(sample_input)
# print(sample_output)

# def _build_model():
#     """Construct the neural network model"""
#     # Calculate input normalization range
#     delta_vals = np.linspace(range_initial_deltas[0], range_initial_deltas[1], number_initial_deltas)
#     omega_vals = np.linspace(range_initial_omegas[0], range_initial_omegas[1], number_initial_omegas)
#     t_vals = np.linspace(0, range_training_time, num_collocation_points)
    
#     # Create sample input tensor for range calculation
#     sample_input = torch.tensor(np.array(np.meshgrid(
#         delta_vals, omega_vals, t_vals)).T.reshape(-1, 3), dtype=torch.float32)
    
#     # Calculate range for normalization
#     input_range = torch.max(sample_input, dim=0)[0] - torch.min(sample_input, dim=0)[0]
    
#     # Build and return the model
#     return FCN(
#         N_INPUT=3,
#         N_OUTPUT=2,
#         N_HIDDEN=32,
#         N_LAYERS=3,
#         range_input=input_range
#     )
# model = _build_model()
# print(model(sample_input))

# def _generate_training_data():
#     params = {
#         'num_collocation_points': 11,
#         'range_training_time': 2,
#         'delta_range': [0, 1],
#         'omega_range': [-0.5, 0.5],
#         'num_initial_deltas': 10,
#         'num_initial_omegas': 10
#     }
#     # Calculate grid steps
#     step_col_points = params['range_training_time'] / (params['num_collocation_points'] - 1)
#     step_delta = (params['delta_range'][1] - params['delta_range'][0]) / (params['num_initial_deltas'] - 1)
#     step_omega = (params['omega_range'][1] - params['omega_range'][0]) / (params['num_initial_omegas'] - 1)
    
#     # Create value arrays
#     delta_vals = np.linspace(*params['delta_range'], params['num_initial_deltas'])
#     omega_vals = np.linspace(*params['omega_range'], params['num_initial_omegas'])
#     t_vals = np.linspace(0, params['range_training_time'], params['num_collocation_points'])
    
#     # Create full grid
#     delta_rep = torch.tensor(np.repeat(delta_vals, len(omega_vals) * len(t_vals))).float().view(-1, 1)
#     print(delta_rep)
#     omega_rep = torch.tensor(np.tile(np.repeat(omega_vals, len(t_vals)), len(delta_vals))).float().view(-1, 1)
#     print(omega_rep)
#     t_rep = torch.tensor(np.tile(t_vals, len(delta_vals) * len(omega_vals))).float().view(-1, 1)
#     t_rep.requires_grad = True
    
#     return torch.cat([delta_rep, omega_rep, t_rep], dim=1), delta_rep

# sample_phy_input, d = _generate_training_data()
# print(sample_phy_input.shape)
# sample_output = model(sample_phy_input)
# print(sample_output.shape)

# a = sample_phy_input.detach().numpy()

# def solve(ini_cond=None, t_final=None, num_points=None):
#     """
#     Solve the ODE using the neural network model.
    
#     Returns:
#         t : np.ndarray of time points
#         Y : np.ndarray of shape (num_points, len(ini_cond)) with the solution
#     """
#     # Use instance values if not provided
#     # ini_cond = ini_cond or self.ini_cond
#     # t_final = t_final or self.t_final
#     # num_points = num_points or self.num_points
    
#     # Prepare inputs
#     delta0, omega0 = ini_cond
#     t_tensor = torch.linspace(0, t_final, num_points, dtype=torch.float32).view(-1, 1)
#     inputs = torch.cat([
#         torch.full((num_points, 1), delta0),
#         torch.full((num_points, 1), omega0),
#         t_tensor
#     ], dim=1)
    
#     # Run prediction
#     model.eval()
#     with torch.no_grad():
#         outputs = model(inputs)
    
#     # Reconstruct states
#     delta_hat = delta0 + outputs[:, 0].numpy() * t_tensor.numpy().squeeze()
#     omega_hat = omega0 + outputs[:, 1].numpy() * t_tensor.numpy().squeeze()
    
#     return t_tensor.numpy().squeeze(), np.column_stack((delta_hat, omega_hat))

# T, Y = solve(ini_cond=[0.2, 0.4], t_final=2.0, num_points=30)







    


# Example usage
if __name__ == "__main__":
    # Define a placeholder ODE function (not used directly in NN solver)
    def my_system(Y, t, *args, **kwargs):
        # this function takes as input the current state of the variables, and outputs their derivatives w.r.t t
        delta, omega = Y
        (D1, B12, m1, P_gen) = args
        ddelta_dt = omega
        domega_dt = (-D1*omega - B12*np.sin(delta) + P_gen) / m1 
        return [ddelta_dt, domega_dt]

    # Physical parameters
    params = (0.10525, 0.1, 0.2, 0.4)  # P_gen, D1, B12_V1_V2, m1

    # Configure solver
    solver = Solver_NN(
        func=my_system,
        physical_params={
            'P_gen': 0.10525,
            'D_1': 0.1,
            'B12_V1_V2': 0.2,
            'm1': 0.4        
            },
        network_params={
            'N_INPUT': 3,
            'N_OUTPUT': 2,
            'N_HIDDEN': 64,
            'N_LAYERS': 3
        },
        training_params={
            'epochs': 12000,
            'lr': 1e-3,
            'gamma': 0.9996,
            'recording_step': 1000
        },
        domain_params={
            'num_collocation_points': 11,
            'range_training_time': 2,
            'delta_range': [0, 1],
            'omega_range': [-0.5, 0.5],
            'num_initial_deltas': 10,
            'num_initial_omegas': 10
        },
        loss_weights=[1, 3]
    )

    # Train the model
    loss_history = solver.train()

    # Solve for specific initial conditions
    t, solution = solver.solve(
        ini_cond=[0.05, 0.0],
        t_final=15,
        num_points=200
    )

    # Extract solutions
    delta = solution[:, 0]
    omega = solution[:, 1]

    print(f"Solution computed at {len(t)} time points")
    print(f"Delta range: {delta.min():.4f} to {delta.max():.4f}")
    print(f"Omega range: {omega.min():.4f} to {omega.max():.4f}")
