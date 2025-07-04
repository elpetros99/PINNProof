
import torch
import torch.nn as nn
import numpy as np
import random
from solvers.solver import Solver

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
    
    def __init__(self, func, ini_cond=None, t_final=None, num_points=None, model=None, *args, **kwargs):
        
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
        Solve the ODE using the neural network model.
        
        Returns:
            t : np.ndarray of time points
            Y : np.ndarray of shape (num_points, len(ini_cond)) with the solution
        """
        # Use instance values if not provided
        ini_cond = ini_cond or self.ini_cond
        t_final = t_final or self.t_final
        num_points = num_points or self.num_points
        
        # Prepare inputs
        delta0, omega0 = ini_cond
        t_tensor = torch.linspace(0, t_final, num_points, dtype=torch.float32).view(-1, 1)
        inputs = torch.cat([
            torch.full((num_points, 1), delta0),
            torch.full((num_points, 1), omega0),
            t_tensor
        ], dim=1)
        
        # Run prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # Reconstruct states
        delta_hat = delta0 + outputs[:, 0].numpy() * t_tensor.numpy().squeeze()
        omega_hat = omega0 + outputs[:, 1].numpy() * t_tensor.numpy().squeeze()
        
        return t_tensor.numpy().squeeze(), np.column_stack((delta_hat, omega_hat))
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
        ini_cond = ini_cond or self.ini_cond
        t_final = t_final or self.t_final
        num_points = num_points or self.num_points

        delta0, omega0 = ini_cond
        dt = t_final / (num_points - 1)  # defining the intervals
        
        # Initialise time and solution arrays
        t_array = np.linspace(0, t_final, num_points, dtype=np.float32)
        Y = np.zeros((num_points, 2), dtype=np.float32)
        Y[0] = [delta0, omega0]
        
        self.model.eval()
        with torch.no_grad():
            delta, omega = delta0, omega0
            for i in range(1, num_points):
                t_curr = t_array[i-1]
                input_tensor = torch.tensor([[delta, omega, t_curr]], dtype=torch.float32)
                output = self.model(input_tensor).squeeze(0)  # shape should be [2,]
                delta_dot, omega_dot = output.numpy()
                
                # Euler-style update
                delta = delta + dt * delta_dot
                omega = omega + dt * omega_dot
                Y[i] = [delta, omega]

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
        samples = self.generate_samples(bounds, num_trajectories, num_points)  # generate samples function needs to be corrected, use either this class or the other one
        u0s = samples[:, 0, 0]  # shape: (N,)
        v0s = samples[:, 1, 0]  # shape: (N,)
        t_grid = samples[:, 2, :] if samples.shape[1] > 2 else torch.linspace(
            bounds['t'][0], bounds['t'][1], num_points).repeat(num_trajectories, 1)  # shape: (N, T)

        residuals = []

        for i in range(num_trajectories):
            u0 = u0s[i]
            v0 = v0s[i]
            t = t_grid[i].view(-1, 1)  # (T, 1)
            t.requires_grad_(True)

            # Input tensor for the network: should be of shape (T, 3)
            u0_repeat = u0.repeat(num_points, 1) if isinstance(u0, torch.Tensor) else torch.full_like(t, u0)
            v0_repeat = v0.repeat(num_points, 1) if isinstance(v0, torch.Tensor) else torch.full_like(t, v0)
            inp = torch.cat([u0_repeat, v0_repeat, t], dim=1)  # (T, 3)

            # Passing through and predicting using the NN
            self.model.eval()
            out = self.model(inp)  # should be having shape of (T, 2)

            u_pred = u0 + out[:, 0] * t.squeeze()
            v_pred = v0 + out[:, 1] * t.squeeze()

            # Stack to get y_pred = [u(t), v(t)]
            y_pred = torch.stack([u_pred, v_pred], dim=1)  # (T, 2)

            # Computing the dy/dt via autograd
            du_dt = torch.autograd.grad(
                u_pred, t,
                grad_outputs=torch.ones_like(u_pred),
                create_graph=True,
                retain_graph=True
            )[0]

            dv_dt = torch.autograd.grad(
                v_pred, t,
                grad_outputs=torch.ones_like(v_pred),
                create_graph=True,
                retain_graph=True
            )[0]

            dy_dt = torch.stack([du_dt.squeeze(), dv_dt.squeeze()], dim=1)  # (T, 2)

            # Compute true RHS using self.func, func should be passed appropriately
            f = self.func(y_pred, t)  # (T, 2)

            # Residual: dy/dt - f(y, t)
            res = dy_dt - f  # (T, 2)
            residuals.append(res)

        return torch.stack(residuals, dim=0)  # shape should be of (N, T, 2)


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
# if __name__ == "__main__":
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
