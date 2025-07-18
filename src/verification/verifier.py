import torch 
import torch.nn as nn 
from utils import *
import numpy as np

class verifier(nn.Module):
    """
    A base class for verification of ODE solutions.
    
    This class is intended to be subclassed for specific verification strategies.
    imput 2 solvers
    verify the solution of solver1 against solver2
    find the worst case error between the two solutions
    """
    def __init__(self, solver1, solver2, bounds):
        super(verifier, self).__init__()
        self.solver1 = solver1
        self.solver2 = solver2
        self.var_bounds = bounds

    def forward(self, *args, **kwargs):
        """
        Verify the solution of solver1 against solver2.
        
        This method should be implemented in subclasses to define the verification strategy.
        
        Returns:
            A tensor representing the verification result (e.g., error).
        """
        # solver1 and solver2 must have some flag variable (eg: boolean contains_grad) to denote whether they have gradients or not, depending on this the lipschitz will be computed
        if self.solver1.contains_grad:
            L1 = self.lipschitz_method_grads(self.solver1, self.bounds, n_samples=1000) # compute lipschitz
        else:
            L1 = self.lipschitz_method(self.solver1, self.bounds, n_samples=1000)

        if self.solver2.contains_grad:
            L2 = self.lipschitz_method_grads(self.solver2, self.bounds, n_samples=1000) # compute lipschitz
        else:
            L2 = self.lipschitz_method(self.solver2, self.bounds, n_samples=1000)

        # convert bounds from dictionary to tensors
        x_lower = torch.tensor([
            self.bounds['u'][0],
            self.bounds['v'][0],
            self.bounds['t'][0]
        ], dtype=torch.float32).unsqueeze(0)  # shape: (1, 3)
        x_upper = torch.tensor([
            self.bounds['u'][1],
            self.bounds['v'][1],
            self.bounds['t'][1]
        ], dtype=torch.float32).unsqueeze(0)  # shape: (1, 3)

        # Evaluating both solvers at x_lower
        y1 = self.solver1(x_lower)  # shape: (1, 2)
        y2 = self.solver2(x_lower)  # shape: (1, 2)

        # Norm of the initial output difference
        initial_error = torch.norm(y1 - y2, p=2)  # L2 norm
        # Input difference norm
        input_range = torch.norm(x_upper - x_lower, p=2)

        # Total error bound according to the lipschitz expressoion
        max_error_bound = initial_error + (L1 + L2) * input_range

        return max_error_bound
        #raise NotImplementedError("Subclasses must implement this method.")
    
    def lipschitz_method_grads(self, solver, bounds, n_samples=1000):
        # Generate samples [u0, v0, t]
        bounds = torch.tensor(list(bounds.values()))
        bounds_for_sampling = list(map(tuple, bounds.tolist()))
        samples = sampling_domain(bounds_for_sampling, num_points=5)  # this creates a large number of samples, exponential
        # samples = generate_samples(bounds, n_samples, method='uniform')  # shape (n_samples, 3)
        samples_tensor = torch.tensor(samples, dtype=torch.float32, requires_grad=True)
        max_norm = 0.0
        for sample in samples_tensor:
            preds = solver(sample)  # forward pass
            u0, v0, t = sample
            # Compute state predictions: u = u0 + A*t, v = v0 + B*t
            u_pred = u0 + preds[0] * t
            v_pred = v0 + preds[1] * t
            outputs = torch.cat([u_pred.reshape(1), v_pred.reshape(1)])
            
            # Compute Jacobian: d(outputs)/d(inputs)
            jacobian = torch.zeros(2, 3)  # 2 outputs, 3 inputs
            for i in range(2):
                # Retain graph for second output
                retain = i < 1
                grad_outputs = torch.zeros(2)
                grad_outputs[i] = 1.0
                gradients = torch.autograd.grad(
                    outputs, sample, grad_outputs=grad_outputs,
                    retain_graph=retain, create_graph=False
                )[0]
                jacobian[i] = gradients
            
            # Compute spectral norm
            jac_np = jacobian.detach().numpy()
            _, s, _ = svd(jac_np)
            spectral_norm = max(s)
            if spectral_norm > max_norm:
                max_norm = spectral_norm
        
        return max_norm
    
    def lipschitz_method(self, bounds, n_samples=1000, eps=1e-5, model=None):
        # Generate samples [u0, v0, t]

        bounds = torch.tensor(list(bounds.values()))
        bounds_for_sampling = list(map(tuple, bounds.tolist()))
        samples = sampling_domain(bounds_for_sampling, num_points=5)

        # samples = generate_samples(bounds, n_samples, method='uniform')
        max_norm = 0.0
        
        for sample in samples:
            u0, v0, t = sample
            # Evaluate solver at original point (use n=2 for start and end points only)
            solver = Solver(ini_cond=[u0, v0], t_final=t, num_points=2, P_gen=power_gen, D1=D1, B12_V1_V2=B12_V1_V2, m1=m1)
            t_test, delta_values, omega_values = solver.solve_system()
            #traj = solver.solve_system(u0, v0, t, n=2)
            F_orig = np.array([delta_values[-1], omega_values[-1]])  # State at t
            jacobian = np.zeros((2, 3))  # 2 outputs, 3 inputs
            for i in range(3):  # Perturb each input dimension
                sample_perturbed = sample.copy()
                sample_perturbed[i] += eps
                u, v, t_ = sample_perturbed
                # Evaluate solver at perturbed point
                solver = Solver(ini_cond=[u, v], t_final=t_, num_points=2, P_gen=power_gen, D1=D1, B12_V1_V2=B12_V1_V2, m1=m1)
                t_test, delta_values, omega_values = solver.solve_system()
                # traj_pert = solver.solve(*sample_perturbed, t, n=2)
                F_pert = np.array([delta_values[-1], omega_values[-1]])
                # Finite difference derivative
                jacobian[:, i] = (F_pert - F_orig) / eps
            
            # Compute spectral norm of Jacobian
            _, s, _ = svd(jacobian)
            spectral_norm = max(s)
            if spectral_norm > max_norm:
                max_norm = spectral_norm
                    
        #elif solver_type == 'pinn':
            # samples_tensor = torch.tensor(samples, dtype=torch.float32, requires_grad=False)
            # for sample in samples_tensor:
            #     preds = model(sample)
            #     u0, v0, t = sample
            #     # Evaluate solver at original point (use n=2 for start and end points only)
            #     u_pred = u0 + preds[0] * t
            #     v_pred = v0 + preds[1] * t
            #     #traj = solver.solve_system(u0, v0, t, n=2)
            #     F_orig = np.array([u_pred.detach(), v_pred.detach()])  # State at t
            #     jacobian = np.zeros((2, 3))  # 2 outputs, 3 inputs
            #     for i in range(3):  # Perturb each input dimension
            #         sample_perturbed = sample.detach().clone()
            #         sample_perturbed[i] += eps
            #         u, v, t_ = sample_perturbed
            #         # Evaluate solver at perturbed point
            #         preds_ = model(sample_perturbed)
            #         u_pred_ = u + preds_[0] * t_
            #         v_pred_ = v + preds_[1] * t_
            #         # traj_pert = solver.solve(*sample_perturbed, t, n=2)
            #         F_pert = np.array([u_pred_.detach(), v_pred_.detach()])
            #         # Finite difference derivative
            #         jacobian[:, i] = (F_pert - F_orig) / eps
                
            #     # Compute spectral norm of Jacobian
            #     _, s, _ = svd(jacobian)
            #     spectral_norm = max(s)
            #     if spectral_norm > max_norm:
            #         max_norm = spectral_norm
        
        return max_norm
    
    def gradient_attack(self): 
        return 
        
    def every_call_counts(self): #https://github.com/fouratifares/ECP
        return

    def calculate_NTK(self): # petros work
        return

    def split_conformal_interface(self,cal_dataset):
        return