# -*- coding: utf-8 -*-
"""
Created on Wed May 14 03:13:55 2025

@author: INDRAJIT
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import odeint
import random
import torchode


class Solver():
    def __init__(self, ini_cond, t_final, num_points, P_gen, D1, B12_V1_V2, m1):
        self.t_final = t_final
        self.num_points = num_points
        self.delta0 = ini_cond[0]
        self.ddelta_dt0 = ini_cond[1]  #omega0
        self.P_gen = P_gen
        self.D1 = D1
        self.B12_V1_V2 = B12_V1_V2
        self.m1 = m1    
    def system_diff_eq(self, Y, t):
        # this function takes as input the current state of the variables, and outputs their derivatives w.r.t t
        delta, omega = Y
        ddelta_dt = omega
        domega_dt = (-(self.D1)*omega - (self.B12_V1_V2)*np.sin(delta) + self.P_gen) / self.m1 
        return [ddelta_dt, domega_dt]
    def solve_system(self):
        t = np.linspace(0, self.t_final, self.num_points) 
        solution = odeint(self.system_diff_eq, [self.delta0, self.ddelta_dt0], t)    # using 'odeint' to obtain actual values of δ and ω
        delta = solution[:, 0]
        omega = solution[:, 1]
        return t, delta, omega


class Solver_torchode():
    def __init__(self, ini_cond, t_final, num_points, P_gen, D1, B12_V1_V2, m1):
        self.t_final = t_final
        self.num_points = num_points
        self.delta0 = ini_cond[0]
        self.ddelta_dt0 = ini_cond[1]  #omega0
        self.P_gen = P_gen
        self.D1 = D1
        self.B12_V1_V2 = B12_V1_V2
        self.m1 = m1
    def system_diff_eq(self, t, y):
        # delta, omega = y[..., 0], y[..., 1]
        delta, omega = y.unbind(-1)
        ddelta_dt = omega
        domega_dt = (-(self.D1)*omega - (self.B12_V1_V2)*torch.sin(delta) + self.P_gen) / self.m1 
        return torch.stack([ddelta_dt, domega_dt], dim=-1)
    def solve_system(self):
        self.D1_t = torch.tensor(self.D1, dtype=torch.float32)
        self.B12_V1_V2_t = torch.tensor(self.B12_V1_V2, dtype=torch.float32)
        self.m1_t = torch.tensor(self.m1, dtype=torch.float32)
        self.P_gen_t = torch.tensor(self.P_gen, dtype=torch.float32)
        t_eval = torch.linspace(0, self.t_final, self.num_points).unsqueeze(0)
        y0 = torch.tensor([[self.delta0, self.ddelta_dt0]], dtype=torch.float32)
        
        # Create the problem
        term = torchode.ODETerm(self.system_diff_eq)
        step_method = torchode.Tsit5(term=term)
        step_size_controller = torchode.IntegralController(atol=1e-6, rtol=1e-3, term=term)
        solver = torchode.AutoDiffAdjoint(step_method, step_size_controller)
        jit_solver = torch.compile(solver)
        
        sol = jit_solver.solve(torchode.InitialValueProblem(y0=y0, t_eval=t_eval))
        t = sol.ts[0].numpy()
        delta = sol.ys[0,:,0].numpy()
        omega = sol.ys[0,:,1].numpy()
        
        # solution = sol.evaluate(t_eval)
        # t = t_eval.numpy()
        # delta = solution[..., 0].numpy()
        # omega = solution[..., 1].numpy()
        return t, delta, omega
        
deltas_check = [0.1, 0.225, 0.47, 0.64, 0.85] 
omegas_check = [-0.2, 0.1, 0.3, 0.4]  
test_time = 10                         

solver = Solver(ini_cond=[deltas_check[i], omegas_check[j]], t_final=test_time, num_points=200000000, P_gen=power_gen, D1=D1, B12_V1_V2=B12_V1_V2, m1=m1)
t_test, delta_values, omega_values = solver.solve_system()

solver_torchode = Solver_torchode(ini_cond=[deltas_check[i], omegas_check[j]], t_final=test_time, num_points=200000000, P_gen=power_gen, D1=D1, B12_V1_V2=B12_V1_V2, m1=m1)
t_test_1, delta_values_1, omega_values_1 = solver_torchode.solve_system()


class Normalization_strat(nn.Module):
    def __init__(self, tensor_range):
        super(Normalization_strat, self).__init__()
        self.tensor_range = tensor_range
    def forward(self, x):
        return (x/(self.tensor_range+1e-5))
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, range_input):
        super().__init__()
        activation = nn.Tanh   
        torch.manual_seed(123) 
        torch.use_deterministic_algorithms(True)
        random.seed(123)
        np.random.seed(123)
        self.norm = Normalization_strat(range_input.clone().detach()) 
        self.fcs = nn.Sequential(*[self.norm,
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        nn.init.xavier_normal_(self.fcs[1].weight)
        for module in self.fch:
            if isinstance(module[0], nn.Linear):
                nn.init.xavier_normal_(module[0].weight)
        nn.init.xavier_normal_(self.fce.weight)
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

num_points_solver = 201
num_collocation_points = 11
range_training_time =  2
power_gen = 0.10525252525252525
m1 = 0.4
D1 = 0.1
B12_V1_V2 = 0.2

number_initial_deltas = 10   
range_initial_deltas = [0, 1]
number_initial_omegas = 10   
range_initial_omegas = [-0.5, 0.5]
step_col_points = range_training_time/(num_collocation_points-1)    
step_initial_deltas = (range_initial_deltas[1] - range_initial_deltas[0]) / (number_initial_deltas - 1)
step_initial_omegas = (range_initial_omegas[1] - range_initial_omegas[0]) / (number_initial_omegas - 1)

num_neurons = 64
num_layers = 3
input_neurons = 3   
output_neurons = 2
num_training_epochs = 12000  
recording_step = 500
learning_rate = 1e-3 
gamma = 0.9996

delta_values = np.arange(range_initial_deltas[0], range_initial_deltas[1]+step_initial_deltas, step_initial_deltas)
omega_values = np.arange(range_initial_omegas[0], range_initial_omegas[1]+step_initial_omegas, step_initial_omegas)
t_values = np.arange(0, range_training_time+step_col_points, step_col_points)
delta_rep = torch.tensor(np.repeat(delta_values, len(omega_values)*len(t_values))).float().view(-1,1)
omega_rep = torch.tensor(np.tile(np.repeat(omega_values, len(t_values)), len(delta_values))).float().view(-1,1)
t_rep = torch.tensor(np.tile(t_values, len(delta_values) * len(omega_values))).float().view(-1,1).requires_grad_(True)
phy_input = torch.cat([delta_rep, omega_rep, t_rep], dim=1)  
range_input = torch.max(phy_input, dim=0)[0] - torch.min(phy_input, dim=0)[0]  

lambda_l1 = 1 
lambda_l2 = 3
PINN = FCN(input_neurons, output_neurons, num_neurons, num_layers, range_input)  
torch.manual_seed(123)  
torch.use_deterministic_algorithms(True)
random.seed(123)
np.random.seed(123)
loss_history = []
optimizer = torch.optim.Adam(PINN.parameters(),lr=learning_rate) 
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) 


for i in range(num_training_epochs):
    optimizer.zero_grad()
    preds = PINN(phy_input)
    delta_hat = delta_rep + preds[:, 0:1]*t_rep
    omega_hat = omega_rep + preds[:, 1:2]*t_rep
    ddelta_dt = torch.autograd.grad(delta_hat, t_rep, torch.ones_like(delta_hat), create_graph=True)[0]
    domega_dt = torch.autograd.grad(omega_hat, t_rep, torch.ones_like(omega_hat), create_graph=True)[0]
    loss1 = torch.mean((m1*domega_dt + D1*omega_hat + B12_V1_V2*torch.sin(delta_hat) - power_gen)**2)
    loss2 = torch.mean((omega_hat-ddelta_dt)**2)
    loss = lambda_l1*loss1 + lambda_l2*loss2  
    loss_history.append(loss.detach())
    loss.backward()
    optimizer.step()
    scheduler.step()
    if i % recording_step == 0:
        print(f'{i}\t/{num_training_epochs} : \tTotal Loss (loss1+loss2) = {loss.item():.8f}')
print(f'Final Loss at End of Training = {loss_history[-1]:.8f}')


torch.save(PINN.state_dict(), 'pinn_model.pth')
PINN = FCN(input_neurons, output_neurons, num_neurons, num_layers, range_input)
PINN.load_state_dict(torch.load('pinn_model.pth'))
PINN.eval()

deltas_check = [0.1, 0.225, 0.47, 0.64, 0.85] 
omegas_check = [-0.2, 0.1, 0.3, 0.4]  
test_time = 3
colors1 = plt.cm.rainbow((np.linspace(min(deltas_check), max(deltas_check), len(deltas_check)) - min(deltas_check)) / (max(deltas_check) - min(deltas_check)))
colors2 = plt.cm.rainbow((np.linspace(min(omegas_check), max(omegas_check), len(omegas_check)) - min(omegas_check)) / (max(omegas_check) - min(omegas_check)))
fig = plt.figure(figsize=(24, 8))
gs = GridSpec(1, 2, figure=fig)
ax1= fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
delta_ini = np.linspace(range_initial_deltas[0], range_initial_deltas[1], number_initial_deltas)   
omega_ini = np.linspace(range_initial_omegas[0], range_initial_omegas[1], number_initial_omegas)   
for i in range(len(deltas_check)):    
    c1 = colors1[i]
    for j in range (len(omegas_check)):
        c2 = colors2[j]
        l1a = '_' if j>0 else f'Actual Trajectories for different ω_0 values, when: δ_0 = {delta_ini[i]:.3f}'
        l2a = '_' if i>0 else f'Actual Trajectories for different δ_0 values, when: ω_0 = {omega_ini[j]:.3f}'
        solver = Solver(ini_cond=[deltas_check[i], omegas_check[j]], t_final=test_time, num_points=num_points_solver, P_gen=power_gen, D1=D1, B12_V1_V2=B12_V1_V2, m1=m1)
        t_test, delta_values, omega_values = solver.solve_system()
        ax1.plot(t_test, delta_values, c=c1, linestyle='--', label=l1a)
        ax2.plot(t_test, omega_values, c=c2, linestyle='--', label=l2a)
        delta_ini_check = torch.full((num_points_solver,1), deltas_check[i])
        omega_ini_check = torch.full((num_points_solver,1), omegas_check[j])
        t_test_torch = torch.linspace(0,test_time,num_points_solver).view(-1,1).requires_grad_(True)
        test_phy = torch.cat([delta_ini_check, omega_ini_check, t_test_torch], dim=1)
        preds = PINN(test_phy)
        delta_pred = delta_ini_check + preds[:, 0:1]*t_test_torch
        omega_pred = omega_ini_check + preds[:, 1:2]*t_test_torch
        delta_pred_array = delta_pred.detach().numpy().flatten()
        omega_pred_array = omega_pred.detach().numpy().flatten()
        l1p = '_' if j>0 else f'Predicted Trajectories for different ω_0 values, when: δ_0 = {delta_ini[i]:.3f}'
        l2p = '_' if i>0 else f'Predicted Trajectories for different δ_0 values, when: ω_0 = {omega_ini[j]:.3f}'
        ax1.plot(t_test, delta_pred_array, c=c1, label=l1p)
        ax2.plot(t_test, omega_pred_array, c=c2, label=l2p)
ax1.set_xlabel('t')
ax2.set_xlabel('t')
ax1.set_ylabel('\u03B4(t)')
ax2.set_ylabel('\u03C9(t)')
ax1.grid(True)
ax1.axhline(y=0, color='k', lw=1.5)
ax1.axvline(x=0, color='k', lw=1.5)
ax2.grid(True)
ax2.axhline(y=0, color='k', lw=1.5)
ax2.axvline(x=0, color='k', lw=1.5)
ax1.scatter(np.linspace(0, range_training_time, num_collocation_points+1), np.zeros(num_collocation_points+1), label='Collocation Point Locations on t-axis', color='black', s=40)
ax2.scatter(np.linspace(0, range_training_time, num_collocation_points+1), np.zeros(num_collocation_points+1), label='Collocation Point Locations on t-axis', color='black', s=40)
ax1.axvline(x=2, color='k', lw=1.5)
ax2.axvline(x=2, color='k', lw=1.5)
ax1.legend()
ax2.legend()
ax1.set_title(f'Predicted vs Actual trajectory of δ over time, till t={test_time:.2f} for different Initial Values of δ_0')
ax2.set_title(f'Predicted vs Actual trajectory of ω over time, till t={test_time:.2f} for different Initial Values of ω_0')
plt.tight_layout()
plt.show()




# considering coefficients as 1 and constants as 0
def jacobian_true(u, v):
    # Jacobian of the system
    # F = [v, -v - sin(u)]
    # dF/du = [0, -cos(u)]
    # dF/dv = [1, -1]
    return np.array([[0.0, 1.0],
                     [-np.cos(u)/2, -3.0/8]])
def estimate_lipschitz_true(domain_bounds, num_samples=2000):
    max_singular_value = 0.0
    u_bounds, v_bounds = domain_bounds
    for _ in range(num_samples):
        u = np.random.uniform(u_bounds[0], u_bounds[1])
        v = np.random.uniform(v_bounds[0], v_bounds[1])
        # Jacobian at (u, v)
        J = jacobian_true(u, v)
        # Compute operator norm (largest singular value)
        sigma_max = np.linalg.norm(J, ord=2)
        max_singular_value = max(max_singular_value, sigma_max)
    return max_singular_value
bounds = np.array([
    [0, 2*np.pi],  # u0
    [-5.0, 5.0],  # v0
])
L = estimate_lipschitz_true(bounds, num_samples=2000)
print("Lipschitz constant (PINN):", L)


def compute_lipschitz_constant(pinn_model, domain_bounds, num_samples=1000, device='cpu'):
    pinn_model = pinn_model.to(device)
    max_singular_value = 0.0
    for _ in range(num_samples):
        # Sample a point (u0, v0, t)
        x = torch.FloatTensor(1, 3).uniform_(0, 1).to(device)
        x = domain_bounds[:, 0] + x * (domain_bounds[:, 1] - domain_bounds[:, 0])
        x.requires_grad_(True)
        u0, v0, t = x[0, 0], x[0, 1], x[0, 2]

        # Forward pass through the PINN
        A, B = pinn_model(x)[0]
        u = u0 + A * t
        v = v0 + B * t
        output = torch.stack([u, v])

        # Compute Jacobian ∂(u,v)/∂(u0,v0,t)
        J = []
        for i in range(2):  # for u and v
            grad = torch.autograd.grad(outputs=output[i], inputs=x,
                                       retain_graph=True, create_graph=True, allow_unused=True)[0]
            J.append(grad)
        J = torch.stack(J)  # Shape: (2, 3)

        # Compute operator norm (largest singular value)
        svd = torch.linalg.svdvals(J)
        sigma_max = svd.max().item()
        max_singular_value = max(max_singular_value, sigma_max)

    return max_singular_value
bounds = torch.tensor([
    [0, 2*np.pi],  # u0
    [-5.0, 5.0],  # v0
    [0.0, 2.0],  # t
])

L = compute_lipschitz_constant(PINN, bounds, num_samples=2000)
print("Lipschitz constant (PINN):", L)








# for pinn - code 1
def estimate_lipschitz_pinn(pinn_model, u_vals, v_vals, t_vals):
    lipschitz_vals = []
    for u0 in u_vals:
        for v0 in v_vals:
            for t in t_vals:
                inp = torch.tensor([[u0, v0, t]], dtype=torch.float32, requires_grad=True)
                def full_output(x):
                    u0_, v0_, t_ = x[:, 0:1], x[:, 1:2], x[:, 2:3]
                    uv_hat = PINN(x)  # shape: [1, 2]
                    u_t = u0_ + t_ * uv_hat[:, 0:1]
                    v_t = v0_ + t_ * uv_hat[:, 1:2]
                    return torch.cat([u_t, v_t], dim=1)
                J = torch.autograd.functional.jacobian(full_output, inp)
                J = J.squeeze(0).squeeze(1).detach().numpy()  # shape (2, 3)
                L = np.linalg.norm(J, ord=2)  # Spectral norm
                lipschitz_vals.append(L)
    return max(lipschitz_vals)
u_vals = np.linspace(0, 2*np.pi, 10)
v_vals = np.linspace(-5, 5, 10)
t_vals = np.linspace(0, 2, 20)
L_pinn = estimate_lipschitz_pinn(PINN, u_vals, v_vals, t_vals)
print("Lipschitz constant (PINN):", L_pinn)




# for pinn - code 2
def estimate_pinn_lipschitz(pinn, num_samples=1000, u0_range=(0,2*np.pi), v0_range=(-5,5), t_range=(0,2)):
    pinn.eval()
    max_sv = 0.0
    for _ in range(num_samples):
        u0 = torch.FloatTensor(1).uniform_(*u0_range)
        v0 = torch.FloatTensor(1).uniform_(*v0_range)
        t = torch.FloatTensor(1).uniform_(*t_range)
        inputs = torch.cat([u0, v0, t], dim=0).unsqueeze(0)
        inputs.requires_grad_(True)
        def transformed_output(x):
            delta = pinn(x)  # PINN outputs [delta_u, delta_v]
            u_pred = x[:, 0] + delta[:, 0] * x[:, 2]  # u0 + delta_u * t
            v_pred = x[:, 1] + delta[:, 1] * x[:, 2]  # v0 + delta_v * t
            return torch.stack([u_pred, v_pred], dim=1)
        # Compute Jacobian of the transformed outputs w.r.t. inputs
        J = torch.autograd.functional.jacobian(transformed_output, inputs)
        J = J.squeeze(0).squeeze(0).detach().numpy()  # Shape: [2, 3]
        # Compute largest singular value
        _, singular_values, _ = np.linalg.svd(J, full_matrices=False)
        current_max_sv = singular_values[0]
        max_sv = max(max_sv, current_max_sv)
    return max_sv

pinn_lipschitz = estimate_pinn_lipschitz(PINN, num_samples=2000)
print("Lipschitz constant (PINN):", pinn_lipschitz)






















