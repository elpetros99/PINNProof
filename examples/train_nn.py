# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 14:49:49 2025

@author: INDRAJIT
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

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