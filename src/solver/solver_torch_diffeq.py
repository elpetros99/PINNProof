import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from scipy.integrate import odeint
from torchdiffeq import odeint
import random

#import solver.solver as solver

class GeneralODESolver():
    """
    A general-purpose ODE solver wrapper around scipy.integrate.odeint.
    
    You supply:
      • func       : a callable f(Y, t, *args, **kwargs) that returns dY/dt
      • ini_cond   : list or array of initial values [y1(0), y2(0), ..., yN(0)]
      • t_final    : final time up to which you want the solution
      • num_points : number of time points between 0 and t_final
      • args       : any extra positional arguments needed by func
      • kwargs     : any extra keyword arguments needed by func
    
    After initialization, calling solve() returns (t, Y):
      • t is a 1D numpy array of length num_points
      • Y is a 2D numpy array of shape (num_points, len(ini_cond)),
        where each column of Y corresponds to one dependent variable.
    """

    def __init__(self, func, *args, **kwargs):
        # Save inputs
        self.func = func
        # self.ini_cond   = np.array(ini_cond, dtype=float)
        # self.t_final    = float(t_final)
        # self.num_points = int(num_points)
        # Any extra positional args/keyword args for your ODE function
        self.args = args
        self.kwargs = kwargs

    def solve(self, ini_cond, t_final, num_points):
        """
        Solve dY/dt = func(Y, t, *args, **kwargs)
        from t = 0 to t = t_final, with initial state = self.ini_cond.
        Returns:
          t : np.ndarray, shape (num_points,)
          Y : np.ndarray, shape (num_points, len(ini_cond))
        """

        t = torch.linspace(0.0, t_final, num_points, dtype=torch.float32)
        ini_cond = torch.tensor(ini_cond, dtype=torch.float32)
        f_wrapped = lambda t, y: self.func(t, y, *self.args)
        # Call odeint with our stored function + parameters
        sol = odeint(f_wrapped, ini_cond, t)

        return t, sol
    
    

# if __name__ == "__main__":
    # Example usage
def my_system(t, Y, D1, B12, m1, P_gen):
    # Y     is a 1D array of length N (number of state variables)
    # t is the current time
    # args, kwargs are extra parameters you pass in through the solver
    #
    # You must return a length‐N array: dY/dt
    #
    # Example for a 2‐state system (Y = [δ, ω]):
    delta, omega = Y
#    D1, B12, m1, P_gen = args  # or from kwargs, whichever you prefer
    dδ_dt = omega
    dω_dt = ( -D1*omega - B12 * np.sin(delta) + P_gen ) / m1
    return torch.stack([dδ_dt, dω_dt])


# Example initial conditions for δ(0) and ω(0):
ini_cond = [0.1, 0.0]     # δ₀ = 0.1 rad, ω₀ = 0.0 rad/s

# Example parameter values:
D1 = 0.5
B12 = 1.2
m1 = 0.8
P_gen = 0.3

# Total simulation time and number of points
t_final    = 10.0     # simulate up to t = 10 seconds
num_points = 1000     # sample 1000 points between 0 and 10

# Pass the parameters as *args in the same order you unpack inside my_system:
solver = GeneralODESolver(
    my_system,
    # ini_cond = ini_cond,
    # t_final = t_final,
    # num_points = num_points,
    D1, B12, m1, P_gen)

t, solution = solver.solve(ini_cond=ini_cond, t_final=t_final, num_points=num_points)

import matplotlib.pyplot as plt

δ_vals = solution[:, 0]
ω_vals = solution[:, 1]

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(t, δ_vals)
plt.title("δ(t)")
plt.xlabel("Time [s]")
plt.ylabel("δ [rad]")

plt.subplot(1,2,2)
plt.plot(t, ω_vals)
plt.title("ω(t)")
plt.xlabel("Time [s]")
plt.ylabel("ω [rad/s]")
plt.tight_layout()
plt.show()


    # `solution` is a (1000 × 2) array; 
    #   solution[:,0] = δ(t) at each time, 
    #   solution[:,1] = ω(t) at each time.
    #   solution[:,1] = ω(t) at each time.

