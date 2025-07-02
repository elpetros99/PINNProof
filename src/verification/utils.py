<<<<<<< HEAD
=======
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 14:49:10 2025

@author: INDRAJIT
"""

>>>>>>> master
import numpy as np
import autolirpa
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from autolirpa.operators import JacobianOP

def sampling_domain(dimensions_list, num_points): # reimplement
    """
    Sample points uniformly in a multi-dimensional space defined by dimensions_list.
    Each dimension in dimensions_list is a tuple (min, max) defining the range for that dimension.
    The function returns a 2D numpy array where each row is a point in the sampled space.
    """
    if not dimensions_list:
        raise ValueError("dimensions_list must not be empty")

    # Create a grid of points for each dimension
    grids = [np.linspace(dim[0], dim[1], num_points) for dim in dimensions_list]

    # Create a meshgrid and stack the points
    mesh = np.meshgrid(*grids, indexing='ij')
    sampled_points = np.vstack(map(np.ravel, mesh)).T

    return sampled_points

def local_lipschitz_via_jacobian(model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
    """
    Computes the local Lipschitz constant at x0 by:
      1) Forwarding x0 through `model` to get y = f(x0).
      2) Using JacobianOP.apply(y, x0) to obtain J ∈ ℝ^{out_dim × in_dim}.
      3) Returning σ_max(J), the largest singular value of J (i.e. operator norm).

    Args:
        model:  A PyTorch nn.Module (any architecture).
        x0:     A single input tensor of shape [1, in_dim], with requires_grad=True.

    Returns:
        A tensor of shape [1], containing σ_max(J_f(x0)).  This is the local Lipschitz
        (ℓ₂→ℓ₂) at x0.
    """
    # 1) Forward to get output y.  Ensure x0.requires_grad=True.
    y = model(x0)  # y.shape = [1, out_dim]

    # 2) Compute the full Jacobian J of shape [1, out_dim, in_dim]
    J_full = JacobianOP.apply(y, x0)  # shape = [batch=1, out_dim, in_dim]

    # 3) Slice off the batch dimension:
    J = J_full[0]  # shape = [out_dim, in_dim]

    # 4) Compute singular values of J, pick the largest:
    #    torch.linalg.svdvals returns a 1‐D tensor of singular values in descending order.
    s_vals = torch.linalg.svdvals(J)  # shape = [min(out_dim, in_dim)]
    sigma_max = s_vals[0]             # largest singular value

    return sigma_max.unsqueeze(0)     # return as shape [1]

# -------------------------
# Example usage:

if __name__ == "__main__":
    # 1) Define a simple net (or load your pretrained model):
    class MyNet(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super(MyNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )

        def forward(self, x):
            return self.net(x)

    # 2) Instantiate and (optionally) load weights:
    in_dim = 8
    hidden_dim = 32
    out_dim = 4
    model = MyNet(in_dim, hidden_dim, out_dim)

    # (If you have saved weights: model.load_state_dict(torch.load("...")))

    # 3) Pick a point x0 ∈ ℝ^{1×in_dim} with requires_grad=True:
    x0 = torch.randn(1, in_dim, requires_grad=True)

    # 4) Compute local Lipschitz at x0:
    L_local = local_lipschitz_via_jacobian(model, x0)
    print(f"Local Lipschitz at x0 (σ_max of J) = {L_local.item():.6f}")

# def find_lipchitz_NN_autolirpa(NN):
#     class JVPWrapper(nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model


#         def forward(self, x):
#             y = self.model(x)

#             jacobian = JacobianOP.apply(y, x)#.flatten(2)

#             jvp1 = jacobian[:,:,3]

#             return jvp1 #self.fixed_weights_layer(y)#jvp1 #jvp1 #jacobian#jvp
<<<<<<< HEAD
#     return
=======
#     return
>>>>>>> master
