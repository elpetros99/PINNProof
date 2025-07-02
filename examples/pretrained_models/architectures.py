
import torch
import torch.nn as nn
#%cd /teamspace/studios/this_studio/Physics-Informed-Neural-Networks-for-Synchronous-Machine-Models
import os

class CosineActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Network(nn.Module):
    """
    A class to represent a dynamic neural network model with dynamic number of layers based on the respective argument.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers,activation="tanh"):
        super(Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        if activation=="swish":
            self.activation = SwishActivation()#CosineActivation #nn.Tanh
        elif activation=="cos":
            self.activation = CosineActivation() #nn.Tanh
        elif activation=="tanh":
            self.activation = nn.Tanh()
        self.hidden = []
        self.hidden.append(nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Forward pass of the dynamic neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        # x = nn.Flatten(x)            # e.g. [B, 9] → [B,9] or [B,C,H,W]→[B,C*H*W]
        for i in range(self.num_layers):
            x = self.activation(self.hidden[i](x))
        x = self.output(x)
        return x