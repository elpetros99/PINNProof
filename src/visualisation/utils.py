import torch
import pandas as pd

def split_data(dataset):
    data = torch.Tensor(dataset)
    x_list, y_list = [], []

    for sample in data:
        # no need to re‐wrap sample in torch.tensor; it's already a Tensor
        # sample shape: (M, N) for instance
        y = sample[1:].T                 # shape (N, M-1)
        x = sample.T                     # shape (N, M)
        x[:, 1:] = x[0, 1:]              # broadcast first row
        x.requires_grad_(True)
        y.requires_grad_(True)

        x_list.append(x)
        y_list.append(y)

    # one concat per axis
    x_train = torch.cat(x_list, dim=0)
    y_train = torch.cat(y_list, dim=0)

    return x_train, y_train


