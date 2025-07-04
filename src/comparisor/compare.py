import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from torch_cka import CKA
from solvers import Solver

class Comparisor:
    """
    Compare two PyTorch models either by a simple weight metric or by CKA similarity.
    """
    def __init__(self,
                 model1: Solver,
                 model2: Solver,
                 model1_name: str = "Model1",
                 model2_name: str = "Model2",
                 metric_fn=None,
                #  cka_obj=None,
                 device=None):
        """
        Args:
            model1, model2: the two nn.Modules to compare.
            model1_name, model2_name: labels for plotting.
            metric_fn: function(w1, w2) -> scalar similarity (e.g., cosine). Ignored if cka_obj provided.
            cka_obj: a torch_cka.CKA instance; if given, uses CKA to compute similarity matrix.
            device: torch device to run comparisons on (defaults to model1's device).
        """
        self.model1 = model1.model
        self.model2 = model2.model
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.metric_fn = metric_fn
        self.cka_obj = cka = CKA(model1, model2,
          model1_name="ALL_LOSS",   # good idea to provide names to avoid confusion
          model2_name="ONLY_PHYSICS",
          device='cpu')

        if device:
            self.device = device
        else:
            try:
                self.device = next(model1.parameters()).device
            except StopIteration:
                self.device = torch.device('cpu')

        # collect weights if not using CKA
        if self.cka_obj is None:
            self.weights1 = self._collect_weights(model1)
            self.weights2 = self._collect_weights(model2)
            self.layer_names1 = list(self.weights1.keys())
            self.layer_names2 = list(self.weights2.keys())

    def _collect_weights(self, model: nn.Module):
        """Collect named weight tensors."""
        wdict = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                wdict[name] = module.weight.detach().to(self.device)
        return wdict

    def compute_similarity_matrix(self, loader1=None, loader2=None):
        """
        Returns a similarity matrix:
        - If cka_obj is set, runs CKA.compare on the provided dataloaders and returns its hsic matrix.
        - Otherwise, computes metric_fn between every pair of weight tensors.
        """
        if self.cka_obj:
            if loader1 is None:
                raise ValueError("Loader1 must be provided for CKA comparison.")
            # optionally support two loaders for two models
            self.cka_obj.compare(loader1, loader2)
            return self.cka_obj.hsic_matrix

        # simple weight-based
        n1, n2 = len(self.layer_names1), len(self.layer_names2)
        mat = torch.zeros(n1, n2, dtype=torch.float32)
        for i, n1_key in enumerate(self.layer_names1):
            w1 = self.weights1[n1_key]
            for j, n2_key in enumerate(self.layer_names2):
                w2 = self.weights2[n2_key]
                mat[i, j] = self.metric_fn(w1, w2)
        return mat.numpy()

    def plot(self,
             loader1=None,
             loader2=None,
             cmap='magma',
             title: str = None):
        """
        Plot the similarity matrix. If using CKA, pass in dataloaders.
        """
        sim_mat = self.compute_similarity_matrix(loader1, loader2)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(sim_mat, origin='lower', cmap=cmap)
        if self.cka_obj:
            # label as CKA
            ax.set_xticks(range(len(self.cka_obj.model2_info['Layers'])))
            ax.set_yticks(range(len(self.cka_obj.model1_info['Layers'])))
            ax.set_xticklabels(self.cka_obj.model2_info['Layers'], rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(self.cka_obj.model1_info['Layers'], fontsize=10)
            ax.set_xlabel(f"CKA Layers of {self.model2_name}", fontsize=12)
            ax.set_ylabel(f"CKA Layers of {self.model1_name}", fontsize=12)
        else:
            ax.set_xticks(range(len(self.layer_names2)))
            ax.set_yticks(range(len(self.layer_names1)))
            ax.set_xticklabels(self.layer_names2, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(self.layer_names1, fontsize=10)
            ax.set_xlabel(f"Layers of {self.model2_name}", fontsize=12)
            ax.set_ylabel(f"Layers of {self.model1_name}", fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        divider = axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.show()

# Define two random networks with identical architecture
net1 = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

net2 = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Instantiate and run the comparison
comp = Comparisor(net1, net2,
                  model1_name="RandomNet1",
                  model2_name="RandomNet2")

# Compute the similarity matrix (cosine by default)
sim_matrix = comp.compute_similarity_matrix()
print("Similarity matrix:\n", sim_matrix)

# Plot the heatmap
comp.plot(title="Cosine Similarity: RandomNet1 vs RandomNet2")
# Example usage with CKA:
# from torch_cka import CKA
# cka = CKA(model, model1,
#           model1_name="ALL_LOSS", model2_name="ONLY_PHYSICS",
#           device='cpu')
# comp = Comparisor(model, model1, cka_obj=cka)
# comp.plot(loader1=dataloader1)
