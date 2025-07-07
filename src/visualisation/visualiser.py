import math
import copy

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1, mplot3d
from sklearn.decomposition import PCA

# import utils
from utils import split_data

from tqdm import tqdm


# matplotlib.rcParams['figure.figsize'] = [18, 12]

# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics
import torchlens as tl

class visualizer():
    '''A class to visualize the results of a simulation.
    This class is used to visualize the results of a simulation.
    it takes a solver object as input and uses it to visualize the results.
    It can be used to visualize the results of a simulation in a variety of ways.
    '''
    def __init__(self,solver):
        self.solver = solver
        self.fig = None
        self.ax = None
        self.fig2 = None
        self.model_h = None
        self.A= None # matrix with flattened activation values for many samples
        self.svd_matrices=None # svd matrices for A in the form of (U, S, Vt)
    
    def plot(self):
        '''Plot the results of the simulation.'''
        if self.fig is None:
            import matplotlib.pyplot as plt
            self.fig, self.ax = plt.subplots()
        
        self.ax.clear()
        self.ax.plot(self.solver.time, self.solver.results)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Results')
        self.ax.set_title('Simulation Results')
        
        plt.draw()
        plt.pause(0.01)

    def plot_weights(self, bins=50, ncols=3):
        """
        Plots weight-value histograms for each trainable parameter tensor
        in a neat grid.

        Args:
            named_parameters: iterable of (name, param) — e.g. model.named_parameters()
            bins:            number of bins in each histogram
            ncols:           how many histogram columns you want
        """
        # 1) Gather weight arrays
        named_parameters = self.solver.model.named_parameters()
        
        weight_params = [(name, param.detach().cpu().view(-1).numpy())
                        for name, param in named_parameters if param.requires_grad]

        n_weights = len(weight_params)
        nrows = math.ceil(n_weights / ncols)

        # 2) Create subplots grid
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3))
        axes = axes.flatten()

        # 3) Plot each histogram
        for ax, (name, data) in zip(axes, weight_params):
            ax.hist(data, bins=bins, alpha=0.7)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel('weight value', fontsize=8)
            ax.set_ylabel('frequency', fontsize=8)
            ax.tick_params(axis='both', labelsize=7)

        # 4) Turn off any extra axes
        for ax in axes[n_weights:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()


    def save_activations_grads(self, dataset):
        x, y = self.solver.split_data(dataset)

        model_history = tl.log_forward_pass(self.solver.model, x.detach().requires_grad_(True), detach_saved_tensors=False, save_gradients=True)
        loss = ((model_history[-1].tensor_contents-y[:,:-self.solver.control_variables].detach().clone())**2).mean()
        loss.backward()

    def plot_latent_space_pca_2d(self, dataset, layers_to_print=[], grad=False):
        if self.model_h == None:
            raise Exception("Oops, you should first run the save_activations_grads function")
        
        x, y = self.solver.split_data(dataset)
        if grad==False:
            features_dict = {
                layer: self.model_h[layer].tensor_contents.detach().cpu().view(x.size(0), -1).numpy()
                for layer in layers_to_print
            }
        else:
            features_dict = {
                layer: self.model_h[layer].grad_contents.detach().cpu().view(x.size(0), -1).numpy()
                for layer in layers_to_print
            }
        cmap='viridis'
        layers = list(features_dict.keys())
        num_layers = len(layers)
        num_targets = y.shape[1]
        print(num_targets)

        fig, axes = plt.subplots(num_targets, num_layers, figsize=(5 * num_layers, 4 * num_targets), squeeze=False)

        for j, (layer, features) in enumerate(features_dict.items()):
            reduced = PCA(n_components=2).fit_transform(features)

            for i in range(num_targets):
                ax = axes[i, j]
                scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=y[:, i], cmap=cmap, s=10)
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label(f"y[:, {i}]")

                ax.set_title(f"{layer}")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")

        plt.tight_layout()
        plt.show()

    def plot_latent_space_pca_3d(self, dataset, layers_to_print=[], feature_names=[], grad=False):
        if self.model_h == None:
            raise Exception("Oops, you should first run the save_activations_grads function")
        
        x, y = self.solver.split_data(dataset)

        if grad==False:
            features_dict = {
                layer: self.model_h[layer].tensor_contents.detach().cpu().view(x.size(0), -1).numpy()
                for layer in layers_to_print
            }
        else:
            features_dict = {
                layer: self.model_h[layer].grad_contents.detach().cpu().view(x.size(0), -1).numpy()
                for layer in layers_to_print
            }
        cmap='viridis'
        layers = list(features_dict.keys())
        num_layers = len(layers)
        num_targets = y.shape[1]

        fig = plt.figure(figsize=(5 * num_layers, 5 * num_targets))

        for layer_idx, (layer, features) in enumerate(features_dict.items()):
            pca = PCA(n_components=3)
            reduced = pca.fit_transform(features)

            for target_idx in range(num_targets):
                row = target_idx
                col = layer_idx
                ax = fig.add_subplot(num_targets, num_layers, row * num_layers + col + 1, projection='3d')
                scatter = ax.scatter(
                    reduced[:, 0], reduced[:, 1], reduced[:, 2],
                    c=y[:, target_idx], cmap=cmap, s=10
                )
                cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)

                cbar.set_label(feature_names[target_idx])

                ax.set_title(f"{layer}")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
        plt.tight_layout()
        plt.show()


    def print_pca_contributions(self, dataset, layers_to_print=[], feature_names=[], top_k=5, pc_components=3):
        if self.model_h == None:
            raise Exception("Oops, you should first run the save_activations_grads function")
        
        x, y = self.solver.split_data(dataset)

        features_dict = {
            layer: self.model_h[layer].tensor_contents.detach().cpu().view(x.size(0), -1).numpy()
            for layer in layers_to_print
        }
        cmap='viridis'
        layers = list(features_dict.keys())
        num_layers = len(layers)
        num_targets = y.shape[1]

        # fig = plt.figure(figsize=(5 * num_layers, 5 * num_targets))
        print("\n=== Principal Component Contributions (top {} weights) ===".format(top_k))

        for layer_idx, (layer, features) in enumerate(features_dict.items()):
            pca = PCA(n_components=3)
            reduced = pca.fit_transform(features)
            components = pca.components_  # shape (3, num_features)

                # Print top contributors to each PC
                # print(f"\nLayer: {layer} | Target y[:, {target_idx}]")
            # for pc_idx in range(3):
            #     top_indices = np.argsort(np.abs(components[pc_idx]))[::-1][:top_k]
            #     print(f"  PC{pc_idx+1} top-{top_k} contributing feature indices: {top_indices}")
            # feature_names_dict = {
            #     "input_1": ["t","theta","omega","E_d_dash","E_q_dash","P_m","Vs","theta_vs"],
            #     "tanh_1_2": [f"h1_{i}" for i in range( features_dict["tanh_1_2"].shape[1] )],
            #     "tanh_2_4": [f"h2_{i}" for i in range( features_dict["tanh_2_4"].shape[1] )],
            #     "tanh_3_6": [f"h2_{i}" for i in range( features_dict["tanh_3_6"].shape[1] )],
            #     "linear_4_7": [f"h2_{i}" for i in range( features_dict["linear_4_7"].shape[1] )],
            #     # …and so on for each layer
            # }

            feature_names_dict ={}
            # Grab the first layer’s key and copy in your original feature_names
            first_layer = layers_to_print[0]
            feature_names_dict[first_layer] = feature_names

            # Now handle the rest of the layers by name
            for layer_name in layers_to_print[1:]:
                # Get the number of features for *this* layer from your features_dict
                num_feats = features_dict[layer_name].shape[1]
                # Build names like "h_tanh1_2_0", "h_tanh1_2_1", …
                feature_names_dict[layer_name] = [
                    f"h_{layer_name}_{j}" for j in range(num_feats)
                ]
            # ["tanh_1_2","tanh_2_4","tanh_3_6", "linear_4_7","input_1"]
            for pc_idx in range(pc_components):
                # 1) pick the top k by absolute value
                top_indices = np.argsort(np.abs(components[pc_idx]))[::-1][:top_k]
                # 2) grab their signed loadings
                top_loadings = components[pc_idx, top_indices]
                # 3) map to names
                names = feature_names_dict[layer]
                top_names = [names[i] for i in top_indices]

                # top_names = [feature_names[i] for i in top_indices]
                # 4) format strings like "theta(+0.352)"
                entries = [f"{name}({loading:+.3f})"
                        for name, loading in zip(top_names, top_loadings)]
                print(f"  PC{pc_idx+1} top-{top_k}: " + ", ".join(entries))

    def print_model_structure(self,dataset):
        x, y = self.solver.split_data(dataset)

        model_history = tl.log_forward_pass(self.solver.model, x.detach().clone().requires_grad_(False), vis_opt='rolled')


    def plot_interface_pca(self, dataset, what_to_plot="current_mag_error"):

        x, y = split_data(dataset)
        y_pred = self.solver.model(x)

        I_D_pred,I_Q_pred= self.solver.interface_func(x, y_pred)
        I_D_solv,I_Q_solv= self.solver.interface_func(x, y)

        # 1. Your data and labels as PyTorch tensors
        #    x_train_list: shape (n_samples, n_features)
        #    I_D:        shape (n_samples,)  (float tensor)

        # 2. Convert to NumPy
        X = x.detach().cpu().numpy()
        # if what_to_plot == "current_mag_error":
        lab = I_D_pred - I_D_solv #.shape
        text="I_D error"
        if what_to_plot == "current_ang_error":
            lab = I_Q_pred - I_Q_solv
            text="I_Q error"
        labels = lab.detach().cpu().numpy()

        # 3. Fit PCA to 2 dimensions
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X)  # shape (n_samples, 2)

        # 4. Plot, coloring by your continuous labels
        plt.figure(figsize=(8,6))
        sc = plt.scatter(
            reduced[:, 0], 
            reduced[:, 1], 
            c=labels, 
            cmap='viridis',    # or any other Matplotlib colormap
            s=30,              # marker size
            alpha=0.8, 
            linewidths=0
        )
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('2D PCA of x_train_list colored by I_D')

        # 5. Colorbar shows the text → color mapping
        cbar = plt.colorbar(sc)
        cbar.set_label(text+ ' value')

        plt.tight_layout()
        plt.show()


    def plot_activation_grid(self, layers_to_print=[], sample_indices=[], 
                            ncols=None, figsize_per_plot=(5, 4),
                            cmap='viridis', interpolation='none'):
        """
        Plots activation heatmaps for the given layers and sample indices.
        
        Parameters
        ----------
        layers : list of str
            The layer names to include (rows of each heatmap).
        sample_indices : list of int
            The sample indices to plot; one subplot per index.
        model_history : dict
            Mapping layer_name -> object with .tensor_contents (list/array of tensors).
        ncols : int, optional
            How many columns in the grid. Default is ceil(sqrt(n_samples)).
        figsize_per_plot : tuple, optional
            Width, height in inches per subplot.
        cmap : str or Colormap, optional
            Matplotlib colormap for imshow.
        interpolation : str, optional
            Matplotlib interpolation mode.
        """
        if self.model_h == None:
            raise Exception("Oops, you should first run the save_activations_grads function")
        nplots = len(sample_indices)
        if ncols is None:
            ncols = int(math.ceil(math.sqrt(nplots)))
        nrows = int(math.ceil(nplots / ncols))
        
        # Precompute activations dicts for each sample
        act_dicts = []
        for idx in sample_indices:
            # extract and flatten each layer's activations
            d = {
                layer: self.model_h[layer]
                        .tensor_contents[idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                for layer in layers_to_print
            }
            act_dicts.append(d)
        
        # For each act_dict, compute its max width
        max_lens = [max(v.shape[0] for v in d.values()) for d in act_dicts]
        
        # Prepare the figure
        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(figsize_per_plot[0]*ncols,
                                        figsize_per_plot[1]*nrows),
                                squeeze=False)
        
        for plot_idx, (ax, idx, act_dict, max_len) in enumerate(zip(
                axes.flatten(), sample_indices, act_dicts, max_lens)):
            # Build heat array
            heat = np.full((len(layers_to_print), max_len), np.nan, dtype=float)
            for i, layer in enumerate(layers_to_print):
                vec = act_dict[layer]
                heat[i, :vec.shape[0]] = vec
            
            im = ax.imshow(heat,
                        aspect='auto',
                        interpolation=interpolation,
                        cmap=cmap)
            ax.set_yticks(range(len(layers_to_print)))
            ax.set_yticklabels(layers_to_print)
            ax.set_xlabel('Neuron index')
            ax.set_title(f'Sample #{idx}')
            # add a colorbar next to each subplot
            fig.colorbar(im, ax=ax, label='Activation value')
        
        # Hide any unused subplots
        for empty_ax in axes.flatten()[nplots:]:
            empty_ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def run_svd_global(self, layers_to_print=[],grad=False):
        """
        Plots activation heatmaps for the given layers and sample indices.
        
        Parameters
        ----------
        layers : list of str
            The layer names to include (rows of each heatmap).
        sample_indices : list of int
            The sample indices to plot; one subplot per index.
        model_history : dict
            Mapping layer_name -> object with .tensor_contents (list/array of tensors).
        ncols : int, optional
            How many columns in the grid. Default is ceil(sqrt(n_samples)).
        figsize_per_plot : tuple, optional
            Width, height in inches per subplot.
        cmap : str or Colormap, optional
            Matplotlib colormap for imshow.
        interpolation : str, optional
            Matplotlib interpolation mode.
        """
        if self.model_h == None:
            raise Exception("Oops, you should first run the save_activations_grads function")
        
        num_samples = self.model_h[layers_to_print[0]].tensor_contents.shape[0]
        if grad==False:
            flat_per_layer = [
                self.model_h[L].tensor_contents
                                .view(num_samples, -1)
                                .detach()
                                .cpu()
                                .numpy()
                for L in layers_to_print
            ]
        else:
            flat_per_layer = [
                self.model_h[L].grad_contents
                                .view(num_samples, -1)
                                .detach()
                                .cpu()
                                .numpy()
                for L in layers_to_print
            ]
        # 4) concatenate them into one big matrix A of shape (num_samples, total_neurons)
        A = np.concatenate(flat_per_layer, axis=1)
        self.A = A
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        self.svd_matrices = (U, S, Vt)
        return

    def plot_svd_spectrum(self):
        # --- Plot spectrum ---
        if self.svd_matrices == None:
            raise Exception("Oops, you should first run the run_svd_global function")
        (_, S, _) = self.svd_matrices
        plt.figure(figsize=(6,4))
        plt.plot(S, marker='o')
        plt.yscale('log')
        plt.xlabel('Mode index')
        plt.ylabel('Singular value (log scale)')
        plt.title('Global singular‐value spectrum')
        plt.show()

    def plot_svd_energy(self):
        # --- Plot spectrum ---
        if self.svd_matrices == None:
            raise Exception("Oops, you should first run the run_svd_global function")
        (_, S, _) = self.svd_matrices
        expl = np.cumsum(S**2) / np.sum(S**2)
        plt.plot(expl, marker='o')
        plt.xlabel('Number of modes')
        plt.ylabel('Cumulative explained variance')
        plt.yscale('linear')
        plt.axhline(0.9, color='k', linestyle='--')
        plt.show()

    def plot_svd_mode_analysis(self,layers_to_print=[], modes_considered=5):
        if self.svd_matrices == None:
            raise Exception("Oops, you should first run the run_svd_global function")
        if self.model_h == None:
            raise Exception("Oops, you should first run the save_activations_grads function")
        (U, S, Vt) = self.svd_matrices
        n_per_layer = [int(self.model_h[L].tensor_contents[1].numel()) for L in layers_to_print]
        offsets     = np.cumsum([0] + n_per_layer)
        total_neurons = offsets[-1]
        # ---Top 3 modes as heatmaps ---
        for m in range(modes_considered):
            vec = Vt[m]
            # split vec back per layer
            maps = [vec[offsets[i]:offsets[i+1]] for i in range(len(layers_to_print))]
            max_n = max(len(mp) for mp in maps)
            heat  = np.full((len(layers_to_print), max_n), np.nan)
            for i, mp in enumerate(maps):
                heat[i, :len(mp)] = mp

            plt.figure(figsize=(8,3))
            plt.imshow(heat, aspect='auto', interpolation='none')
            plt.colorbar(label=f'Mode {m+1} loadings')
            plt.yticks(range(len(layers_to_print)), layers_to_print)
            plt.xlabel('Neuron index')
            plt.title(f'Global mode #{m+1}')
            plt.show()

        # --- Extract “circuits” for modes 1–3 by thresholding loadings ---
        circuits = {}
        for m in range(modes_considered):
            thresh = 2 * np.std(Vt[m])
            idxs   = np.where(np.abs(Vt[m]) > thresh)[0]
            # map back to each layer
            circuits[m+1] = {
                layers_to_print[i]: (
                    idxs[(idxs >= offsets[i]) & (idxs < offsets[i+1])] - offsets[i]
                ).tolist()
                for i in range(len(layers_to_print))
            }

        print("Circuits (mode → {layer: [neuron indices]}):")
        # self.circuits = circuits
        for m, mp in circuits.items():
            print(f"\nMode {m}:")
            for L, nl in mp.items():
                print(f"  {L}: {nl}")

        # --- Bar-chart of circuit sizes per layer ---
        for m in range(modes_considered):
            counts = [len(circuits[m+1][L]) for L in layers_to_print]
            plt.figure(figsize=(6,3))
            plt.bar(layers_to_print, counts)
            plt.xticks(rotation=45)
            plt.ylabel('Num. neurons')
            plt.title(f'Mode {m+1} circuit sizes by layer')
            plt.tight_layout()
            plt.show()
        
        print("How many neurons each mode has in common with other modes")
        # layers = list(circuits[1].keys())
        modes  = sorted(circuits.keys())
        for L in layers_to_print:
            # build a matrix M where M[i,j] = |circuit_i ∩ circuit_j| for layer L
            M = np.zeros((len(modes), len(modes)), int)
            for a, i in enumerate(modes):
                for b, j in enumerate(modes):
                    set_i = set(circuits[i][L])
                    set_j = set(circuits[j][L])
                    M[a,b] = len(set_i & set_j)
            print(f"\nOverlap counts in layer {L}:")
            print("    " + "  ".join(f"M{j}" for j in modes))
            for a, i in enumerate(modes):
                row = "  ".join(f"{M[a,b]:2d}" for b in range(len(modes)))
                print(f"M{i}  {row}")

        return



        # '''Plot the interface of the simulation.'''
        # if self.fig2 is None:
        #     import matplotlib.pyplot as plt
        #     self.fig2, self.ax2 = plt.subplots()
        
        # self.ax2.clear()
        # self.ax2.plot(self.solver.interface_time, self.solver.interface_results)
        # self.ax2.set_xlabel('Time')
        # self.ax2.set_ylabel('Interface Results')
        # self.ax2.set_title('Simulation Interface Results')
        
        # plt.draw()
        # plt.pause(0.01)

    # def plot_tsne_interface(self):
    #     '''Plot the t-SNE results of the simulation.'''
    #     if self.fig2 is None:
    #         import matplotlib.pyplot as plt
    #         self.fig2, self.ax2 = plt.subplots()
        
    #     self.ax2.clear()
    #     self.ax2.scatter(self.solver.tsne_results[:, 0], self.solver.tsne_results[:, 1])
    #     self.ax2.set_xlabel('t-SNE Component 1')
    #     self.ax2.set_ylabel('t-SNE Component 2')
    #     self.ax2.set_title('t-SNE Results')
        
    #     plt.draw()
    #     plt.pause(0.01)


    # def plot_umap_interface(self):
    #     '''Plot the UMAP results of the simulation.'''
    #     if self.fig2 is None:
    #         import matplotlib.pyplot as plt
    #         self.fig2, self.ax2 = plt.subplots()
        
    #     self.ax2.clear()
    #     self.ax2.scatter(self.solver.umap_results[:, 0], self.solver.umap_results[:, 1])
    #     self.ax2.set_xlabel('UMAP Component 1')
    #     self.ax2.set_ylabel('UMAP Component 2')
    #     self.ax2.set_title('UMAP Results')
        
    #     plt.draw()
    #     plt.pause(0.01)

    def plot_residuals(self, dataset):
        residuals = self.solver.compute_residuals(dataset) # needs to be fixed


    def plot_loss_landscape(self,dataset, num_state_variables, criterion,STEPS):
        # criterion = torch.nn.MSELoss() # specify your loss function
        # metric = metrics.Loss(criterion, x_plot, y_plot, device) # device - 'cuda' or 'cpu'"

        '''Plot the loss landscape of the simulation.'''
        
        model_final = copy.deepcopy(self.solver.model)
        # data that the evaluator will use when evaluating loss
        x_train, y_train = self.solver.split_data(dataset)
        y_filtered = y_train[:, 1:num_state_variables+1]            # shape (N, original_dim-3)

        # 2) convert to tensors of the right type
        X = x_train   # features as float32
        Y = y_filtered

        # 3) wrap in a dataset (and optionally a loader)
        train_dataset = TensorDataset(X, Y)
        dataloader  = DataLoader(train_dataset, batch_size=30000, shuffle=True)

        x, y = iter(dataloader).__next__()
        metric = loss_landscapes.metrics.Loss(criterion, x, y)

        # compute loss data
        loss_data_fin = loss_landscapes.random_plane(model_final, metric, 10, STEPS, normalization='filter', deepcopy_model=True)

        plt.contour(loss_data_fin, levels=50)
        plt.title('Loss Contours around Trained Model')
        plt.show()
        # if self.fig2 is None:
        #     import matplotlib.pyplot as plt
        #     self.fig2, self.ax2 = plt.subplots()
        
        # self.ax2.clear()
        # self.ax2.contourf(self.solver.loss_landscape_x, self.solver.loss_landscape_y, self.solver.loss_landscape_z)
        # self.ax2.set_xlabel('Parameter 1')
        # self.ax2.set_ylabel('Parameter 2')
        # self.ax2.set_title('Loss Landscape')
        
        # plt.draw()
        # plt.pause(0.01)

    def plot_loss_landscape_3d(self,dataset, num_state_variables, criterion,STEPS):
        '''Plot the 3D loss landscape of the simulation.'''
        model_final = copy.deepcopy(self.solver.model)
        # data that the evaluator will use when evaluating loss
        x_train, y_train = self.solver.split_data(dataset)
        y_filtered = y_train[:, 1:num_state_variables+1]            # shape (N, original_dim-3)

        # 2) convert to tensors of the right type
        X = x_train   # features as float32
        Y = y_filtered

        # 3) wrap in a dataset (and optionally a loader)
        train_dataset = TensorDataset(X, Y)
        dataloader  = DataLoader(train_dataset, batch_size=30000, shuffle=True)

        x, y = iter(dataloader).__next__()
        metric = loss_landscapes.metrics.Loss(criterion, x, y)

        loss_data_fin = loss_landscapes.random_plane(model_final, metric, 10, STEPS, normalization='filter', deepcopy_model=True)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
        Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
        ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('Surface Plot of Loss Landscape')
        fig.show()
        # if self.fig2 is None:
        #     import matplotlib.pyplot as plt
        #     from mpl_toolkits.mplot3d import Axes3D
        #     self.fig2 = plt.figure()
        #     self.ax2 = self.fig2.add_subplot(111, projection='3d')
        
        # self.ax2.clear()
        # self.ax2.plot_surface(self.solver.loss_landscape_x, self.solver.loss_landscape_y, self.solver.loss_landscape_z, cmap='viridis')
        # self.ax2.set_xlabel('Parameter 1')
        # self.ax2.set_ylabel('Parameter 2')
        # self.ax2.set_zlabel('Loss')
        # self.ax2.set_title('3D Loss Landscape')
        
        # plt.draw()
        # plt.pause(0.01)
