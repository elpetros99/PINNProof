


class visualizer():
    '''A class to visualize the results of a simulation.
    This class is used to visualize the results of a simulation.
    it takes a solver object as input and uses it to visualize the results.
    It can be used to visualize the results of a simulation in a variety of ways.
    '''
    def __init__(self, solver):
        self.solver = solver
        self.fig = None
        self.ax = None
        self.fig2 = None
    
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

    def plot_interface(self):
        '''Plot the interface of the simulation.'''
        if self.fig2 is None:
            import matplotlib.pyplot as plt
            self.fig2, self.ax2 = plt.subplots()
        
        self.ax2.clear()
        self.ax2.plot(self.solver.interface_time, self.solver.interface_results)
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Interface Results')
        self.ax2.set_title('Simulation Interface Results')
        
        plt.draw()
        plt.pause(0.01)

    def plot_tsne_interface(self):
        '''Plot the t-SNE results of the simulation.'''
        if self.fig2 is None:
            import matplotlib.pyplot as plt
            self.fig2, self.ax2 = plt.subplots()
        
        self.ax2.clear()
        self.ax2.scatter(self.solver.tsne_results[:, 0], self.solver.tsne_results[:, 1])
        self.ax2.set_xlabel('t-SNE Component 1')
        self.ax2.set_ylabel('t-SNE Component 2')
        self.ax2.set_title('t-SNE Results')
        
        plt.draw()
        plt.pause(0.01)


    def plot_umap_interface(self):
        '''Plot the UMAP results of the simulation.'''
        if self.fig2 is None:
            import matplotlib.pyplot as plt
            self.fig2, self.ax2 = plt.subplots()
        
        self.ax2.clear()
        self.ax2.scatter(self.solver.umap_results[:, 0], self.solver.umap_results[:, 1])
        self.ax2.set_xlabel('UMAP Component 1')
        self.ax2.set_ylabel('UMAP Component 2')
        self.ax2.set_title('UMAP Results')
        
        plt.draw()
        plt.pause(0.01)

    def plot_loss_landscape(self):
        '''Plot the loss landscape of the simulation.'''
        if self.fig2 is None:
            import matplotlib.pyplot as plt
            self.fig2, self.ax2 = plt.subplots()
        
        self.ax2.clear()
        self.ax2.contourf(self.solver.loss_landscape_x, self.solver.loss_landscape_y, self.solver.loss_landscape_z)
        self.ax2.set_xlabel('Parameter 1')
        self.ax2.set_ylabel('Parameter 2')
        self.ax2.set_title('Loss Landscape')
        
        plt.draw()
        plt.pause(0.01)

    def plot_loss_landscape_3d(self):
        '''Plot the 3D loss landscape of the simulation.'''
        if self.fig2 is None:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            self.fig2 = plt.figure()
            self.ax2 = self.fig2.add_subplot(111, projection='3d')
        
        self.ax2.clear()
        self.ax2.plot_surface(self.solver.loss_landscape_x, self.solver.loss_landscape_y, self.solver.loss_landscape_z, cmap='viridis')
        self.ax2.set_xlabel('Parameter 1')
        self.ax2.set_ylabel('Parameter 2')
        self.ax2.set_zlabel('Loss')
        self.ax2.set_title('3D Loss Landscape')
        
        plt.draw()
        plt.pause(0.01)

