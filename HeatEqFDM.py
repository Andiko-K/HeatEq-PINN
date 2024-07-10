import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class heat_equation:
    def __init__(self, X,Y, T,alpha = 1.85e-5):
        self.X = X
        self.Y = Y
        self.dx = X[1,1]-X[0,0]
        self.dy = Y[1,1]-Y[0,0]
        self.dt = T[1]-T[0]
        self.a = alpha    
        self.conditions = None
        self.temps = None
        self.__convergence_check()
    
    def initial_temp(self, init_heat):
        self.init_heat = init_heat
        
    def time_array(self, times, snapshot):
        self.times = times
        self.snapshot = snapshot
        self.num_frames = int(self.times/(self.snapshot))
        height, width = self.X.shape
        self.heat_frames = np.zeros([self.num_frames, height, width])
        self.heat_frames[0] = self.init_heat
        del self.init_heat
    
    def custom_boundaries(self, conditions = None, temps = None):
        self.conditions = conditions
        self.temps = temps
        
    def solve_heat(self, neumann = True, dirichlet = True, bound_temp = 273):
        cs = self.heat_frames[0].copy() #current state
        current_frame = 0
        for t in range(self.times-1):
            ns = cs.copy() #new state
            cs[1:-1, 1:-1] = (ns[1:-1, 1:-1] + 
                  self.a * self.dt / self.dx**2 * 
                  (ns[1:-1, 2:] - 2 * ns[1:-1, 1:-1] + ns[1:-1, 0:-2]) +
                  self.a * self.dt / self.dy**2 * 
                  (ns[2:, 1:-1] - 2 * ns[1:-1, 1:-1] + ns[0:-2, 1:-1]))
            
            # Neumann Boundary Conditions
            if neumann:
                ns[:, 0] = ns[:, 1]
                ns[:, -1] = ns[:, -2]
                ns[0, :] = ns[1, :]
                ns[-1, :] = ns[-2, :]
        
            # Dirichlet Boundary Conditions
            if dirichlet:
                ns[0, :] = bound_temp[0]
                ns[-1, :] = bound_temp[1]
                ns[:, 0] = bound_temp[2]
                ns[:, -1] = bound_temp[3]
                
            if self.conditions is not None:
                for i in range(len(self.conditions)):
                    cs[self.conditions[i]] = self.temps[i]
            
            if t%self.snapshot == 0:
                self.heat_frames[current_frame] = cs
                current_frame += 1
    
    def visualize(self):
        fig, ax = plt.subplots(figsize = (4,3))
        vmin = self.heat_frames.min(); vmax = self.heat_frames.max()
        contour = ax.contourf(self.X, self.Y, self.heat_frames[0], vmin=vmin, vmax=vmax,
                             cmap = 'magma')

        def update(frame):
            ax.clear()
            contour = ax.contourf(self.X, self.Y, self.heat_frames[frame], 
                          vmin = vmin, vmax = vmax, cmap = 'magma')
            ax.set_title(f'{(self.snapshot*self.dt*frame):.2f} seconds')
            return contour

        ani = FuncAnimation(fig, update, frames=self.num_frames, interval=100)
        cbar = plt.colorbar(contour, ax=ax)
        ani.save('HeatEqAnimation.gif', writer='Pillow', fps=10)
    
    def save(self, path = 'HeatEqFDM.npy'):
        np.save(path, self.heat_frames)
        
    def __convergence_check(self):
        string = "Function is Diverging! Please lower the increment (dx, dy, dt) or Configure the diffusivity"
        if self.a*self.dt/(self.dx**2) > .25:
            print(string)
        elif self.a*self.dt/(self.dy**2) > .25:
            print(string)