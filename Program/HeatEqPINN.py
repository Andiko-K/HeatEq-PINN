import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class heat_equation_NN():
    def __init__(self, X, Y, T, alpha = 1.85e-5):
        '''Assign X,Y and T variables'''
        self.X = tf.constant(X.reshape(-1,1), dtype = tf.float32)
        self.Y = tf.constant(Y.reshape(-1,1), dtype = tf.float32)
        self.T = tf.constant(T.reshape(-1,1), dtype = tf.float32)
        self.a = alpha
    
    def nn_model(self, input_shape = (3,), layers = None):
        '''Declare Neural Network Model'''
        input_ = tf.keras.layers.Input(shape = input_shape)
        x = tf.keras.layers.Dense(layers[0], activation = 'tanh')(input_)
        for layer in layers[1:-1]:
            x = tf.keras.layers.Dense(layer, activation = 'tanh')(x)
        output = tf.keras.layers.Dense(layers[-1])(x)
        self.model = tf.keras.Model(input_, output)
    
    def init_bound(self, x_bounds, y_bounds, u_bounds, t_bounds, neumann = True):
        '''Assign Dirichlet Boundaries'''
        self.dirichlet = True
        self.neumann = neumann
        self.X_bounds = tf.constant(x_bounds.reshape(-1,1), dtype = tf.float32)
        self.Y_bounds = tf.constant(y_bounds.reshape(-1,1), dtype = tf.float32)
        self.U_bounds = tf.constant(u_bounds.reshape(-1,1), dtype = tf.float32)
        self.T_bounds = tf.constant(t_bounds.reshape(-1,1), dtype = tf.float32)
    
    def init_value(self, x_init, y_init, u_init, t_init):
        '''Assign Initial Value'''
        self.X_init = tf.constant(x_init.reshape(-1,1), dtype = tf.float32)
        self.Y_init = tf.constant(y_init.reshape(-1,1), dtype = tf.float32)
        self.T_init = tf.constant(t_init.reshape(-1,1), dtype = tf.float32)
        self.U_init = tf.constant(u_init.reshape(-1,1), dtype = tf.float32)
    
    def __compute_pde(self):
        '''Function for Computing PDE Losses'''
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([self.X, self.Y, self.T])
            u = self.model(tf.stack([self.X, self.Y, self.T], axis=1))

            u_x = tape.gradient(u, self.X)
            u_y = tape.gradient(u, self.Y)
            u_t = tape.gradient(u, self.T)

        u_xx = tape.gradient(u_x, self.X)
        u_yy = tape.gradient(u_y, self.Y)

        f = u_t - self.a * (u_xx + u_yy)   
        mse_f = tf.reduce_mean(tf.square(f))
        return mse_f
   
    def __compute_loss(self):
        '''Function for Computing Whole Losses'''
        #PDE Loss
        mse_f = self.__compute_pde()
        
        #Initial Loss -- u(x, t = 0)
        u_init_p = self.model(tf.stack([self.X_init, self.Y_init, self.T_init], axis = 1))
        mse_init = tf.reduce_mean(tf.square(u_init_p-self.U_init))
        
        #Boundary Loss (Dirichlet and Neumann)
        mse_dirichlet = 0
        mse_neumann = 0      
        
        if self.dirichlet: #Dirichlet Loss
            u_bound_p = self.model(tf.concat([self.X_bounds, self.Y_bounds, self.T_bounds], axis = 1))
            mse_dirichlet += tf.reduce_mean(tf.square(u_bound_p-self.U_bounds))
            
        if self.neumann: #Neumann Loss
            pass
            '''
            with tf.GradientTape(persistent = True) as tape:
                tape.watch([xi, yi])
                u_bound_p = self.model(tf.stack[self.X_bounds,self.Y_bounds,self.T_bounds], axis = 1)
            u_x = tape.gradient(u_bound_p, xi)
            u_y = tape.gradient(u_bound_p, yi)
                    
            mse_neumann += tf.reduce_mean(tf.square(u_x-0))
            mse_neumann += tf.reduce_mean(tf.square(u_y-0))
                    
            del tape
            '''
        mse_total = mse_f+mse_init+mse_dirichlet+mse_neumann
        return mse_total
    

    def train_step(self,epochs, print_step = 100, optimizer = tf.keras.optimizers.Adam()):
        '''Training Step'''
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.__compute_loss()
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if epoch % print_step == 0:
                print(f'Epoch: {epoch} Loss: {loss.numpy():.4f}')
    
    def predict(self,X, Y, T):
        self.heat_frames = np.zeros(T.shape + X.shape)
        self.X_new = X; self.Y_new = Y
        self.X_new_tf = tf.constant(X.reshape(-1,1), dtype = tf.float32)
        self.Y_new_tf = tf.constant(Y.reshape(-1,1), dtype = tf.float32)
        self.T_new = T
        
        for i in range(len(self.T_new)):
            T_array = np.zeros_like(X)+self.T_new[i]
            T_array = tf.constant(T_array.reshape(-1,1),dtype = tf.float32)
            input_new = tf.concat([self.X_new_tf, self.Y_new_tf, T_array], axis = 1)
            U_pred = self.model.predict(input_new)
            
            U_pred = U_pred.reshape(self.X_new.shape)
            self.heat_frames[i] = U_pred
            if i%10 == 0:
                print(f'Frame {i} Done!')
    
    def visualize(self):
        fig, ax = plt.subplots(figsize = (4,3))
        vmin = self.heat_frames.min(); vmax = self.heat_frames.max()
        contour = ax.contourf(self.X_new, self.Y_new, self.heat_frames[0], vmin=vmin, vmax=vmax,
                             cmap = 'magma')
        
        def update(frame):
            ax.clear()
            contour = ax.contourf(self.X_new, self.Y_new, self.heat_frames[frame], 
                          vmin = vmin, vmax = vmax, cmap = 'magma')
            ax.set_title(f'{(self.T_new[frame]):.2f} seconds')
            return contour
        
        num_frames = self.heat_frames.shape[0]
        ani = FuncAnimation(fig, update, frames=num_frames, interval=100)
        cbar = plt.colorbar(contour, ax=ax)
        ani.save('HeatEqPINNAnimation.gif', writer='Pillow', fps=10)
    
    def save(self, path = 'HeatEqPINN.npy'):
        np.save(path, self.heat_frames)
