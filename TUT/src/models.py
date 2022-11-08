"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Henrik Hembrock, Jonathan Stollberg
         
08/2022
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import non_neg

#%% custom layers

class _x_to_y(layers.Layer):
    """
    Custom trainable layer for scalar output.
    """
    def __init__(self, 
                 nlayers=3, 
                 units=8,
                 activation="softplus", 
                 constraint=non_neg()):
        super(_x_to_y, self).__init__()
        
        # define hidden layers with activation functions
        self.ls = [layers.Dense(units, activation=activation)]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, activation=activation, 
                                     kernel_constraint=constraint)]
            
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint=constraint)]
            
    def __call__(self, x):     
        for l in self.ls:
            x = l(x)
        return x
    
class _x_to_y_and_grad(tf.keras.Model):
    """
    Custom trainable layer for gradients.
    """
    def __init__(self,
                 nlayers=3,
                 units=8,
                 activation="softplus",
                 constraint=non_neg()):
        super(_x_to_y_and_grad, self).__init__()
        
        # define hidden layers with activation functions
        self.ls = _x_to_y(nlayers, units, activation, constraint)
        
    def call(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.ls(x)
        g = tape.gradient(y, x)
        
        return y, g
    
#%% construction functions

def main(input_shape=[1], **kwargs):
    # define input shape
    xs = tf.keras.Input(shape=input_shape)
    
    # define which (custom) layers the model uses
    ys = _x_to_y(**kwargs)(xs)
    
    # connect input and output
    model = tf.keras.Model(inputs=xs, outputs=ys)
    
    # define optimizer and loss function
    model.compile('adam', 'mse')
    
    return model

def main_grad():
    model = _x_to_y_and_grad()
    model.compile("adam", "mse")
    
    return model

#%% prototyping

if __name__ == "__main__":
    
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    
    import data as ld
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    xs, ys, zs, grad, xs_c, ys_c, zs_c, grad_c = ld.f2_data()
    model_input = np.hstack((xs_c, ys_c))
    
    m = _x_to_y_and_grad()
    m.compile("adam", "mse")
    
    h = m.fit(model_input, [zs_c, grad_c], epochs=1500, verbose=2)
    
    #%% Loss
    plt.figure(1, dpi=600)
    plt.semilogy(h.history['loss'], label='training loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend()
    plt.savefig("loss_convex.pdf")
    
    # Interpolated data
    # zs_model = m.predict(np.hstack((xs, ys)))[0]
    zs_model = m.predict(np.hstack((xs, ys)))[1][:,1]
    X = xs.reshape((20,20))
    Y = ys.reshape((20,20))
    Z_MODEL = zs_model.reshape((20,20))
    plt.figure(2, dpi=600)
    ax2 = plt.axes(projection="3d")
    ax2.plot_surface(X, Y, Z_MODEL, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="model")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("model_convex.pdf")
    
    # Reference data
    # Z = zs.reshape((20,20))
    Z = grad[:,1].reshape((20,20))
    plt.figure(3, dpi=600)
    ax3 = plt.axes(projection="3d")
    ax3.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="data")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("training_data_convex.pdf")
    
    plt.show()
    