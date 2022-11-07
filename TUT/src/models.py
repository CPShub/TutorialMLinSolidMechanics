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

#%% Custom layers

class _x_to_y(layers.Layer):
    """
    Convex custom trainable layer for scalar input and scalar output.
    """
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus', kernel_constraint=non_neg())]
        self.ls += [layers.Dense(4, 'softplus', kernel_constraint=non_neg())]
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
            
    def __call__(self, x):     
        for l in self.ls:
            x = l(x)
        return x

class _xy_to_z_convex(layers.Layer):
    """
    Convex custom trainable layer for 2d input and scalar output.
    """
    def __init__(self):
        super(_xy_to_z_convex, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        self.ls += [layers.Dense(8, 'softplus', kernel_constraint=non_neg())]
        # scalar-valued output function
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]

    def __call__(self, x):
        for l in self.ls:
            x = l(x)
        return x

class _xy_to_z_nonconvex(layers.Layer):
    """
    Nonconvex custom trainable layer for 2d input and scalar output.
    """
    def __init__(self):
        super(_xy_to_z_nonconvex, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]

    def __call__(self, x):
        for l in self.ls:
            x = l(x)
        return x
    
#%% Construction functions

def main(**kwargs):
    # define input shape
    xs = tf.keras.Input(shape=[1])
    # define which (custom) layers the model uses
    ys = _x_to_y(**kwargs)(xs)
    # connect input and output
    model = tf.keras.Model(inputs=[xs], outputs=[ys])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model

def main_2d_convex(**kwargs):
    xsys = tf.keras.Input(shape=[2])
    zs = _xy_to_z_convex(**kwargs)(xsys)
    model = tf.keras.Model(inputs=[xsys], outputs=[zs])
    model.compile('adam', 'mse')
    return model

def main_2d_nonconvex(**kwargs):
    xsys = tf.keras.Input(shape=[2])
    zs = _xy_to_z_nonconvex(**kwargs)(xsys)
    model = tf.keras.Model(inputs=[xsys], outputs=[zs])
    model.compile('adam', 'mse')
    return model
