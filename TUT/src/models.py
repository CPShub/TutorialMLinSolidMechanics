"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein
         
08/2022
"""


# %%   
"""
Import modules

"""
import tensorflow as tf
import datetime

from tensorflow.keras import layers
from tensorflow.keras.constraints import non_neg
now = datetime.datetime.now


# %%   
"""
_x_to_y: custom trainable layer

"""

class _x_to_y(layers.Layer):
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


class _xy_to_z(layers.Layer):
    def __init__(self):
        super(_xy_to_z, self).__init__()

        self.ls = [layers.Dense(8, 'softplus')]
        self.ls += [layers.Dense(8, 'softplus')]

        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]

    def __call__(self, x):
        for l in self.ls:
            x = l(x)
        return x


class _xy_to_z_nonconvex(layers.Layer):
    def __init__(self):
        super(_xy_to_z_nonconvex, self).__init__()

        self.ls = [layers.Dense(4, 'softplus')]
        self.ls += [layers.Dense(4, 'softplus')]

        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]

    def __call__(self, x):
        for l in self.ls:
            x = l(x)
        return x
# %%   
"""
main: construction of the NN model

"""

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


def main_2(**kwargs):

    xs = tf.keras.Input(shape=[20, 2])
    ys = _xy_to_z(**kwargs)(xs)

    model = tf.keras.Model(inputs=[xs], outputs=[ys])
    model.compile('adam', 'mse')
    return model
