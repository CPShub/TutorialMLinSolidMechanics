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

class _BaseNeuralNetwork(layers.Layer):
    """
    Custom trainable layer for scalar output.
    """
    def __init__(self, 
                 nlayers=3, 
                 units=8,
                 activation="softplus", 
                 constraint=non_neg()):
        super(_BaseNeuralNetwork, self).__init__()
        
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
    
class StdNeuralNetwork(tf.keras.Model):
    """
    Neural network that computes scalar output.
    """
    def __init__(self,
                  nlayers=3,
                  units=8,
                  activation="softplus",
                  constraint=non_neg()):
        super(StdNeuralNetwork, self).__init__()
        self.ls = _BaseNeuralNetwork(nlayers, units, activation, constraint)
        
    def call(self, x):
        y = self.ls(x)
        return y
    
class GradNeuralNetwork(tf.keras.Model):
    """
    Neural network that computes scalar output and its gradient.
    """
    def __init__(self,
                  nlayers=3,
                  units=8,
                  activation="softplus",
                  constraint=non_neg()):
        super(GradNeuralNetwork, self).__init__()
        self.ls = _BaseNeuralNetwork(nlayers, units, activation, constraint)
        
    def call(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.ls(x)
        g = tape.gradient(y, x)
        
        return y, g