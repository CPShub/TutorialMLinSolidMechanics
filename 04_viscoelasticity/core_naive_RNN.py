"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Dominik K. Klein
         
01/2023
"""


# %%   
"""
Import modules

"""


import tensorflow as tf
from tensorflow.keras import layers
    
    

class RNNCell(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.state_size = [[1]]
        self.output_size = [[1]]
     
        self.ls = [layers.Dense(32, 'softplus')]
        self.ls += [layers.Dense(2)]

        
    def call(self, inputs, states):
        
        #   states are the internal variables
        
        #   n: current time step, N: old time step
                
        eps_n = inputs[0]
        hs = inputs[1]
        
        #   gamma: history variable
        
        gamma_N = states[0]
        
        #   x contains the current strain, the current time step size, and the 
        #   history variable from the previous time step
        
        x = tf.concat([eps_n, hs, gamma_N], axis = 1)
                
        #   x gets passed to a FFNN which yields the current stress and history
        #   variable
        
        for l in self.ls:
            x = l(x)
         
        sig_n = x[:,0:1]
        gamma_n = x[:,1:2]
            
                
        return sig_n , [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        
        #   define initial values of the internal variables
                
        return [tf.zeros([batch_size, 1])]


def main(**kwargs):
    
    # define inputs
    
    eps = tf.keras.Input(shape=[None, 1],name='input_eps')
    hs = tf.keras.Input(shape=[None, 1], name='input_hs')
        
    # define RNN cell
    
    cell = RNNCell()
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1((eps, hs))


    model = tf.keras.Model([eps, hs], [sigs])
    model.compile('adam', 'mse')
    return model



