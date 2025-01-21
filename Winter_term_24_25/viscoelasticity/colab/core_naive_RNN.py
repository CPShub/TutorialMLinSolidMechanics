import tensorflow as tf
from tensorflow.keras import layers, constraints


class MLP(layers.Layer):
    ''' A feed-forward neural network '''
    def __init__(self, units, activation, non_neg):
        super().__init__()
        self.ls = []
        for (u, a, n) in zip(units, activation, non_neg):
            if n:
                kernel_constraint = tf.keras.constraints.non_neg()
            else:
                kernel_constraint = None
            self.ls += [layers.Dense(u, a, kernel_constraint=kernel_constraint)]  

    def call(self, x):    
        for l in self.ls:
            x = l(x)
        return x


class RNNCell(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1
     
        self.ls = [layers.Dense(32, 'softplus')]
        self.ls += [layers.Dense(2)]

        
    def call(self, inputs, states):
        
        #   states are the internal variables
        #   n: current time step, N: old time step
                
        eps_n = inputs[:, 0:1]
        hs = inputs[:, 1:2]

        #eps_n = inputs[0]
        #hs = inputs[1]
        
        #   gamma: history variable
        gamma_N = states[0]
        
        #   x contains the current strain, the current time step size, and the 
        #   history variable from the previous time step
        x = tf.concat([eps_n, hs, gamma_N], axis = 1)
                
        #   x gets passed to a FFNN which yields the current stress and history
        #   variable
    
        for l in self.ls:
            x = l(x)
         
        sig_n = x[:, 0:1]
        gamma_n = x[:, 1:2]
            
                
        return sig_n, [gamma_n]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        #   define initial values of the internal variables      
        return [tf.zeros([batch_size, 1])]


def main(**kwargs):
    # define inputs
    eps = tf.keras.Input(shape=[None, 1],name='input_eps')
    hs = tf.keras.Input(shape=[None, 1], name='input_hs')

    # concatenate inputs
    concatenated_inputs = tf.keras.layers.Concatenate(axis=-1)([eps, hs])

    # define RNN cell
    cell = RNNCell()
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False) # return_state=True => hier wird sig_n zurückgegeben, return_state=False => hier wird gamma nicht zurückgeben, aber für nächste Berechnung verwendet
    sigs = layer1(concatenated_inputs)

    model = tf.keras.Model([eps, hs], [sigs])
    model.compile('adam', 'mse')
    return model