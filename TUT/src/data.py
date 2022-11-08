"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Henrik Hembrock, Jonathan Stollberg
         
08/2022
"""
  
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

#%% Data generation functions

def bathtub():
    """
    Generate data for a bathtub function.
    """
    xs = np.linspace(1, 10, 450)
    ys = np.concatenate([np.square(xs[0:150] - 4) + 1,
                         1 + 0.1 * np.sin(np.linspace(0, 3.14, 90)), 
                         np.ones(60), np.square(xs[300:450] - 7) + 1])

    xs = xs / 10.0
    ys = ys / 10.0

    xs_c = np.concatenate([xs[0:240], xs[330:420]])
    ys_c = np.concatenate([ys[0:240], ys[330:420]])

    xs = tf.expand_dims(xs, axis=1)
    ys = tf.expand_dims(ys, axis=1)

    xs_c = tf.expand_dims(xs_c, axis=1)
    ys_c = tf.expand_dims(ys_c, axis=1)

    return xs, ys, xs_c, ys_c

def f1_data():
    """
    Generate data for `f1 = x**2 + y**2`.
    """
    xs = np.linspace(-4, 4, 20)
    ys = np.linspace(-4, 4, 20)
    xs, ys = np.meshgrid(xs, ys)
    
    # Cut out the 4x4 grid in the the middle for calibration data
    cut = np.concatenate([range(0,8), range(12,20)])
    cut = np.ix_(cut, cut)
    xs_c = xs[cut]
    ys_c = ys[cut]

    xs = xs.reshape((-1,1))
    ys = ys.reshape((-1,1))
    zs = F1()(xs, ys)
    
    xs_c = xs_c.reshape((-1,1))
    ys_c = ys_c.reshape((-1,1))
    zs_c = F1()(xs_c, ys_c)
    
    return xs, ys, zs, xs_c, ys_c, zs_c

def f2_data():
    """
    Generate data for `f2 = x**2 + 0.5*y**2`.
    """
    xs = np.linspace(-4, 4, 20)
    ys = np.linspace(-4, 4, 20)
    xs, ys = np.meshgrid(xs, ys)
    
    # Cut out the 4x4 grid in the middle for calibration data
    cut = np.concatenate([range(0,8), range(12,20)])
    cut = np.ix_(cut, cut)
    xs_c = xs[cut]
    ys_c = ys[cut]
    
    xs = xs.reshape((-1,1))
    ys = ys.reshape((-1,1))
    (zs, grad) = F2()(xs, ys)
    
    xs_c = xs_c.reshape((-1,1))
    ys_c = ys_c.reshape((-1,1))
    (zs_c, grad_c) = F2()(xs_c, ys_c)
    
    return xs, ys, zs, grad, xs_c, ys_c, zs_c, grad_c

#%% Non-trainable layers
class F1(layers.Layer):
    """
    Non-trainable layer `f1 = x**2 + y**2`.
    """
    def __call__(self, x, y):
        return x ** 2 + y ** 2

class F2(layers.Layer):
    """
    Non-trainable layer `f2 = x**2 + 0.5*y**2`.
    """
    def __call__(self, x, y):
        return x ** 2 + 0.5 + y ** 2, np.hstack([2*x, y])