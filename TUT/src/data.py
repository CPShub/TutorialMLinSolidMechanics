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
import numpy as np
from tensorflow.keras import layers

# %%
"""
Generate data for a bathtub function

"""


def bathtub():
    xs = np.linspace(1, 10, 450)
    ys = np.concatenate([np.square(xs[0:150] - 4) + 1,
                         1 + 0.1 * np.sin(np.linspace(0, 3.14, 90)), np.ones(60),
                         np.square(xs[300:450] - 7) + 1])

    xs = xs / 10.0
    ys = ys / 10.0

    xs_c = np.concatenate([xs[0:240], xs[330:420]])
    ys_c = np.concatenate([ys[0:240], ys[330:420]])

    xs = tf.expand_dims(xs, axis=1)
    ys = tf.expand_dims(ys, axis=1)

    xs_c = tf.expand_dims(xs_c, axis=1)
    ys_c = tf.expand_dims(ys_c, axis=1)

    return xs, ys, xs_c, ys_c


def F1_data():
    xs = np.linspace(-4, 4, 20)
    ys = np.linspace(-4, 4, 20)

    zs = np.zeros((20, 20))
    mesh = np.zeros((20, 20, 2))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            mesh[i, j, :] = [x, y]
            zs[i, j] = generate_data_f1(x, y)

    #mesh = tf.expand_dims(mesh, axis=1)
    #zs = tf.expand_dims(zs, axis=1)
    return mesh, zs, xs, ys


class F1(layers.Layer):
    def __call__(self, x, y):
        return x ** 2 + y ** 2


class F2(layers.Layer):
    def __call__(self, x, y):
        return x ** 2 - 0.5 + y ** 2


def generate_data_f1(x, y):
    return F1()(x, y)


def generate_data_f2(x, y):
    return F2()(x, y)
