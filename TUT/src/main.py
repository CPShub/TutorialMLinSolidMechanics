"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein
         
08/2022
"""
import numpy as np

# %%
"""
Import modules

"""
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime
from mpl_toolkits import mplot3d
now = datetime.datetime.now

# %% Own modules
import data as ld
from TUT.src import models as lm

# %%
"""
Load model

"""
mod_type = 'second'

if mod_type == 'standard':
    model = lm.main()
elif mod_type == 'second':
    model = lm.main_2()


# %%   
"""
Load data

"""
data_type = 'second'

if data_type == 'standard':
    xs, ys, xs_c, ys_c = ld.bathtub()
elif data_type == 'second':
    mesh, zs, xs, ys = ld.F1_data()

# %%   
"""
Model calibration

"""

t1 = now()
print(t1)

tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit([mesh], [zs], epochs=1500,  verbose=2)

t2 = now()
print('it took', t2 - t1, '(sec) to calibrate the model')

ploten = False

# plot some results

if ploten:
    plt.figure(1, dpi=600)
    plt.semilogy(h.history['loss'], label='training loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend()
    plt.savefig('losses_3L4_5000_nn.pdf')

    plt.figure(2, dpi=600)
    plt.scatter(xs_c[::10], ys_c[::10], c='green', label='calibration data')
    plt.plot(xs, ys, c='black', linestyle='--', label='bathtub function')
    plt.plot(xs, model.predict(xs), label='model', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('calidata_3L4_5000_nn.pdf')
    plt.show()

X, Y = np.meshgrid(xs, ys)
zssa = model.predict(mesh).reshape((20, 20))
plt.figure(3, dpi=600)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, zssa, label='model', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

zzzs = model.predict(mesh)
mesha, zsa, xsa, ysa = ld.F1_data()

print(zzzs[10, 10])
print(zsa[10, 10])
print(zzzs.shape)
print(zsa.shape)
zzzs = zzzs.reshape((20, 20))
print(zzzs.shape)

# %%   
"""
Evaluation

"""







