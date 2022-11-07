"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Henrik Hembrock, Jonathan Stollberg
         
08/2022
"""

import numpy as np
import tensorflow as tf
import datetime
import data as ld
import models as lm
from matplotlib import pyplot as plt
from matplotlib import cm
now = datetime.datetime.now

# Set this to avoid conflicts with matplotlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

#%% Load model and data

# Options: 1d_convex, 2d_convex, 2d_nonconvex
model_type = "2d_convex"

if model_type == "1d_convex":
    model = lm.main()
    
elif model_type == "2d_convex":
    model = lm.main_2d_convex()
    
elif model_type == "2d_nonconvex":
    model = lm.main_2d_nonconvex()
    
else:
    raise NotImplementedError(f"Model {model_type} does not exist.")
    
#%% Load data

# Options: bathtub, f1, f2
data = "f1"

if data == "bathtub":
    xs, ys, xs_c, ys_c = ld.bathtub()
    model_input = xs_c
    model_output = ys_c
    
elif data == "f1":
    xs, ys, zs, xs_c, ys_c, zs_c = ld.f1_data()
    model_input = np.hstack((xs_c, ys_c))
    model_output = zs_c
    
elif data == "f2":
    xs, ys, zs, xs_c, ys_c, zs_c = ld.f2_data()
    model_input = np.hstack((xs_c, ys_c))
    model_output = zs_c
    
else:
    raise NotImplementedError(f"Data function {data} does not exist.")

#%% Model calibration

t1 = now()
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit([model_input], [model_output], epochs=1500, verbose=2)
print(f"It took {now() - t1} sec to calibrate the model.")

#%% Result plots

if model_type == "1d_convex" and data == "bathtub":
    plt.figure(1, dpi=600)
    plt.semilogy(h.history['loss'], label='training loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend()
    #plt.savefig('losses_3L4_5000_nn.pdf')

    plt.figure(2, dpi=600)
    plt.scatter(xs_c[::10], ys_c[::10], c='green', label='calibration data')
    plt.plot(xs, ys, c='black', linestyle='--', label='bathtub function')
    plt.plot(xs, model.predict(xs), label='model', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    #plt.savefig('calidata_3L4_5000_nn.pdf')
    plt.show()

elif model_type == "2d_convex" or model_type == "2d_nonconvex":
    
    # Loss
    plt.figure(1, dpi=600)
    plt.semilogy(h.history['loss'], label='training loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend()
    
    # Interpolated data
    zs_model = model.predict(np.hstack((xs, ys)))
    X = xs.reshape((20,20))
    Y = ys.reshape((20,20))
    Z_MODEL = zs_model.reshape((20,20))
    plt.figure(2, dpi=600)
    ax2 = plt.axes(projection="3d")
    ax2.plot_surface(X, Y, Z_MODEL, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="model")
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Reference data
    Z = zs.reshape((20,20))
    plt.figure(3, dpi=600)
    ax3 = plt.axes(projection="3d")
    ax3.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="data")
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()
