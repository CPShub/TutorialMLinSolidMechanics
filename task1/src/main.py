"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Henrik Hembrock, Jonathan Stollberg
         
08/2022
"""

import numpy as np
import os

# removes no gpu message
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.constraints import non_neg
import datetime

from data import bathtub_data, f1_data, f2_data
from models import StdNeuralNetwork, GradNeuralNetwork
from plots import plot_bathtub, plot_f1, plot_f2

now = datetime.datetime.now

# set this to avoid conflicts with matplotlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

#%% load data

data = "f2"  # options: bathtub, f1, f2
plotten = False

if data == "bathtub":
    xs, ys, xs_c, ys_c = bathtub_data()
    training_input = xs_c
    training_output = ys_c
    
elif data == "f1":
    xs, ys, zs, xs_c, ys_c, zs_c = f1_data()
    training_input = np.hstack((xs_c, ys_c))
    training_output = zs_c
    
elif data == "f2":
    xs, ys, zs, grad, xs_c, ys_c, zs_c, grad_c = f2_data()
    training_input = np.hstack((xs_c, ys_c))
    training_output = [zs_c, grad_c]
    
else:
    raise NotImplementedError(f"Data function {data} does not exist.")
    
#%% load model

# loss_weights = [0, 1]  # only gradient: [0,1], only output: [1,0]
loss_weights = None
kwargs = {"nlayers": 3, "units": 8, "activation": "softplus", 
          "constraint": non_neg()}

# model = StdNeuralNetwork(**kwargs)
model = GradNeuralNetwork(**kwargs)
model.compile("adam", "mse", loss_weights=loss_weights)

#%% model calibration

epochs = 100

t1 = now()
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit(training_input, training_output, epochs=epochs, verbose=2)
print(f"It took {now() - t1} sec to calibrate the model.")

#%% result plots

save = False

if plotten:
    if data == "bathtub":
        plot_bathtub(model, h, save=save)

    elif data == "f1":
        plot_f1(model, h, save=save)    
        
    elif data == "f2":
        plot_f2(model, h, save=save)

