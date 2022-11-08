# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:55:38 2022

@author: jonat
"""

"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein, Henrik Hembrock, Jonathan Stollberg
         
08/2022
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from data import bathtub_data, f1_data, f2_data

def plot_bathtub(model, history, save=False, file="bathtub"):
    xs, ys, xs_c, ys_c = bathtub_data()
    
    plt.figure(1, dpi=600)
    plt.semilogy(history.history['loss'], label='training loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend()
    if save:
        plt.savefig(file + "_loss.pdf")

    plt.figure(2, dpi=600)
    plt.scatter(xs_c[::10], ys_c[::10], c='green', label='calibration data')
    plt.plot(xs, ys, c='black', linestyle='--', label='bathtub function')
    plt.plot(xs, model.predict(xs), label='model', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    if save:
        plt.savefig(file + "_model.pdf")
    
    plt.show()
    
def plot_f1(model, history, save=False, file="f1"):
    xs, ys, zs, xs_c, ys_c, zs_c = f1_data()
    
    plt.figure(1, dpi=600)
    plt.semilogy(history.history['loss'], label='training loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend()
    if save:
        plt.savefig(file + "_loss.pdf")
    
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
    if save:
        plt.savefig(file + "_model.pdf")
    
    Z = zs.reshape((20,20))
    plt.figure(3, dpi=600)
    ax3 = plt.axes(projection="3d")
    ax3.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="data")
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        plt.savefig(file + "_data.pdf")
    
    plt.show()
    
def plot_f2(model, history, save=False, file="f2"):
    xs, ys, zs, grad, xs_c, ys_c, zs_c, grad_c = f2_data()
    
    plt.figure(1, dpi=600)
    plt.semilogy(history.history['loss'], label='training loss')
    plt.grid(which='both')
    plt.xlabel('calibration epoch')
    plt.ylabel('log$_{10}$ MSE')
    plt.legend()
    if save:
        plt.savefig(file + "_loss.pdf")
    
    zs_model = model.predict(np.hstack((xs, ys)))[0]
    X = xs.reshape((20,20))
    Y = ys.reshape((20,20))
    Z_MODEL = zs_model.reshape((20,20))
    plt.figure(2, dpi=600)
    ax2 = plt.axes(projection="3d")
    ax2.plot_surface(X, Y, Z_MODEL, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="model")
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        plt.savefig(file + "_model.pdf")
    
    Z = zs.reshape((20,20))
    plt.figure(3, dpi=600)
    ax3 = plt.axes(projection="3d")
    ax3.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="data")
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        plt.savefig(file + "_data.pdf")
        
    zs_model = model.predict(np.hstack((xs, ys)))[1][:,0]
    X = xs.reshape((20,20))
    Y = ys.reshape((20,20))
    Z_MODEL = zs_model.reshape((20,20))
    plt.figure(4, dpi=600)
    ax2 = plt.axes(projection="3d")
    ax2.plot_surface(X, Y, Z_MODEL, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="model")
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        plt.savefig(file + "_model_grad_x.pdf")
    
    Z = grad[:,0].reshape((20,20))
    plt.figure(5, dpi=600)
    ax3 = plt.axes(projection="3d")
    ax3.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="data")
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        plt.savefig(file + "_data_grad_x.pdf")
        
    zs_model = model.predict(np.hstack((xs, ys)))[1][:,1]
    X = xs.reshape((20,20))
    Y = ys.reshape((20,20))
    Z_MODEL = zs_model.reshape((20,20))
    plt.figure(6, dpi=600)
    ax2 = plt.axes(projection="3d")
    ax2.plot_surface(X, Y, Z_MODEL, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="model")
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        plt.savefig(file + "_model_grad_y.pdf")
    
    Z = grad[:,1].reshape((20,20))
    plt.figure(7, dpi=600)
    ax3 = plt.axes(projection="3d")
    ax3.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.25, 
                     edgecolors="black", label="data")
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        plt.savefig(file + "_data_grad_y.pdf")
    
    plt.show()