"""
Tutorial Machine Learning in Solid Mechanics (WiSe 23/24)
Task 4: Viscoelasticity
==================
Authors: Dominik K. Klein
         
01/2024
"""


# %%   
"""
Import modules

"""

from matplotlib import pyplot as plt
import numpy as np

colors = np.array([[(194/255, 76/255, 76/255)],
                   [(246/255, 163/255, 21/255)],
                   [(67/255, 83/255, 132/255)],
                   [(22/255, 164/255, 138/255)],
                   [(104/255, 143/255, 198/255)]])


def plot_data(eps, eps_dot, sig, omegas, As):
    
    
    n = len(eps[0])
    ns = np.linspace(0, 2*np.pi, n)
    
    plt.figure(dpi = 600, figsize = (10, 8))
    
    plt.subplot(2,2,1)
    
    plt.title('Data')
    
    for i in range(len(eps)):
                
        plt.plot(ns, sig[i], label = '$\\omega$: %.2f, $A$: %.2f' \
                 %(omegas[i], As[i]), color=colors[i], linestyle='--')
        plt.xlim([0, 2*np.pi])
        plt.ylabel('stress $\\sigma$')
        plt.xlabel('time $t$')
        plt.legend()
        
        
    plt.subplot(2,2,2)
        
    for i in range(len(eps)):
        
        plt.plot(eps[i], sig[i], color=colors[i], linestyle='--')
        plt.xlabel('strain $\\varepsilon$')
        plt.ylabel('stress $\\sigma$')
        
        
    plt.subplot(2,2,3)
    
    for i in range(len(eps)):
        
        plt.plot(ns, eps[i], color=colors[i], linestyle='--')
        plt.xlim([0, 2*np.pi])
        plt.xlabel('time $t$')
        plt.ylabel('strain $\\varepsilon$')
        
        
    plt.subplot(2,2,4)
    
    for i in range(len(eps)):
        
        plt.plot(ns, eps_dot[i], color=colors[i], linestyle='--')
        plt.xlim([0, 2*np.pi])
        plt.xlabel('time $t$')
        plt.ylabel(r'strain rate $\.{\varepsilon}$')
        
    


    plt.show()
    
    
def plot_model_pred(eps, sig, sig_m, omegas, As):
    
    
    n = len(eps[0])
    ns = np.linspace(0, 2*np.pi, n)
    
    plt.figure(dpi = 600, figsize = (10,4))
    
    plt.subplot(1,2,1)
    
    plt.title('Data: dashed line, model prediction: continuous line')
    
    for i in range(len(eps)):
                
        plt.plot(ns, sig[i], label = '$\\omega$: %.2f, $A$: %.2f'%(omegas[i], As[i]),linestyle='--', color=colors[i])
        plt.plot(ns, sig_m[i], color=colors[i])
        plt.xlim([0, 2*np.pi])
        plt.ylabel('stress $\\sigma$')
        plt.xlabel('time $t$')
        plt.legend()
        
        
    plt.subplot(1,2,2)
        
    for i in range(len(eps)):
        
        plt.plot(eps[i], sig[i], linestyle='--', color=colors[i])
        plt.plot(eps[i], sig_m[i], color=colors[i])
        plt.xlabel('strain $\\varepsilon$')
        plt.ylabel('stress $\\sigma$')

        
    


    plt.show()

