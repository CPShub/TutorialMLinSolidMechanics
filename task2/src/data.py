#Order of data:
# Data format 
#   F11 F12 F13 F21 F22 F23 F31 F32 F33   P11 P12 P13 P21 P22 P23 P31 P32 P33   W

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

loc = "./../data/calibration/"
loc = "./task2/data/calibration/"
loc_biaxial = loc + "biaxial.txt"
loc_shear = loc + "pure_shear.txt"
loc_uniax = loc + "uniaxial.txt"

def load_data(loc):
    data = np.loadtxt(loc)
    print("Shape of data:")
    print(data.shape)

    #get data count
    N = data.shape[0]


    #create empy arrays
    defgrad = np.zeros((3, 3, N))
    pkstress = np.zeros((3, 3, N))
    energy = np.zeros((1, N))

    for i, j in enumerate(data[:, 1]):
        defgrad[:, :, i] = np.array([[data[i, 0], data[i, 1], data[i, 2]], 
                                     [data[i, 3], data[i, 4], data[i, 5]], 
                                     [data[i, 6], data[i, 7], data[i, 8]]])
        
        pkstress[:, :, i] = np.array([[data[i, 9], data[i, 10], data[i, 11]], 
                                      [data[i, 12], data[i, 13], data[i, 14]], 
                                      [data[i, 15], data[i, 16], data[i, 17]]])
        
        energy[0, i] = np.array(data[i, 18])

    return defgrad, pkstress, energy

def invariants(F):
    # define transersely isotropic structural tensro
    G = np.array([[4.0, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
    
    # compute right Cauchy-Green tensor
    C = np.dot(F.T, F)
    
    # compute invariants
    I1 = np.trace(C)
    J = np.det(F)
    I4 = np.trace(np.dot(C, G))
    cofC = np.det(C)*np.inv(C)
    I5 = np.trace(np.dot(cofC, G))
    
    return I1, J, I4, I5

def read_invariants(file):
    data = np.loadtxt(file)
    I1 = data[:,0]
    J = data[:,1]
    I4 = data[:,2]
    I5 = data[:,3]
    
    return I1, J, I4, I5

def strain_energy_by_invariants(I1, J, I4, I5):
    W = 8*I1 + 10*J**2 - 56*np.log(J) + 0.2*(I4**2 + I5**2) - 44
    return W

def strain_energy_by_F(F):
    I1, J, I4, I5 = invariants(F)
    return strain_energy_by_invariants(I1, J, I4, I5)

def piola_kirchhoff(F):
    with tf.GradientTape() as tape:
        tape.watch(F)
        W = strain_energy_by_F
    P = tape.gradient(W, F)
    
    return P


def plot_load_path(F, P):
    # plot stress and strain in normal direction
    F11, F22, F33 = F[0,0,:], F[4,4,:], F[8,8,:]
    P11, P22, P33 = P[0,0,:], P[4,4,:], P[8,8,:]
    
    normal_deformation = np.array([F11, F22, F33])
    normal_stress = np.array([P11, P22, P33])
    
    fig1, ax1 = plt.subplots()
    for f, p in zip(normal_deformation, normal_stress):
        ax1.plot(f[0], p[0])
        ax1.plot(f[1], p[1])
        ax1.plot(f[2], p[2])
        
        ax1.set(xlabel="deformation gradient",
               ylabel="first Piola-Kirchhoff stress")
        ax1.grid()
        
    # plot stress and strain in shear direction
    F12, F13, F21, F23, F31, F32 = (F[1,1,:], F[2,2,:], F[3,3,:], F[5,5,:],
                                    F[6,6,:], F[7,7,:])
    P12, P13, P21, P23, P31, P32 = (P[1,1,:], P[2,2,:], P[3,3,:], P[5,5,:],
                                    P[6,6,:], P[7,7,:])
    
    shear_deformation = np.array([F12, F13, F21, F23, F31, F32])
    shear_stress = np.array([P12, P13, P21, P23, P31, P32])
    
    fig2, ax2 = plt.subplots()
    for f, p in zip(shear_deformation, shear_stress):
        ax2.plot(f[0], p[0])
        ax2.plot(f[1], p[1])
        ax2.plot(f[2], p[2])
        ax2.plot(f[3], p[3])
        ax2.plot(f[4], p[4])
        ax2.plot(f[5], p[5])
        
        ax2.set(xlabel="deformation gradient",
               ylabel="first Piola-Kirchhoff stress")
        ax2.grid()
        
    plt.show()
    
if __name__ == "__main__":
    
    defgrad, pkstress, energy = load_data(loc_biaxial)
    print(defgrad[:, :, 0])