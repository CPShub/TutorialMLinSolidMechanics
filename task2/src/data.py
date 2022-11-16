#Order of data:
# Data format 
#   F11 F12 F13 F21 F22 F23 F31 F32 F33   P11 P12 P13 P21 P22 P23 P31 P32 P33   W

import numpy as np


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
        defgrad[:, :, i] = np.array([[data[i, 0], data[i, 1], data[i, 2]], [data[i, 3], data[i, 4], data[i, 5]], [data[i, 6], data[i, 7], data[i, 8]]])
        pkstress[:, :, i] = np.array([[data[i, 9], data[i, 10], data[i, 11]], [data[i, 12], data[i, 13], data[i, 14]], [data[i, 15], data[i, 16], data[i, 17]]])
        energy[0, i] = np.array(data[i, 18])


 

    return defgrad, pkstress, energy


if __name__ == "__main__":
    defgrad, pkstress, energy = load_data(loc_biaxial)
    print(defgrad[:, :, 0])