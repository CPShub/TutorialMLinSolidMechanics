import  numpy as np

def get_deviator(field):
    N = field.shape[2]
    deviator = np.zeros((3, 3, N))
    print(deviator.shape)

    for i in range(0, N):
        tr = np.trace(field[:, :, i])
        deviator[:, :, i] = field[:, :, i] - 1/3*tr*np.eye(3)

    return deviator


def get_equivalent(field, type):
    N = field.shape[2]
    dev = get_deviator(field)
    eq = np.zeros(N)

    if type == "strain":
        k = 2/3
    elif type == "stress":
        k = 3/2
    else:
        print("no valid field type.")

    for i in range(0, N):
        eq[i] = np.sqrt(k*np.tensordot(dev[:, :, i], dev[:, :, i]))

    return eq





