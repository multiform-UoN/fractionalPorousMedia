import numpy as np
import scipy.io as sio

# sol = np.load('./u.npz', allow_pickle=True)
# print(sol['u'].shape)

matlab = sio.loadmat('../matlab/results/out.mat')
print(type(matlab))

for key in matlab.keys():
    print()
    print(key)
    print(matlab[key])