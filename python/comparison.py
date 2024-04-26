import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


fmat = sio.loadmat('../matlab/results/out.mat')
fnp = np.load('./results/out.npz')

u_mat = fmat['u']
u_np  = fnp['u']

plt.figure()
plt.semilogy(np.linalg.norm(u_mat, axis=1)-np.linalg.norm(u_np, axis=1), '.-')
plt.grid()
plt.show()