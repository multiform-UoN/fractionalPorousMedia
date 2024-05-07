import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


fmat = sio.loadmat('../matlab/results/out.mat')
fnp = np.load('./results/out.npz')

u_mat = fmat['u']
u_np  = fnp['u']
t_np  = fnp['t']

plt.figure()
plt.semilogy(t_np, np.linalg.norm(u_mat-u_np, axis=0), '-')
plt.xlabel(r'$t$')
plt.ylabel(r'$\|u_{mat}(t)-u_{py}(t)\|_2$')
plt.grid()
plt.show()