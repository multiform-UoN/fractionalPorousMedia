import numpy as np

sol = np.load('./u.npz', allow_pickle=True)

print(sol['u'].shape)

