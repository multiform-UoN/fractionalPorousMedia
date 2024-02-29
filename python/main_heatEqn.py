import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

nu = 0.1
T = 2.0 
N = 401
h = T/(N-1)

a = 0.0
b = 1.0
M = 1001
x = np.linspace(a, b, M)
dx = x[1]-x[0]

L = sparse.diags([-2.0*np.ones(M-2), np.ones(M-3), np.ones(M-3)], [0, -1, 1])
Id = sparse.diags(np.ones(M-2),0)

A = sparse.csc_matrix(Id - (nu*h/np.square(dx))*L)

# Preallocation
u = np.zeros((M, N))
f = np.zeros_like(u)

# Setting of the known term
for i in range(N):
    t = i*h
    f[:,i] = t*np.sin(t)*np.cos(8*x)*np.tanh(x)

# Initial + Boundary Conditions
u[:,0] = 4*x*(1.0-x)*np.cos(9*x)
u[0,0] = 2
u[-1,0] = 0

fBC = np.zeros(M-2)
fBC[0]  = (nu*h/np.square(dx))*u[0,0]
fBC[-1] = (nu*h/np.square(dx))*u[-1,0]

for n in range(1,N):
    u[1:-1,n] = sparse.linalg.spsolve(A, u[1:-1,n-1] + h*f[1:-1,n] + fBC)
    u[0,n]  = u[0,0]
    u[-1,n] = u[-1,0] 



# PLOTTING
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def update(val):
    l1.set_ydata(u[:,int(u.shape[1]*val/T)])
    l2.set_ydata(f[:,int(u.shape[1]*val/T)])

fig, ax = plt.subplots()
plt.grid()
plt.subplots_adjust(bottom=0.35)
plt.ylim([np.min(u)-0.05*np.abs(np.max(u)), np.max(u)+0.05*np.abs(np.max(u))])
l1, = ax.plot(x, u[:,0], '-')
l2, = ax.plot(x, f[:,0], '-')
axfreq = plt.axes([0.25, 0.15, 0.65, 0.03])
freq = Slider(axfreq, 'Time', 0, T-0.0001*h, valinit=0, valstep=h)
freq.on_changed(update)
plt.show()