import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.special import gamma
import datetime


def b_fun(k, alpha):
    return (np.power(k+1, 1-alpha)-np.power(k, 1-alpha))/gamma(2-alpha)


alpha = 0.8
phi = 0.5
beta = 0.5
nu = 0.01

T = 2.0
N = 5001
h = T/(N-1)

a = 0.0
b = 1.0
M = 21
x = np.linspace(a, b, M)
dx = x[1]-x[0]

L  = sparse.diags([-2.0*np.ones(M-2), np.ones(M-3), np.ones(M-3)], [0, -1, 1])
Id = sparse.diags(np.ones(M-2), 0)

u = np.zeros((M, N)) #(x, t)
f = np.zeros_like(u)

u[:,0] = 1.0 + 0.0*x
u[0,0] = 0.0
u[-1,0] = 0.0

for n in range(1, N):
    halpha = np.power(h, 1-alpha)
    NTX = nu*h/np.square(dx)

    bb = b_fun(n-np.arange(1,n+1), alpha)

    A = (phi + bb[-1]*halpha)*Id - NTX*L

    y = np.sum(u[:,1:n], axis=1)

    f1 = (NTX*L) @ y[1:-1]
    f2 = halpha*(u[1:-1,1:n]@bb[:-1])
    f3 = (phi + beta*(((n*h)**(1-alpha))/gamma(2-alpha)))*u[1:-1,0]
    f4 = h*np.sum(f[1:-1,1:n], axis=1)

    fBC = np.zeros(M-2)    
    fBC[0]  = (h*nu/(dx**2))*(u[0,0] + y[0])
    fBC[-1] = (h*nu/(dx**2))*(u[-1,0] + y[-1])

    u[1:-1,n] = sparse.linalg.spsolve(A, f1 - f2 + f3 + f4 + fBC)
    u[0,n]  = u[0,0]
    u[-1,n] = u[-1,0]



# SAVING SOLUTION
# params = {
#     'T':T,
#     'N':N,
#     'a':a,
#     'b':b,
#     'M':M,
#     'alpha':alpha,
#     'phi':phi,
#     'beta':beta,
#     'nu':nu,
# }
# np.savez(f'./results/u_{datetime.datetime.now().strftime("%Y-%d-%m_%H-%M-%S")}.npz', u=u, f=f, params=params)



# exit()
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
l1, = plt.plot(x, u[:,0], '.-')
l2, = plt.plot(x, f[:,0], '--')
axfreq = plt.axes([0.25, 0.15, 0.65, 0.03])
freq = Slider(axfreq, 'Time', 0, T-0.5*h, valinit=0, valstep=h)
freq.on_changed(update)
plt.show()