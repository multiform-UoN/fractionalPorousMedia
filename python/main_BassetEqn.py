# %%
# Importing
import numpy as np
from scipy import sparse
from scipy.special import gamma
import datetime

# %%
# Parameters

# Domain
xL = 0 # Domain left boundary
xR = 1.0 # Domain right boundary
T = 1 # Final time

# Discretisation
N_space = 100 # Number of space steps
advection = "upwind" # or "central" or "blended"
N_time = 100 # Number of time steps
time = np.linspace(0.0, T, N_time) # Time mesh

# Physical parameters
alpha = 0.5 # Fractional derivative order
# phi = 0.5 * np.ones(N_space)  # Porosity (standard derviative coefficient)
beta = 0.5 * np.ones(N_space)  # Fractional derivative coefficient
nu = 0.1 * np.ones(N_space)  # Diffusion coefficient field
vel = 1.1 * np.ones(N_space)  # Advection velocity field

# Boundary conditions
zetaL = 0.0 # Left boundary condition (Neumann coefficient)
xiL = 1.0 # Left boundary condition (Dirichlet coefficient)
zetaR = 1.0 # Left boundary condition (Neumann coefficient)
xiR = 0.0 # Left boundary condition (Dirichlet coefficient)
# NB Dirichlet values are imposed by the initial condition

# Initial conditions
# initial_condition = lambda x: (1 - x) * x
initial_condition = lambda x: x>0

# Forcing
# forcing = lambda t, x: np.outer((np.abs(xR-x)*np.exp(x))*(0.3 + np.sin(15*x)/4), np.exp(-t*0.5*np.sin(5*t)))
forcing = lambda t, x: np.outer(0.0*x, 0.0*t)

# %%
# Setup the problem

# Create Mesh
mesh_x = np.linspace(xL, xR, N_space) # Space mesh
dx = mesh_x[1]-mesh_x[0] # Space step

# Diffusion
d0 = -2.0 * nu
d0[0]  = 0
d0[-1] = 0
d1 = nu[1:]
d2 = nu[:-1]
d1[-1] = 0
d2[0]  = 0
L_diff = (1/np.square(dx))*sparse.diags([d0, d1, d2], [0, -1, 1]) # Laplacian


# Advection
d0 = vel
d1 = vel[1:]
d2 = vel[:-1]
d0[0]  = 0
d0[-1] = 0
d1[-1] = 0
d2[0]  = 0

L_a_r = -(1/dx)*sparse.diags([-d0,  d2], [0, 1]) # right advection
L_a_l = -(1/dx)*sparse.diags([ d0, -d1], [0, -1]) # left advection
L_a_c = -(1/dx)*sparse.diags([-d1,  d2], [-1, 1])/2 # Central advection

# Assemble full matrix
L = None

if advection == "upwind":
    L_a = sparse.diags((vel>0)*1.0, 0)@L_a_l + sparse.diags((vel<0)*1.0, 0)@L_a_r
    L = L_diff + L_a
elif advection == "central":
    L_a = L_a_c
    L = L_diff + L_a

# Mass (time derivative) matrix
M = np.eye(N_space)
M[0,1] =  -zetaL/(zetaL-xiL*dx)
M[-1,-2] = -zetaR/(zetaR+xiR*dx)
M = sparse.csr_matrix(M)

# Fractional derivative matrix
B = sparse.diags(beta, 0) @ M

# %%
# Time stepping


# useful definitions
dt = T/(N_time-1); print(f'dt = {dt}') # Time step
halpha = np.power(dt, 1-alpha)
f = forcing(np.linspace(0.0, T, N_time), mesh_x)
f[0,:] = 0.0
f[-1,:] = 0.0
b_fun = lambda k, alpha: (np.power(k+1, 1-alpha)-np.power(k, 1-alpha))/gamma(2-alpha)

# Set solution vector and initial condition
u = np.zeros((N_space, N_time)) #(x, t)
u[ :,0] = initial_condition(mesh_x)

# Time loop
for n in range(1, N_time):

    bb = b_fun(n-np.arange(1,n+1), alpha)
    A = M + (bb[-1]*halpha)*B - dt*L
    y = np.sum(u[:,1:n], axis=1)

    f1 = (M + B*(((n*dt)**(1-alpha))/gamma(2-alpha)))@u[:,0]
    f2 = - halpha * B @ (u[:,1:n] @ bb[:-1])
    f3 = (dt * L) @ y
    f4 = dt*np.sum(f[:,1:n], axis=1)
        
    u[:,n] = sparse.linalg.spsolve(A, f1 + f2 + f3 + f4)


# %%
# Saving solution

# SAVING SOLUTIONsvd_2020_demeaned_id20240227115752
if True:
    params = {
        'T':T,
        'N':N_time,
        'a':xL,
        'b':xR,
        'M':N_space,
        'alpha':alpha,
        # 'phi':phi,
        'beta':beta,
        'nu':nu,
    }
    # np.savez(f'./results/u_{datetime.datetime.now().strftime("%Y-%d-%m_%H-%M-%S")}.npz', u=u, f=f, params=params)
    np.savez(f'./results/out.npz', t=time, u=u, f=f, params=params)


# %%
# Plotting

if True:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox

    fig = plt.figure(figsize=(20,6))
    gs = gridspec.GridSpec(2, 2)

    def update_time(val):
        val = float(val)
        line_u_time.set_ydata(u[:,int(u.shape[1]*val/T)])
        line_f_time.set_ydata(f[:,int(u.shape[1]*val/T)])
        line_im_space.set_xdata([val,val])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_time.set_ylim([np.min(u)-0.05*np.abs(np.max(u)),  np.max(u)+0.05*np.abs(np.max(u))])
    line_u_time, = ax_time.plot(mesh_x, u[:,0], '-' , label='u')
    line_f_time, = ax_time.plot(mesh_x, f[:,0], '-', label='f')
    ax_time.set_xlabel(r'$x$')
    ax_time.set_ylabel(r'$u(\cdot,t)$')
    ax_time.legend()
    ax_time.grid()
    ax_time_slider = fig.add_axes([0.1, 0.04, 0.75, 0.04])
    slider_time = Slider(ax_time_slider, 'time', 0, T-0.99*dt, valinit=0, valstep=dt)
    slider_time.on_changed(update_time)
    

    def update_space(val):
        val = float(val)
        line_u_space.set_ydata(u[int(u.shape[0]*val/(xR-xL)),:])
        line_f_space.set_ydata(f[int(u.shape[0]*val/(xR-xL)),:])
        line_im_time.set_ydata([val,val])
    ax_space = fig.add_subplot(gs[1, 1])
    ax_space.set_ylim([np.min(u)-0.05*np.abs(np.max(u)), np.max(u)+0.05*np.abs(np.max(u))])
    line_u_space, = ax_space.plot(time, u[int(u.shape[0]/2),:], '-', label='u')
    line_f_space, = ax_space.plot(time, f[int(u.shape[0]/2),:], '-', label='f')
    ax_space.set_xlabel(r'$t$')
    ax_space.set_ylabel(r'$u(x,\cdot)$')
    ax_space.grid()
    ax_space_slider = fig.add_axes([0.1, 0.01, 0.75, 0.04])
    slider_space = Slider(ax_space_slider, 'space', xL, xR-0.99*dx, valinit=0, valstep=dx)
    slider_space.on_changed(update_space)


    ax_imshow = fig.add_subplot(gs[0, :])
    ims = ax_imshow.imshow(u, cmap='magma', aspect='auto', origin='lower', extent=(0.0, T, xL, xR))
    cbar = fig.colorbar(ims)
    cbar.set_label(r'$u(x,t)$')
    line_im_space, = ax_imshow.plot([0.0,0.0],[xL,xR], 'k--')
    line_im_time, = ax_imshow.plot([0.0,T],[0.5*(xL+xR),0.5*(xL+xR)], 'k--')
    ax_imshow.set_xlabel(r'$t$')
    ax_imshow.set_ylabel(r'$x$')

    # axbox = fig.add_axes([0.2, 0.06, 0.3, 0.07])
    # text_box = TextBox(axbox, "Time", textalignment="left")
    # text_box.on_submit(update)
    # text_box.set_val(0)
    # fig.align_labels()
    plt.subplots_adjust(
        top=0.975,
        bottom=0.17,
        left=0.045,
        right=0.99,
        hspace=0.275,
        wspace=0.155
    )
    plt.show()
