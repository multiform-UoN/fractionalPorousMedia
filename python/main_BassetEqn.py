import numpy as np
from scipy import sparse
from scipy.special import gamma
import datetime


b_fun = lambda k, alpha: (np.power(k+1, 1-alpha)-np.power(k, 1-alpha))/gamma(2-alpha)

# f_fun = lambda t, x: np.outer((np.abs(b-x)*np.exp(x))*(0.3 + np.sin(15*x)/4), np.exp(-t*0.5*np.sin(5*t)))
f_fun = lambda t, x: np.outer(0.0*x, 0.0*t)


alpha = 0.5 # Fractional derivative order
phi = 0.5   # Porosity (standard derviative coefficient)
beta = 0.5  # Fractional derivative coefficient
nu = 0.01  # Diffusion coefficient
vel = 0.  # Advection velocity
advection = "upwind" # or "central" or "blended"
T = 1 # Final time
N_time = 1001 # Number of time steps
dt = T/(N_time-1); print(f'dt = {dt}') # Time step
time = np.linspace(0.0, T, N_time) # Time mesh

a = 0 # Domain left boundary
b = 1.0 # Domain right boundary
N_space = 51 # Number of space steps
mesh_x = np.linspace(a, b, N_space) # Space mesh
dx = mesh_x[1]-mesh_x[0] # Space step

L  = sparse.diags([-2.0*np.ones(N_space-2), np.ones(N_space-3), np.ones(N_space-3)], [0, -1, 1]) # Laplacian
L_a_r = sparse.diags([-np.ones(N_space-2), np.ones(N_space-3)], [0, 1]) # right advection
L_a_l = sparse.diags([np.ones(N_space-2), -np.ones(N_space-3)], [0, -1]) # left advection
L_a_c = sparse.diags([-np.ones(N_space-3), np.ones(N_space-3)], [-1, 1])/2 # Central advection
if advection == "upwind":
    L_a = L_a_r*(vel < 0) + L_a_l*(vel > 0)
elif advection == "central":
    L_a = L_a_c

Id = sparse.diags(np.ones(N_space-2), 0)

u = np.zeros((N_space, N_time)) #(x, t)
# f = np.zeros_like(u)
f = f_fun(np.linspace(0.0, T, N_time), mesh_x)

u[ :,0] = 1.0 + 0.0*mesh_x
u[ 0,0] = 0.0
u[-1,0] = 1.0

halpha = np.power(dt, 1-alpha)
NUeq = nu*dt/np.square(dx)
vel_eq = vel*dt/dx

for n in range(1, N_time):

    bb = beta*b_fun(n-np.arange(1,n+1), alpha)

    A = (phi + bb[-1]*halpha)*Id - NUeq*L + vel_eq*L_a

    y = np.sum(u[:,1:n], axis=1)

    f1 = (NUeq*L) @ y[1:-1]
    f2 = halpha*(u[1:-1,1:n]@bb[:-1])
    f3 = (phi + beta*(((n*dt)**(1-alpha))/gamma(2-alpha)))*u[1:-1,0]
    f4 = dt*np.sum(f[1:-1,1:n], axis=1)

    fBC = np.zeros(N_space-2)
    # diffusion terms
    fBC[0]  = NUeq*(u[0,0] + y[0])
    fBC[-1] = NUeq*(u[-1,0] + y[-1])
    if advection == "central":
        # advection terms (central)
        fBC[0]  += vel_eq*(u[0,0] + y[0])/2
        fBC[-1] -= vel_eq*(u[-1,0] +  y[-1])/2
    elif advection == "upwind":
        # advection terms (upwind)
        fBC[0]  += vel_eq*(u[0,0] + y[0]) * (vel > 0)
        fBC[-1] += vel_eq*(u[-1,0] +  y[-1]) * (vel < 0)
        

    u[1:-1,n] = sparse.linalg.spsolve(A, f1 - f2 + f3 + f4 + fBC)
    u[0,n]  = u[0,0]
    u[-1,n] = u[-1,0]



# SAVING SOLUTIONsvd_2020_demeaned_id20240227115752
if True:
    params = {
        'T':T,
        'N':N_time,
        'a':a,
        'b':b,
        'M':N_space,
        'alpha':alpha,
        'phi':phi,
        'beta':beta,
        'nu':nu,
    }
    # np.savez(f'./results/u_{datetime.datetime.now().strftime("%Y-%d-%m_%H-%M-%S")}.npz', u=u, f=f, params=params)
    np.savez(f'./results/out.npz', t=time, u=u, f=f, params=params)



# PLOTTING
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
        line_u_space.set_ydata(u[int(u.shape[0]*val/(b-a)),:])
        line_f_space.set_ydata(f[int(u.shape[0]*val/(b-a)),:])
        line_im_time.set_ydata([val,val])
    ax_space = fig.add_subplot(gs[1, 1])
    ax_space.set_ylim([np.min(u)-0.05*np.abs(np.max(u)), np.max(u)+0.05*np.abs(np.max(u))])
    line_u_space, = ax_space.plot(time, u[int(u.shape[0]/2),:], '-', label='u')
    line_f_space, = ax_space.plot(time, f[int(u.shape[0]/2),:], '-', label='f')
    ax_space.set_xlabel(r'$t$')
    ax_space.set_ylabel(r'$u(x,\cdot)$')
    ax_space.grid()
    ax_space_slider = fig.add_axes([0.1, 0.01, 0.75, 0.04])
    slider_space = Slider(ax_space_slider, 'space', a, b-0.99*dx, valinit=0, valstep=dx)
    slider_space.on_changed(update_space)


    ax_imshow = fig.add_subplot(gs[0, :])
    ims = ax_imshow.imshow(u, cmap='magma', aspect='auto', origin='lower', extent=(0.0, T, a, b))
    cbar = fig.colorbar(ims)
    cbar.set_label(r'$u(x,t)$')
    line_im_space, = ax_imshow.plot([0.0,0.0],[a,b], 'k--')
    line_im_time, = ax_imshow.plot([0.0,T],[0.5*(a+b),0.5*(a+b)], 'k--')
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
