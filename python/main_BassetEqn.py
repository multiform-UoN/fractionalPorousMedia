import numpy as np
from scipy import sparse
from scipy.special import gamma
import datetime


b_fun = lambda k, alpha: (np.power(k+1, 1-alpha)-np.power(k, 1-alpha))/gamma(2-alpha)

f_fun = lambda t, x: np.outer((np.abs(xR-x)*np.exp(x))*(0.3 + np.sin(15*x)/4), np.exp(-t*0.5*np.sin(5*t)))
# f_fun = lambda t, x: np.outer(0.0*x, 0.0*t)

xL = 0 # Domain left boundary
xR = 1.0 # Domain right boundary
N_space = 2001 # Number of space steps
advection = "upwind" # or "central" or "blended"
T = 1 # Final time
N_time = 1001 # Number of time steps
dt = T/(N_time-1); print(f'dt = {dt}') # Time step
time = np.linspace(0.0, T, N_time) # Time mesh

alpha = 0.5 # Fractional derivative order

# phi = 0.5 * np.ones(N_space)  # Porosity (standard derviative coefficient)

beta = 0.5 * np.ones(N_space)  # Fractional derivative coefficient

nu = 0.01 * np.ones(N_space)  # Diffusion coefficient field

vel = 0.0 * np.ones(N_space)  # Advection velocity field

mesh_x = np.linspace(xL, xR, N_space) # Space mesh

dx = mesh_x[1]-mesh_x[0] # Space step

d0 = -2.0 * nu
d0[0]  = 0
d0[-1] = 0
d1 = nu[1:]
d2 = nu[:-1]
d1[-1] = 0
d2[0]  = 0
L_diff = (1/np.square(dx))*sparse.diags([d0, d1, d2], [0, -1, 1]) # Laplacian


d0 = vel
d1 = vel[1:]
d2 = vel[:-1]
d0[0]  = 0
d0[-1] = 0
d1[-1] = 0
d2[0]  = 0

L_a_r = (1/dx)*sparse.diags([-d0,  d2], [0, 1]) # right advection
L_a_l = (1/dx)*sparse.diags([ d0, -d1], [0, -1]) # left advection
L_a_c = (1/dx)*sparse.diags([-d1,  d2], [-1, 1])/2 # Central advection

L = None

if advection == "upwind":
    '''WE ASSUME POSITIVE VELOCITY FOR THE MOMENT'''
    # L_a = L_a_r + L_a_l
    L_a = L_a_l
    L = L_diff + L_a
elif advection == "central":
    L_a = L_a_c
    L = L_diff + L_a

# M = sparse.diags(np.ones(N_space), 0)
M = np.eye(N_space)
M[0,1] = 1e-6
M[-1,-2] = 1e-6
M = sparse.csr_matrix(M)


B = sparse.diags(beta, 0) @ M

u = np.zeros((N_space, N_time)) #(x, t)
# f = np.zeros_like(u)
f = f_fun(np.linspace(0.0, T, N_time), mesh_x)

# u[ :,0] = 1.0 + 0.0*mesh_x
u[ :,0] = (1-mesh_x)*mesh_x

# u[ 0,0] = 0.0
# u[-1,0] = 0.0

halpha = np.power(dt, 1-alpha)

for n in range(1, N_time):

    bb = b_fun(n-np.arange(1,n+1), alpha)

    A = M + (bb[-1]*halpha)*B - dt*L

    y = np.sum(u[:,1:n], axis=1)

    f1 = (dt * L) @ y
    f2 = halpha * B * (u[:,1:n] @ bb[:-1])
    f3 = (B*(((n*dt)**(1-alpha))/gamma(2-alpha)))*u[:,0]
    f4 = dt*np.sum(f[:,1:n], axis=1)

    # fBC = np.zeros(N_space-2)
    # # diffusion terms
    # fBC[0]  = dt*(u[0,0] + y[0])
    # fBC[-1] = dt*(u[-1,0] + y[-1])
    # if advection == "central":
    #     # advection terms (central)
    #     fBC[0]  += vel_eq*(u[0,0] + y[0])/2
    #     fBC[-1] -= vel_eq*(u[-1,0] +  y[-1])/2
    # elif advection == "upwind":
    #     # advection terms (upwind)
    #     fBC[0]  += vel_eq*(u[0,0] + y[0]) * (vel > 0)
    #     fBC[-1] += vel_eq*(u[-1,0] +  y[-1]) * (vel < 0)
        

    u[:,n] = sparse.linalg.spsolve(A, f1 - f2 + f3 + f4)
    # u[0,n]  = u[0,0]
    # u[-1,n] = u[-1,0]



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
