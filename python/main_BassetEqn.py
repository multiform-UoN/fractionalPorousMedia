#%%
# Import

import numpy as np
from scipy import sparse
from scipy.special import gamma
import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from matplotlib import colors

from scipy.sparse.linalg import eigs, spsolve

import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.size'] = 11
mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.sans-serif'] = 'Times'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['lines.solid_capstyle'] = 'round'



#%%
# Setup parameters

# DOMAIN OF INTEGRATION
xL = 0 # Domain left boundary
xR = 1.0 # Domain right boundary
T = 10 # Final time


# DISCRETISATION
N_space = 101 # Number of space steps
delta_time = 0.01 # Time step
advection = "upwind" #  "central" or "upwind"


# PHYSICAL PARAMETERS (constants, the solver can however handle fields)
alpha = 0.5 # Fractional derivative order
phi0   = 0.5 # Porosity (standard derviative coefficient) # Not used here
beta0  = 0.5 # Fractional derivative coefficient
nu0    = 1.0 # Diffusion coefficient
vel0   = 1.0 # Advection velocity
reac0  = 0.0 # Reaction coefficient


# BOUNDARY CONDITIONS
zetaL = 0.0 # Left boundary condition (Neumann coefficient)
xiL = 1.0   # Left boundary condition (Dirichlet coefficient)
zetaR = 1.0 # Left boundary condition (Neumann coefficient)
xiR = 0.0   # Left boundary condition (Dirichlet coefficient)
# NB Dirichlet values are imposed by the initial condition


# INITIAL CONDITIONS
# initial_condition = lambda x: (x > 0) # 0 at x=0, 1 elsewhere
initial_condition = None # use eigenfunction


# FORCING
# forcing = lambda t, x: np.outer((np.abs(xR-x)*np.exp(x))*(0.3 + np.sin(15*x)/4), np.exp(-t*0.5*np.sin(5*t)))
forcing = lambda t, x: np.outer(0.0 * x, 0.0 * t)


#%%
# Functions

def shifted_inverse_power_iteration(L, M, num_iterations=1000, tolerance=1e-10, tau=1.0):
    """
    Compute the largest eigenvalue and corresponding eigenvector using the inverse shifted power iteration method.

    Parameters:
    L (numpy.ndarray or scipy.sparse matrix): The input matrix L.
    M (numpy.ndarray or scipy.sparse matrix): The input matrix M.
    num_iterations (int): Maximum number of iterations.
    tolerance (float): Convergence tolerance.

    Returns:
    tuple: Dominant eigenvalue and corresponding eigenvector.
    """
    # Start with a random vector of 1s
    b_k = np.ones(L.shape[1]) # remove last DoF
    b_k[0] = 0.0
    eigenvalue_k = 0.0
    
    matrix = - L - tau * M 
    
    # # assume dirichlet on the lhs
    matrix[0,0] = matrix[1,1]
    matrix[0,1] = 0.0
    # # assume neumann on the rhs, add identity term
    matrix[-1, -1] = matrix[-2, -2]
    matrix[-1, -2] = -matrix[-1, -1]
    matrix[-1, -1] -= tau
    
    # print full matrix
    # print(matrix.toarray())
    
    for i in range(num_iterations):
        # solve forcing the neumann condition
        b_k[-1] = 0.0
        b_k1 = spsolve(matrix,b_k)
        b_k[-1] = b_k[-2]
        
        # Rayleigh quotient
        eigenvalue_k1 = b_k1 @ b_k / (b_k @ b_k)

        # print(f'Iteration: {i}, Eigenvalue: {eigenvalue_k1}')        
        # Check for convergence
        if np.abs(eigenvalue_k - eigenvalue_k1) < tolerance:
            break

        # Update eigenvalue and b_k
        b_k = b_k1 / (np.linalg.norm(b_k1))
        eigenvalue_k = eigenvalue_k1
        
            
    # The eigenvalue is the Rayleigh quotient
    eigenvalue = (1.0/eigenvalue_k + tau)
    
    return eigenvalue, b_k

def C(u0, u1, t1):
    return (u1 / u0 - 1.0)/t1

def early_time(time, u0, C):
    return u0 * (1.0 + C * time)

# Solve the problem
def solve():
    # Setup the problem

    # CREATE THE MESH
    mesh_x = np.linspace(xL, xR, N_space) # Space mesh
    dx = mesh_x[1] - mesh_x[0] # Space step
    N_time  = int(T/delta_time) + 1 # Number of time steps
    time = np.linspace(0.0, T, N_time) # Time mesh

    # CREATE THE COEFFICIENT FIELDS
    phi   = phi0 * np.ones(N_space)  # Porosity (standard derviative coefficient)
    beta  = beta0 * np.ones(N_space)  # Fractional derivative coefficient
    nu    = nu0 * np.ones(N_space)  # Diffusion coefficient field
    vel   = vel0 * np.ones(N_space)  # Advection velocity field
    reac  = reac0 * np.ones(N_space)  # Reaction coefficient field
    
    # REACTION OP.
    d0 = reac
    d0[0] = 0
    d0[-1] = 0
    L_react = sparse.diags(d0, 0)

    # DIFFUSION OP.
    d0 = -2.0 * nu
    d0[0]  = 0
    d0[-1] = 0
    d1 = nu[1:]
    d2 = nu[:-1]
    d1[-1] = 0
    d2[0]  = 0
    L_diff = (1.0 / np.square(dx)) * sparse.diags([d0, d1, d2], [0, -1, 1]) # Laplacian

    # ADVECTION OP.
    d0 = vel
    d1 = vel[1:]
    d2 = vel[:-1]
    d0[0]  = 0
    d0[-1] = 0
    d2[0]  = 0
    d1[-1] = 0
    L_adv_l =  (1.0 / dx) * sparse.diags([ d0, -d1], [ 0, -1])     # left advection
    L_adv_r =  (1.0 / dx) * sparse.diags([-d0,  d2], [ 0,  1])     # right advection
    L_adv_c = -(1.0 / dx) * sparse.diags([-d1,  d2], [-1,  1]) / 2 # central advection

    ## Assemble full matrix
    L = None
    if advection == "upwind":
        L_adv = sparse.diags((vel > 0) * 1.0, 0) @ L_adv_l + sparse.diags((vel < 0) * 1.0, 0) @ L_adv_r
        L = L_diff - L_adv + L_react
    elif advection == "central":
        L_adv = L_adv_c
        L = L_diff - L_adv + L_react
    

    # MASS MATRIX (time derivative)
    M = np.eye(N_space)
    M[0,1]   = -zetaL / (zetaL - xiL * dx) # correct for boundary conditions
    M[-1,-2] = -zetaR / (zetaR + xiR * dx) # correct for boundary conditions
    M = sparse.csr_matrix(M)

    # MASS MATRIX (fractional derivative)
    B = sparse.diags(beta, 0) @ M

    # FORCING TERM
    f = forcing(np.linspace(0.0, T, N_time), mesh_x)
    f[ 0,:] = 0.0
    f[-1,:] = 0.0
    
    # PRINT MATRICES
    # print(L.toarray())
    # print(M.toarray())

    # SOLVER
    ## Useful definitions
    # dt = T / (N_time - 1) # Time step
    dt = delta_time # Time step
    halpha = np.power(dt, 1 - alpha)
    b_fun = lambda k, alpha: (np.power(k + 1, 1 - alpha) - np.power(k, 1 - alpha)) / gamma(2 - alpha)

    ## Set solution vector and initial condition
    u = np.zeros((N_space, N_time))
    if initial_condition==None: # use eigenfunction as initial condition
        eigv,eigf = shifted_inverse_power_iteration(L, M, num_iterations=1000, tolerance=1e-10, tau=0.0)
        print(f'Principal eigenvalue: {eigv}')
        # print(f'Principal eigenvector: {eigf}')
        u[ :,0] = eigf
    else:
        u[ :,0] = initial_condition(mesh_x) # user-defined initial condition
        
    # print(u[:,0])

    ## Time loop
    for n in range(1, N_time):

        bb = b_fun(n - np.arange(1, n + 1), alpha)
        A = M + (bb[-1] * halpha) * B - dt * L
        y = np.sum(u[:,1:n], axis=1)

        f1 = (M + B * (((n * dt) ** (1 - alpha)) / gamma(2 - alpha))) @ u[:,0]
        f2 = -halpha * B @ (u[:,1:n] @ bb[:-1])
        f3 = (dt * L) @ y
        f4 = dt * np.sum(f[:,1:n], axis=1)

        u[:,n] = sparse.linalg.spsolve(A, f1 + f2 + f3 + f4)

    # print(A.toarray())
    # print("f1", f1)
    # print("f2", f2)
    # print("f3", f3)
    # print("f4", f4)

    return time, mesh_x, u



##############################################################################################################################



#%% Plotting parameters
vec_alpha = [0.1, 0.25, 0.5, 0.75, 0.9]
vec_dt    = [0.01, 0.0075, 0.005, 0.0025, 0.001]

SAVEFIG = True
COLORS_2D = False
FIXED_X = False
FIXED_T = False
VARIABLE_T = True
LATE_TIME = False
EARLY_TIME = False
INTERACTIVE = False



#%% Color plots
if COLORS_2D:
    print('Color plots')
    for index, val in enumerate(vec_alpha):
        fig = plt.figure(f'alpha = {val}', figsize=(6,5))
        # Assign new values to global variables
        alpha = val
        # Now call the `solve` function without arguments
        time, mesh_x, u = solve()
        ims = plt.imshow(u, cmap='twilight_shifted', aspect='auto', origin='lower', extent=(0.0, T, xL, xR), clim=(0,0.15))
        # T, X = np.meshgrid(time, mesh_x, indexing='xy')
        # ims = plt.contourf(T, X, u, cmap='jet', aspect='auto', origin='lower', extent=(0.0, T, xL, xR), levels=100)
        cbar = fig.colorbar(ims)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$x$')
        plt.xlim(right=0.5) 
        plt.tight_layout()
        if SAVEFIG:
            plt.savefig(f'./fig_colors_alpha={val}.pdf')
        else:
            plt.show()


#%% Fixed space
if FIXED_X:
    indexmap = [(0,0), (0,1), (1,0), (1,1)]
    print('Fixed space')
    # fig = plt.figure("Fixed space", figsize=(0.5*8.3,0.5*8.3))
    fig, axs = plt.subplots(2, 2, figsize=(8.3, 8.3), sharex=True, sharey=True)
    for iii, xval in enumerate([0.2, 0.4, 0.6, 0.8]):
        for index, val in enumerate(vec_alpha):
            alpha = val # Assign new values to global variables
            time, mesh_x, u = solve() # Now call the `solve` function without arguments
            axs[*indexmap[iii]].plot(time, u[mesh_x==xval, :].flatten(), label=rf'$\alpha={val}$')
    
        axs[*indexmap[iii]].set_title(f'$x={xval}$')
        axs[*indexmap[iii]].set_xlim([-0.05, 1.05])
        axs[*indexmap[iii]].grid()

    axs[1,0].set_xlabel('$t$')
    axs[1,1].set_xlabel('$t$')
    axs[0,0].set_ylabel('$u$')
    axs[1,0].set_ylabel('$u$')
    axs[0,0].legend()

    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(f'./fig_fixedx_varalpha.pdf')
    else:
        plt.show()


#%% Fixed time
if FIXED_T:
    print('Fixed time')
    fig = plt.figure('Fixed time', figsize=(6,6))
    for index, val in enumerate(vec_alpha):
        alpha = val # Assign new values to global variables
        time, mesh_x, u = solve() # Now call the `solve` function without arguments
        plt.plot(mesh_x, u[:, time==5].flatten(), label=rf'$\alpha={val}$')
    plt.xlabel(r'$x$')    
    plt.ylabel(r'$u$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(f'./fig_fixed_t=5.pdf')
    else:
        plt.show()



#%% Variable time
if VARIABLE_T:
    print('Variable time')
    # Second plot for fixed alpha and various times
    alpha = 0.5  # Fixed value of alpha
    time, mesh_x, u = solve()
    fig = plt.figure('Variable time', figsize=(4.15, 4.15))
    # Number of equispaced samples you want
    N = 5
    indices = np.linspace(0, len(time)-1, N, dtype=int)
    for itime in indices:        
        plt.plot(mesh_x, u[:, itime].flatten()/u[-1,itime], label=rf'$t={time[itime]}$')
    plt.xlabel(r'$x$')    
    plt.ylabel(r'$u$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(f'./fig_variable_t.pdf')
    else:
        plt.show()



#%% Late time
if LATE_TIME:
    print('Late time')
    
    asimptote_index = 100
    
    fig = plt.figure('Late time', figsize=(8.3,4))
    for index, val in enumerate(vec_alpha):
        # Assign new values to global variables
        alpha = val
        T = 100
        # Now call the `solve` function without arguments
        time, mesh_x, u = solve()
        plt.loglog(time[asimptote_index:], u[int(u.shape[0]/2), -1] * (time[asimptote_index:]/time[-1])**(-val), f'C{index}--', label=r'$C\,t^{-\alpha}$' + rf'   $\alpha={val}$')
        plt.loglog(time[asimptote_index:], u[int(u.shape[0]/2), asimptote_index:], f'C{index}-')
    plt.xlabel(r'$t$')    
    plt.ylabel(r'$u$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(f'./fig_late_time_var_alpha.pdf')
    else:
        plt.show()



#%% Early time
if EARLY_TIME:
    print('Early time')
    
    asimptote_index = 100
    fig = plt.figure('Early time -  var. alpha', figsize=(4.15, 4.15))
    for index, val in enumerate(vec_alpha):
        # Assign new values to global variables
        alpha = val
        delta_time = 0.001
        T = 0.1
        # Now call the `solve` function without arguments
        time, mesh_x, u = solve()
        xi = int(u.shape[0]/2)
        plt.plot(time[:asimptote_index], u[xi, :asimptote_index], f'C{index}-')
        plt.plot(time[:asimptote_index], 
        early_time(time[:asimptote_index], u[xi, 0], C(u[xi, 0], u[xi, 1], time[1])), f'C{index}--', label=rf'$\alpha = {val}$'
        )
    plt.xlabel(r'$t$')    
    plt.ylabel(r'$u$')
    plt.ylim(top=0.1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(f'./fig_early_time_var_alpha.pdf')
    else:
        plt.show()
    
    fig = plt.figure('Early time - var. dt', figsize=(4.15, 4.15))
    for index, val in enumerate(vec_dt):
        # Assign new values to global variables
        alpha = 0.5
        delta_time = val
        T = 0.1
        # Now call the `solve` function without arguments
        time, mesh_x, u = solve()
        xi = int(u.shape[0]/2)
        plt.plot(time[:asimptote_index], u[xi, :asimptote_index], f'C{index}-')
        plt.plot(time[:asimptote_index], 
        early_time(time[:asimptote_index], u[xi, 0], C(u[xi, 0], u[xi, 1], time[1])), f'C{index}--', label=rf'$\Delta t = {val}$'
        )
    plt.xlabel(r'$t$')    
    plt.ylabel(r'$u$')
    plt.ylim(top=0.1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(f'./fig_early_time_var_dt.pdf')
    else:
        plt.show()

####################################################################################
#%%
# Interactive plots

if INTERACTIVE:
    
    alpha = 0.5
    delta_time = 0.01
    asimptote_index = 50
    
    time, mesh_x, u = solve()
    
    def update_time(val):
        val = float(val)
        line_u_time.set_ydata(u[:,int(u.shape[1]*val/T)])
        # line_u_time.set_ydata(u[:,int(u.shape[1]*val/T)]/max(u[:,int(u.shape[1]*val/T)])) # normalised
        line_im_space1.set_xdata([val,val])
        line_im_space2.set_xdata([val,val])
        ax_time.set_ylabel(r'$u(x,t={})$'.format(np.round(val, 3)))

    def update_space(val):
        val = float(val)
        index = int(u.shape[0] * val / (xR - xL))
        line_u_space.set_ydata(u[index,:])
        line_latetime_space.set_ydata(u[index, -1] * (time[asimptote_index:]/time[-1])**(-alpha))
        line_earlytime_space.set_ydata(early_time(time[:asimptote_index], u[index, 0], C(u[index, 0], u[index, 1], time[1])))
        line_im_time1.set_ydata([val,val])
        line_im_time2.set_ydata([val,val])
        ax_space.set_ylabel(r'$u(x={},t)$'.format(np.round(val, 3)))


    fig = plt.figure(figsize=(20,6))
    gs = gridspec.GridSpec(2, 2)

    ax_time = fig.add_subplot(gs[1, 0])
    ax_time.set_ylim([np.min(u)-0.05*np.abs(np.max(u)),  np.max(u) + 0.05*np.abs(np.max(u))])
    line_u_time, = ax_time.plot(mesh_x, u[:,0], '-' , label='u', markersize=4)
    ax_time.set_xlabel(r'$x$')
    ax_time.set_ylabel(r'$u(x,t={})$'.format(0))
    ax_time.legend()
    ax_time.grid()
    
    ax_time_slider = fig.add_axes([0.1, 0.04, 0.75, 0.04])
    slider_time = Slider(ax_time_slider, 'time', 0, T-0.99*dt, valinit=0, valstep=dt)
    slider_time.on_changed(update_time)

    
    ax_space = fig.add_subplot(gs[1, 1])
    # ax_space.set_ylim([np.min(u)-0.05*np.abs(np.max(u)), np.max(u)+0.05*np.abs(np.max(u))])
    line_u_space,         = ax_space.loglog(time, u[int(u.shape[0]/2), :], '-', label='u')
    line_latetime_space,  = ax_space.loglog(time[asimptote_index:], u[int(u.shape[0]/2), -1] * (time[asimptote_index:]/time[-1])**(-alpha), 'r--', label=r'$C\,t^{-\alpha}$')
    line_earlytime_space, = ax_space.loglog(
        time[:asimptote_index], 
        early_time(time[:asimptote_index], u[int(u.shape[0]/2), 0], C(u[int(u.shape[0]/2), 0], u[int(u.shape[0]/2), 1], time[1])), 'r--', label=r'$u_0(1 + Ct)$')
    # ax_space.set_ylim([np.min(u), np.max(u)])
    ax_space.set_ylim(bottom=5e-4)
    ax_space.set_xlabel(r'$t$')
    ax_space.set_ylabel(r'$u(x={},t)$'.format(0))
    ax_space.legend()
    ax_space.grid()
    
    ax_space_slider = fig.add_axes([0.1, 0.01, 0.75, 0.04])
    slider_space = Slider(ax_space_slider, 'space', xL, xR-0.99*dx, valinit=0, valstep=dx)
    slider_space.on_changed(update_space)


    ax_imshow1 = fig.add_subplot(gs[0, 0])
    ims = ax_imshow1.imshow(u, cmap='jet', aspect='auto', origin='lower', extent=(0.0, T, xL, xR))
    cbar = fig.colorbar(ims)
    cbar.set_label(r'$u(x,t)$')
    line_im_space1, = ax_imshow1.plot([0.0,0.0],[xL,xR], 'k--')
    line_im_time1,  = ax_imshow1.plot([0.0,T],[0.5*(xL+xR),0.5*(xL+xR)], 'k--')
    ax_imshow1.set_xlabel(r'$t$')
    ax_imshow1.set_ylabel(r'$x$')
    
    ax_imshow2 = fig.add_subplot(gs[0, 1])
    ims = ax_imshow2.imshow(np.abs(u[:,1:] - np.outer(u[:, -1], (time[1:]/time[-1])**(-alpha))), cmap='jet', aspect='auto', origin='lower', extent=(0.0, T, xL, xR), norm=colors.LogNorm())
    cbar = fig.colorbar(ims)
    cbar.set_label(r'$u(x,t) - C\,t^{-\alpha}$')
    line_im_space2, = ax_imshow2.plot([0.0,0.0],[xL,xR], 'k--')
    line_im_time2,  = ax_imshow2.plot([0.0,T],[0.5*(xL+xR),0.5*(xL+xR)], 'k--')
    ax_imshow2.set_xlabel(r'$t$')
    ax_imshow2.set_ylabel(r'$x$')

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


##############################################################################################################################
# SAVING SOLUTION
# if True:
#     params = {
#         'T':T,
#         'N':N_time,
#         'a':xL,
#         'b':xR,
#         'M':N_space,
#         'alpha':alpha,
#         # 'phi':phi,
#         'beta':beta,
#         'nu':nu,
#     }
#     # np.savez(f'./results/u_{datetime.datetime.now().strftime("%Y-%d-%m_%H-%M-%S")}.npz', u=u, f=f, params=params)
#     np.savez(f'./results/out.npz', t=time, u=u, f=f, params=params)

# %%
