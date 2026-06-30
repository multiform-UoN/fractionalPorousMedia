"""
main_BassetEqn.py
=================

Reference Python implementation of the numerical scheme for the time-fractional
Basset-type transport PDE arising from micro-macro homogenisation of solute
transport in heterogeneous porous media.

The scheme follows:
    Berardi, Garrappa, Icardi, Nuca
    "From microstructure to memory: Basset-type fractional transport models
     in porous media"

Core discretisation:
  * Finite differences in space (centred diffusion, upwind/central advection)
  * Vectorial Basset formulation so that the same (∂t + β ∂t^α) operator
    applies to both the interior PDE and the Robin boundary conditions.
  * First-order implicit rectangular product-integration (PI) rule in time
    (fractional Adams / L1-type method) for the Caputo derivative.
  * Fast implementation via pre-factorised LU + incremental history summation.

Mathematical model (semi-discrete form):
    M ∂_t u + B ∂_t^α u = L u + f

where
    - M is the (phi-scaled on the interior) mass matrix incorporating Robin BC corrections,
    - B = diag(β) @ M_bc, built from the *un-phi-scaled* BC mass matrix so that the
      fractional coefficient is exactly β, independent of φ,
    - L assembles diffusion + advection + reaction.

See the paper for derivation of the Basset form from immobile-region memory.

Usage
-----
    from main_BassetEqn import BassetConfig, solve

    cfg = BassetConfig(alpha=0.5, T=1.0, delta_time=0.01)
    t, x, u = solve(cfg)

    # or with overrides (backward compatible)
    t, x, u = solve(alpha=0.3, T=10.0, initial_condition=None)

Physical parameters
-------------------
alpha : fractional order (0 < alpha < 1)
phi0  : coefficient of the standard time derivative (often the mobile porosity).
        It scales ONLY the interior standard time derivative.  When phi0 != 1
        the interior equations are:
            phi0 * ∂t u + beta0 * ∂t^α u = spatial_operator(u) + f
        i.e. phi0 and beta0 are *independent* coefficients, exactly as in the
        paper.  This is achieved by building B from the un-phi-scaled mass
        matrix M_bc (see the M/B assembly block in solve()).  Boundary rows
        are left unscaled by phi0, matching the paper's treatment of the
        Robin expressions.
beta0 : strength of the fractional memory term (coefficient of ∂t^α u),
        independent of phi0.
nu0, vel0, reac0 : coefficients for diffusion, advection, reaction (can be
        made spatially varying by modifying the fields inside solve if desired).

BC encoding (Robin form)
------------------------
    zeta * u_x + xi * u = 0   at the boundary
Special cases:
    zeta=0, xi=1  -> Dirichlet u=0
    zeta=1, xi=0  -> Neumann u_x=0
The implementation keeps all degrees of freedom and lets the modified mass
rows enforce the BCs under the Basset operator.

"""

# Import
import numpy as np
from scipy import sparse
from scipy.special import gamma
from scipy.sparse.linalg import eigs, spsolve, factorized
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import colors
import matplotlib as mpl

# -----------------------------------------------------------------------------
# Global Matplotlib setup (applied once on import)
# -----------------------------------------------------------------------------
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.size'] = 11
mpl.rcParams['lines.solid_capstyle'] = 'round'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'

# Try to use LaTeX for nice paper-quality labels; fall back gracefully
# if no LaTeX distribution is available on the system.
try:
    mpl.rcParams['text.usetex'] = True
    # Dummy plot forces actual rendering so we can catch missing latex
    fig_test, ax_test = plt.subplots()
    ax_test.text(0.5, 0.5, r'$x$')
    fig_test.canvas.draw()
    plt.close(fig_test)
except Exception:
    mpl.rcParams['text.usetex'] = False


# =============================================================================
# Configuration dataclass
# =============================================================================

@dataclass
class BassetConfig:
    """
    Structured configuration for the 1D time-fractional Basset solver.

    All fields have sensible defaults that reproduce the paper examples.
    The solver accepts either a BassetConfig instance or keyword overrides
    for backward compatibility with older calling code.

    Extensive documentation of each group of parameters is provided below.
    """

    # -------------------------------------------------------------------------
    # DOMAIN OF INTEGRATION
    # -------------------------------------------------------------------------
    xL: float = 0.0
    """Left boundary of the spatial domain [xL, xR]."""

    xR: float = 1.0
    """Right boundary of the spatial domain."""

    T: float = 10.0
    """Final simulation time (integration from t=0 to t=T)."""

    # -------------------------------------------------------------------------
    # SPATIAL AND TEMPORAL DISCRETISATION
    # -------------------------------------------------------------------------
    N_space: int = 101
    """Number of spatial grid points (including the two boundary points)."""

    delta_time: float = 0.01
    """Time step size Δt. Total steps = int(T / delta_time) + 1."""

    advection: str = "upwind"
    """Advection discretisation: "upwind" (1st-order, stable) or "central"."""

    # -------------------------------------------------------------------------
    # PHYSICAL / MODEL PARAMETERS
    # -------------------------------------------------------------------------
    alpha: float = 0.5
    """Order of the Caputo fractional derivative (0 < α < 1)."""

    phi0: float = 1.0
    """
    Coefficient multiplying the standard (integer-order) time derivative.

    In the interior the discrete system solves an equation of the form
        phi0 * (∂t + beta0 * ∂t^α) u = L u + f
    (phi0 factors both time terms because of how B is constructed from M).

    This corresponds to a mobile porosity or storage coefficient in the
    macro-scale model. Set to 1.0 to recover the exact form written in
    the numerical section of the paper. Boundary rows are deliberately
    left unscaled so that the Robin BC expressions receive the pure
    (∂t + β ∂t^α) operator as required by the vectorial-Basset reformulation.
    """

    beta0: float = 0.5
    """
    Coefficient of the fractional memory term (strength of the immobile-zone
    retention). Corresponds to the effective porosity of the slow regions.
    """

    nu0: float = 1.0
    """Diffusion (dispersion) coefficient. L_diff uses -nu0 * D²u/Dx²."""

    vel0: float = 1.0
    """Advection velocity. Positive = flow to the right."""

    reac0: float = 0.0
    """Linear reaction coefficient (usually 0 in the paper examples)."""

    # -------------------------------------------------------------------------
    # BOUNDARY CONDITIONS (Robin form)
    # -------------------------------------------------------------------------
    zetaL: float = 0.0
    """Left-boundary Neumann coefficient in  zeta*u_x + xi*u = 0 ."""

    xiL: float = 1.0
    """Left-boundary Dirichlet coefficient."""

    zetaR: float = 1.0
    """Right-boundary Neumann coefficient."""

    xiR: float = 0.0
    """Right-boundary Dirichlet coefficient."""

    # -------------------------------------------------------------------------
    # INITIAL CONDITION AND FORCING
    # -------------------------------------------------------------------------
    initial_condition: Optional[Callable[[np.ndarray], np.ndarray]] = None
    """
    Callable(x) -> u0(x).  If None (the default), the solver computes the
    principal eigenfunction of the spatial operator (via shifted inverse
    power iteration) and uses it as initial datum.  This produces the clean
    self-similar late-time behaviour shown in the paper.
    """

    forcing: Callable[[np.ndarray, np.ndarray], np.ndarray] = \
        lambda t, x: np.outer(0.0 * x, 0.0 * t)
    """
    Forcing function f(t, x) returning an array of shape (N_space, N_time)
    or broadcastable to it.  The default is zero forcing.
    The time grid passed as first argument is (N_time,); x is (N_space,).
    """

    # -------------------------------------------------------------------------
    # DIAGNOSTICS / OUTPUT CONTROL
    # -------------------------------------------------------------------------
    print_eigenvalue: bool = True
    """
    If True and initial_condition is None, print the principal eigenvalue
    found by the shifted inverse iteration.  Useful for the paper
    verification plots; set to False to keep output clean in loops or tests.
    """


# Functions

def shifted_inverse_power_iteration(L, M, num_iterations=1000, tolerance=1e-10, tau=1.0):
    """
    Compute the dominant eigenpair of the (generalised) spatial operator
    via shifted inverse power iteration.

    We solve  L v = λ M v   for the eigenvalue with largest |λ| (usually
    the least negative one for a diffusion-advection operator).  This
    eigenfunction is used as a high-quality initial condition that makes
    the late-time solution exactly proportional to the spatial mode
    (self-similar decay).

    The implementation contains a small amount of hard-coded BC surgery
    that matches the default (Dirichlet-left, Neumann-right) case used
    throughout the paper.  For other BC combinations the user should
    supply their own initial_condition.

    Matrix modifications are performed on a LIL copy to avoid the
    SparseEfficiencyWarning that would be emitted when assigning into a CSR.
    """
    # Start with a vector of ones (simple, reproducible).  Force first
    # component to zero because the left BC typically fixes the value.
    b_k = np.ones(L.shape[1])
    b_k[0] = 0.0
    eigenvalue_k = 0.0

    # Build the shifted matrix we actually invert:  -(L + τ M)
    # Negative sign is because we want the eigenvalue of largest magnitude
    # after the Rayleigh-quotient inversion.
    # Use LIL for cheap structural edits, then convert once to CSC.
    matrix = (-L - tau * M).tolil()

    # --- Boundary-condition "surgery" (matches paper defaults) ---
    # Left (Dirichlet-like): replace first row so that the algebraic
    # system is consistent with u_0 being slaved to u_1 (or forced to zero).
    matrix[0, 0] = matrix[1, 1]
    matrix[0, 1] = 0.0

    # Right (Neumann-like): replace last row by a discrete homogeneous
    # Neumann condition and add the shift contribution.
    matrix[-1, -1] = matrix[-2, -2]
    matrix[-1, -2] = -matrix[-1, -1]
    matrix[-1, -1] -= tau

    matrix_csc = matrix.tocsc()

    for i in range(num_iterations):
        # Enforce the Neumann condition on the right before the solve.
        b_k[-1] = 0.0
        b_k1 = spsolve(matrix_csc, b_k)
        b_k[-1] = b_k[-2]          # copy from interior (Neumann)

        # Rayleigh quotient gives 1/(λ - τ) approximately.
        eigenvalue_k1 = b_k1 @ b_k / (b_k @ b_k)

        if np.abs(eigenvalue_k - eigenvalue_k1) < tolerance:
            break

        b_k = b_k1 / np.linalg.norm(b_k1)
        eigenvalue_k = eigenvalue_k1

    # Recover original eigenvalue:  λ = 1/μ + τ   where μ = 1/(λ - τ)
    eigenvalue = (1.0 / eigenvalue_k + tau)
    return eigenvalue, b_k


# =============================================================================
# Small helper functions used by the plotting driver
# =============================================================================

def C(u0, u1, t1):
    """Slope parameter for the early-time linear asymptote u(t) ≈ u0*(1 + C t)."""
    return (u1 / u0 - 1.0) / t1


def solve(config: Optional[BassetConfig] = None, **kwargs):
    """
    Solve the 1D time-fractional Basset transport equation

        M ∂t u + B ∂t^α u = L u + f

    using a first-order implicit product-integration (rectangular) rule in time
    combined with standard second-order centred finite differences in space.

    The implementation follows exactly the vectorial-Basset formulation
    described in the paper so that the fractional operator is applied
    uniformly to interior equations and to the Robin boundary conditions.

    Parameters
    ----------
    config : BassetConfig, optional
        Fully populated configuration object.  If None a default instance
        is created.
    **kwargs
        Any field of BassetConfig can be overridden by keyword argument
        (e.g. solve(alpha=0.25, T=2.0, delta_time=0.005)).  This provides
        full backward compatibility with older scripts.

    Returns
    -------
    time : ndarray, shape (N_time,)
        Uniform time grid from 0 to T.
    mesh_x : ndarray, shape (N_space,)
        Uniform spatial grid.
    u : ndarray, shape (N_space, N_time)
        Numerical solution.  Columns are time snapshots.

    Notes on the discrete system
    ----------------------------
    After spatial discretisation we obtain the matrix ODE
        M u_t + B u_t^α = L u + f

    where
        M  = "mass" matrix with Robin corrections; interior rows multiplied
             by phi0 when phi0 != 1 (see BassetConfig.phi0).
        B  = diag(beta) @ M
        L  = diffusion + advection + reaction (boundary rows zeroed).

    The product-integration rule applied to the integrated form yields
    the linear system solved at every step:

        A u^n = f1 + f2 + f3 + f4

    with A constant (pre-factorised once).
    """
    # ------------------------------------------------------------------
    # 1. Configuration handling (supports both styles)
    # ------------------------------------------------------------------
    if config is None:
        config = BassetConfig()

    # Keyword overrides (used by the plotting driver and tests)
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # ------------------------------------------------------------------
    # 2. Mesh generation
    # ------------------------------------------------------------------
    mesh_x = np.linspace(config.xL, config.xR, config.N_space)
    dx = mesh_x[1] - mesh_x[0]
    N_time = int(config.T / config.delta_time) + 1
    time = np.linspace(0.0, config.T, N_time)
    dt = config.delta_time

    # ------------------------------------------------------------------
    # 3. Coefficient fields (constant-coefficient case)
    #    These can be made spatially varying by replacing the lines below
    #    with arrays of length N_space.
    # ------------------------------------------------------------------
    phi = config.phi0 * np.ones(config.N_space)
    beta = config.beta0 * np.ones(config.N_space)
    nu = config.nu0 * np.ones(config.N_space)
    vel = config.vel0 * np.ones(config.N_space)
    reac = config.reac0 * np.ones(config.N_space)

    # ------------------------------------------------------------------
    # 4. Spatial discretisation – reaction term
    # ------------------------------------------------------------------
    d0 = reac.copy()
    d0[0] = d0[-1] = 0
    L_react = sparse.diags(d0, 0)

    # ------------------------------------------------------------------
    # 5. Spatial discretisation – diffusion (centred, 2nd order)
    #    Boundary rows are zeroed; the BCs are carried by the mass matrix M.
    # ------------------------------------------------------------------
    d0 = -2.0 * nu
    d0[0] = d0[-1] = 0
    d1 = nu[1:]
    d2 = nu[:-1]
    d1[-1] = d2[0] = 0
    L_diff = (1.0 / dx**2) * sparse.diags([d0, d1, d2], [0, -1, 1])

    # ------------------------------------------------------------------
    # 6. Spatial discretisation – advection
    # ------------------------------------------------------------------
    d0 = vel.copy()
    d1 = vel[1:]
    d2 = vel[:-1]
    d0[0] = d0[-1] = d2[0] = d1[-1] = 0

    L_adv_l = (1.0 / dx) * sparse.diags([d0, -d1], [0, -1])   # backward diff (flow right)
    L_adv_r = (1.0 / dx) * sparse.diags([-d0, d2], [0, 1])    # forward diff (flow left)
    L_adv_c = -(1.0 / dx) * sparse.diags([-d1, d2], [-1, 1]) / 2

    if config.advection == "upwind":
        # Classical first-order upwind (stable for |Pe| large)
        L_adv = (sparse.diags((vel > 0) * 1.0, 0) @ L_adv_l +
                 sparse.diags((vel < 0) * 1.0, 0) @ L_adv_r)
        L = L_diff - L_adv + L_react
    else:
        # Central (may need stabilisation for high Péclet)
        L_adv = L_adv_c
        L = L_diff - L_adv + L_react

    # ------------------------------------------------------------------
    # 7. Mass matrices M and B
    #
    # The semi-discrete model is the generalised Basset equation
    #
    #       M ∂_t u  +  B ∂_t^α u  =  L u + f
    #
    # in which phi (the standard time-derivative coefficient, e.g. the
    # mobile porosity) and beta (the fractional memory coefficient) are
    # INDEPENDENT physical parameters, matching the paper:
    #
    #       phi ∂_t u  +  beta ∂_t^α u  =  spatial_operator(u) + f.
    #
    # Both M and B share the same *boundary-row* structure: the Robin
    # conditions are imposed in the vectorial Basset sense, so the operator
    # (∂_t + beta ∂_t^α) annihilates the discrete Robin expression on the
    # boundary rows.  The two matrices differ only in their interior scaling:
    #
    #   * M_bc      : identity interior + Robin boundary rows (no phi, no beta)
    #   * M = diag(phi) on the interior of M_bc  -> coefficient of ∂_t u
    #   * B = diag(beta) @ M_bc                  -> coefficient of ∂_t^α u
    #
    # NB: B is built from the *un-phi-scaled* M_bc on purpose, so that the
    # fractional term carries exactly beta (and not beta*phi).  Building it
    # from the phi-scaled M would couple the two coefficients and silently
    # change the model whenever phi != 1.
    # ------------------------------------------------------------------
    M_bc = np.eye(config.N_space)

    # Robin correction for the two boundary rows (first-order one-sided).
    # These rows are deliberately left unscaled by phi/beta below, so the
    # scaling applied afterwards only touches the interior equations.
    M_bc[0, 1] = -config.zetaL / (config.zetaL - config.xiL * dx)
    M_bc[-1, -2] = -config.zetaR / (config.zetaR + config.xiR * dx)

    # M: apply porosity scaling ONLY to interior equations (BC rows untouched).
    M_scaled = M_bc.copy()
    M_scaled[1:-1, :] = phi[1:-1, np.newaxis] * M_bc[1:-1, :]
    M = sparse.csr_matrix(M_scaled)

    # B: fractional mass matrix = diag(beta) @ M_bc.  Crucially this uses the
    # un-phi-scaled M_bc, so the interior coefficient of ∂_t^α u is exactly
    # beta (independent of phi), while the boundary rows still carry the
    # beta-weighted Robin structure required by the vectorial Basset form.
    B = sparse.diags(beta, 0) @ sparse.csr_matrix(M_bc)

    # ------------------------------------------------------------------
    # 8. Forcing term – zeroed at the artificial boundaries
    #    (the true boundary behaviour is enforced via M and B).
    # ------------------------------------------------------------------
    f = config.forcing(np.linspace(0.0, config.T, N_time), mesh_x)
    f[0, :] = f[-1, :] = 0.0

    # ------------------------------------------------------------------
    # 9. Prepare the constant coefficients of the PI rule
    # ------------------------------------------------------------------
    halpha = dt ** (1.0 - config.alpha)
    # b_k = [(k+1)^{1-α} - k^{1-α}] / Γ(2-α)   (rectangular weights)
    b_fun = lambda k, alpha: ((k + 1)**(1 - alpha) - k**(1 - alpha)) / gamma(2 - alpha)

    # ------------------------------------------------------------------
    # 10. Initial condition
    # ------------------------------------------------------------------
    u = np.zeros((config.N_space, N_time))
    if config.initial_condition is None:
        # Use the principal eigenfunction of the spatial operator.
        # This yields the cleanest late-time asymptotics (u(x,t) → c(t) * χ(x)).
        eigv, eigf = shifted_inverse_power_iteration(L, M, num_iterations=1000,
                                                     tolerance=1e-10, tau=0.0)
        if config.print_eigenvalue:
            print(f'Principal eigenvalue: {eigv}')
        u[:, 0] = eigf
    else:
        u[:, 0] = config.initial_condition(mesh_x)

    # ------------------------------------------------------------------
    # 11. Pre-factorisation (the most important performance optimisation)
    #
    # The matrix on the left-hand side
    #     A = M + b0*halpha*B - dt*L
    # is INDEPENDENT of the time step index n.  We factor it once.
    # ------------------------------------------------------------------
    b_0 = 1.0 / gamma(2.0 - config.alpha)          # weight for the newest term (k=0)
    A = M + (b_0 * halpha) * B - dt * L
    solve_A = factorized(A.tocsc())                # scipy.sparse.linalg.factorized

    # Pre-compute the constant contributions coming from u^0
    Mu0 = M @ u[:, 0]
    Bu0 = B @ u[:, 0]

    # ------------------------------------------------------------------
    # 12. History accumulators (O(Nx) per step instead of O(Nt*Nx))
    # ------------------------------------------------------------------
    Ly = np.zeros(config.N_space)      # will hold L @ sum_{j=1}^{n-1} u^j
    f_sum = np.zeros(config.N_space)   # will hold sum_{j=1}^n f^j   (inclusive fix)

    # Pre-compute all weights so we can do fast strided copies
    k_vals = np.arange(N_time)
    b_vals = b_fun(k_vals, config.alpha)

    # ------------------------------------------------------------------
    # 13. Main time-marching loop (Product Integration)
    # ------------------------------------------------------------------
    for n in range(1, N_time):
        # Incremental update of the integrated spatial term
        # (the current unknown u^n will be added on the left via -dt*L inside A)
        if n > 1:
            Ly += L @ u[:, n - 1]

        # Forcing integral – MUST include the value at the current time level
        # for consistency with the integrated formulation.
        f_sum += f[:, n]

        # Contribution coming from the initial condition (exact for the RL integral)
        t_n = n * dt
        scalar_f1 = (t_n ** (1 - config.alpha)) / gamma(2 - config.alpha)
        f1 = Mu0 + scalar_f1 * Bu0

        # History convolution for the fractional integral (past steps only)
        # b_vals[n-1 : 0 : -1] gives exactly [b_{n-1}, b_{n-2}, ..., b_1]
        bb_history = b_vals[n-1:0:-1].copy()   # contiguous for BLAS
        history_conv = u[:, 1:n] @ bb_history
        f2 = -halpha * (B @ history_conv)

        f3 = dt * Ly          # integrated integer-order term from past
        f4 = dt * f_sum       # integrated forcing up to and including step n

        # One triangular solve with the precomputed factors
        u[:, n] = solve_A(f1 + f2 + f3 + f4)

    return time, mesh_x, u


# =============================================================================
# Plotting / Paper-Reproduction Driver
# =============================================================================
#
# Everything below runs ONLY when the file is executed directly
# (python main_BassetEqn.py).  Importing the module from tests or other
# scripts will never trigger figure generation.
#
# The various flags correspond one-to-one with figures that appear in the
# paper or the companion slides.
# =============================================================================

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Master switches – set any of these to True to regenerate figures
    # ------------------------------------------------------------------
    SAVEFIG = True          # write PDFs instead of plt.show()
    COLORS_2D = False       # space-time colour plots for several α
    FIXED_X = False         # u(t) at fixed x-locations, varying α
    FIXED_T = False         # u(x) at fixed time, varying α
    VARIABLE_T = True       # snapshots at different times (self-similarity)
    LATE_TIME = False       # long-time power-law decay u ~ t^{-α}
    EARLY_TIME = False      # early-time linear ramp validation
    INTERACTIVE = False     # live slider demo (requires display)

    vec_alpha = [0.1, 0.25, 0.5, 0.75, 0.9]
    vec_dt    = [0.01, 0.0075, 0.005, 0.0025, 0.001]

    # A single baseline config used for axis limits, etc.
    base_cfg = BassetConfig()

    #%% Color plots
    if COLORS_2D:
        print('Color plots')
        for index, val in enumerate(vec_alpha):
            fig = plt.figure(f'alpha = {val}', figsize=(6,5))
            time, mesh_x, u = solve(alpha=val)
            ims = plt.imshow(u, cmap='twilight_shifted', aspect='auto', origin='lower', extent=(0.0, base_cfg.T, base_cfg.xL, base_cfg.xR), clim=(0,0.15))
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
        fig, axs = plt.subplots(2, 2, figsize=(8.3, 8.3), sharex=True, sharey=True)
        for iii, xval in enumerate([0.2, 0.4, 0.6, 0.8]):
            for index, val in enumerate(vec_alpha):
                time, mesh_x, u = solve(alpha=val)
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
            time, mesh_x, u = solve(alpha=val)
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
        time, mesh_x, u = solve(alpha=0.5)
        fig = plt.figure('Variable time', figsize=(4.15, 4.15))
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
            # Evaluate over longer simulation window T=100
            time, mesh_x, u = solve(alpha=val, T=100)
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
            time, mesh_x, u = solve(alpha=val, delta_time=0.001, T=0.1)
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
            time, mesh_x, u = solve(alpha=0.5, delta_time=val, T=0.1)
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

    # Interactive plots
    if INTERACTIVE:
        time, mesh_x, u = solve(alpha=0.5, delta_time=0.01)
        dt = time[1] - time[0]
        dx = mesh_x[1] - mesh_x[0]
        asimptote_index = 50
        
        def update_time(val):
            val = float(val)
            line_u_time.set_ydata(u[:,int(u.shape[1]*val/base_cfg.T)])
            line_im_space1.set_xdata([val,val])
            line_im_space2.set_xdata([val,val])
            ax_time.set_ylabel(r'$u(x,t={})$'.format(np.round(val, 3)))

        def update_space(val):
            val = float(val)
            index = int(u.shape[0] * val / (base_cfg.xR - base_cfg.xL))
            line_u_space.set_ydata(u[index,:])
            line_latetime_space.set_ydata(u[index, -1] * (time[asimptote_index:]/time[-1])**(-0.5))
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
        slider_time = Slider(ax_time_slider, 'time', 0, base_cfg.T-0.99*dt, valinit=0, valstep=dt)
        slider_time.on_changed(update_time)

        ax_space = fig.add_subplot(gs[1, 1])
        line_u_space,         = ax_space.loglog(time, u[int(u.shape[0]/2), :], '-', label='u')
        line_latetime_space,  = ax_space.loglog(time[asimptote_index:], u[int(u.shape[0]/2), -1] * (time[asimptote_index:]/time[-1])**(-0.5), 'r--', label=r'$C\,t^{-\alpha}$')
        line_earlytime_space, = ax_space.loglog(
            time[:asimptote_index], 
            early_time(time[:asimptote_index], u[int(u.shape[0]/2), 0], C(u[int(u.shape[0]/2), 0], u[int(u.shape[0]/2), 1], time[1])), 'r--', label=r'$u_0(1 + Ct)$')
        ax_space.set_ylim(bottom=5e-4)
        ax_space.set_xlabel(r'$t$')
        ax_space.set_ylabel(r'$u(x={},t)$'.format(0))
        ax_space.legend()
        ax_space.grid()
        
        ax_space_slider = fig.add_axes([0.1, 0.01, 0.75, 0.04])
        slider_space = Slider(ax_space_slider, 'space', base_cfg.xL, base_cfg.xR-0.99*dx, valinit=0, valstep=dx)
        slider_space.on_changed(update_space)

        ax_imshow1 = fig.add_subplot(gs[0, 0])
        ims = ax_imshow1.imshow(u, cmap='jet', aspect='auto', origin='lower', extent=(0.0, base_cfg.T, base_cfg.xL, base_cfg.xR))
        cbar = fig.colorbar(ims)
        cbar.set_label(r'$u(x,t)$')
        line_im_space1, = ax_imshow1.plot([0.0,0.0],[base_cfg.xL,base_cfg.xR], 'k--')
        line_im_time1,  = ax_imshow1.plot([0.0,base_cfg.T],[0.5*(base_cfg.xL+base_cfg.xR),0.5*(base_cfg.xL+base_cfg.xR)], 'k--')
        ax_imshow1.set_xlabel(r'$t$')
        ax_imshow1.set_ylabel(r'$x$')
        
        ax_imshow2 = fig.add_subplot(gs[0, 1])
        ims = ax_imshow2.imshow(np.abs(u[:,1:] - np.outer(u[:, -1], (time[1:]/time[-1])**(-0.5))), cmap='jet', aspect='auto', origin='lower', extent=(0.0, base_cfg.T, base_cfg.xL, base_cfg.xR), norm=colors.LogNorm())
        cbar = fig.colorbar(ims)
        cbar.set_label(r'$u(x,t) - C\,t^{-\alpha}$')
        line_im_space2, = ax_imshow2.plot([0.0,0.0],[base_cfg.xL,base_cfg.xR], 'k--')
        line_im_time2,  = ax_imshow2.plot([0.0,base_cfg.T],[0.5*(base_cfg.xL+base_cfg.xR),0.5*(base_cfg.xL+base_cfg.xR)], 'k--')
        ax_imshow2.set_xlabel(r'$t$')
        ax_imshow2.set_ylabel(r'$x$')

        plt.subplots_adjust(
            top=0.975,
            bottom=0.17,
            left=0.045,
            right=0.99,
            hspace=0.275,
            wspace=0.155
        )
        
        plt.show()
