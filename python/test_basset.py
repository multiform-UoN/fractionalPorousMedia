"""
test_basset.py
==============

Lightweight but rigorous verification suite for the Basset solver.

- test_initial_condition_none : smoke test of the default (eigenfunction) path.
- test_manufactured_solution_temporal_convergence : Method of Manufactured
  Solutions (MMS) that isolates the temporal error of the Product-Integration
  rule and asserts that the observed order is consistent with the theoretical
  O(Δt) convergence of the first-order rectangular PI scheme.
- test_manufactured_solution_phi_neq_one : MMS regression test pinning the
  *independence* of phi0 (storage) and beta0 (memory) coefficients, i.e. that
  the fractional matrix B carries beta0 alone and not beta0*phi0.
- test_manufactured_solution_pde_spatial : full-PDE MMS with a space-dependent
  exact solution and nonzero diffusion, exercising the L assembly and the
  Robin/Neumann boundary rows that the temporal-only tests do not touch.

These tests also serve as executable documentation of the correct usage of
BassetConfig (especially the forcing signature and BC encoding).

Run with either:
    python -m unittest test_basset -v
    pytest test_basset.py -v
"""

import unittest
import numpy as np
from scipy.special import gamma
import main_BassetEqn as mb

class TestBassetSolver(unittest.TestCase):
    def test_initial_condition_none(self):
        """Test that solver executes with the default eigenfunction initial condition (None)."""
        cfg = mb.BassetConfig(T=0.5, delta_time=0.1, N_space=21, print_eigenvalue=False)
        time, mesh_x, u = mb.solve(cfg)
        self.assertEqual(len(time), 6)
        self.assertEqual(len(mesh_x), 21)
        self.assertEqual(u.shape, (21, 6))
        # Ensure solution is not all zeros
        self.assertTrue(np.max(np.abs(u)) > 0)

    def test_manufactured_solution_temporal_convergence(self):
        """
        Verify the solver accuracy using the Method of Manufactured Solutions (MMS).
        We choose the manufactured solution u_exact(x, t) = t^2, which is independent of space.
        By setting both left and right boundary conditions to Neumann (u_x = 0) and setting
        all spatial operator coefficients to 0, we eliminate all spatial discretisation and BC errors.
        This isolates the temporal integration error of the Product Integration rule.
        
        We verify first-order convergence rate in time (O(dt)).
        """
        # Physical parameters
        alpha = 0.5
        beta0 = 0.5
        nu0 = 0.0  # Zero out spatial operators to eliminate spatial error
        vel0 = 0.0
        reac0 = 0.0

        # Exact solution
        u_exact = lambda t, x: t**2 * np.ones_like(x)
        
        def forcing_func(t, x):
            # t has shape (N_time,), x has shape (N_space,)
            T_grid, X_grid = np.meshgrid(t, x, indexing='xy')
            
            # d_t u = 2 * t
            dt_u = 2.0 * T_grid
            
            # d_t^alpha u = 2 / gamma(3 - alpha) * t^(2 - alpha)
            dt_alpha_u = (2.0 / gamma(3.0 - alpha)) * (T_grid**(2.0 - alpha))
            
            return dt_u + beta0 * dt_alpha_u

        # Run the solver for multiple time-step sizes
        N_space = 11
        T = 1.0
        dts = [0.02, 0.01, 0.005]
        errors = []

        for dt in dts:
            cfg = mb.BassetConfig(
                xL=0.0,
                xR=1.0,
                T=T,
                N_space=N_space,
                delta_time=dt,
                advection="upwind",
                alpha=alpha,
                phi0=1.0,
                beta0=beta0,
                nu0=nu0,
                vel0=vel0,
                reac0=reac0,
                zetaL=1.0,  # Neumann left
                xiL=0.0,
                zetaR=1.0,  # Neumann right
                xiR=0.0,
                initial_condition=lambda x: np.zeros_like(x), # u(x, 0) = 0
                forcing=forcing_func,
                print_eigenvalue=False
            )
            time, mesh_x, u_num = mb.solve(cfg)
            
            # Compute exact solution at final time T = 1.0
            u_ex = u_exact(T, mesh_x)
            
            # Measure L2 error on interior nodes
            err = np.sqrt(np.mean((u_num[:, -1] - u_ex)**2))
            errors.append(err)
            print(f"dt = {dt:.4f}, L2 Error = {err:.3e}")

        # Compute convergence rates
        rates = []
        for i in range(len(errors) - 1):
            rate = np.log2(errors[i] / errors[i+1])
            rates.append(rate)
            print(f"Rate from dt={dts[i]} to dt={dts[i+1]} is {rate:.3f}")
            # The PI scheme is 1st-order, so rate should be close to 1.0
            self.assertTrue(rate > 0.9) # Expect rate near 1.0

    def test_manufactured_solution_phi_neq_one(self):
        """
        MMS regression test for the porosity coefficient phi0 != 1.

        This guards against a subtle coupling bug: phi0 (coefficient of the
        standard time derivative) and beta0 (coefficient of the fractional
        derivative) must be *independent*.  The semi-discrete model is

            phi0 * d_t u  +  beta0 * d_t^alpha u  =  L u + f,

        so the fractional matrix B must carry beta0 ONLY (not beta0*phi0).
        If B were accidentally built from the phi-scaled mass matrix, the
        interior fractional coefficient would become beta0*phi0 and the
        manufactured solution below would NOT be recovered -> this test would
        fail.  It therefore pins the corrected, paper-consistent model.

        Setup: same space-independent exact solution u_exact = t^2 with
        Neumann/Neumann BCs and zeroed spatial operators, but with phi0 = 2.
        The forcing is built directly from the independent-coefficient model.
        """
        alpha = 0.5
        beta0 = 0.5
        phi0 = 2.0          # <-- the parameter under test (must stay decoupled from beta0)

        # Manufactured solution (independent of x).
        u_exact = lambda t, x: t**2 * np.ones_like(x)

        def forcing_func(t, x):
            # t -> (N_time,), x -> (N_space,); return shape (N_space, N_time).
            T_grid, _ = np.meshgrid(t, x, indexing='xy')
            # f = phi0 * d_t u  +  beta0 * d_t^alpha u, with
            #   d_t u       = 2 t
            #   d_t^alpha u = 2 / Gamma(3 - alpha) * t^(2 - alpha)
            dt_u = 2.0 * T_grid
            dt_alpha_u = (2.0 / gamma(3.0 - alpha)) * (T_grid ** (2.0 - alpha))
            return phi0 * dt_u + beta0 * dt_alpha_u

        N_space = 11
        T = 1.0
        dts = [0.02, 0.01, 0.005]
        errors = []

        for dt in dts:
            cfg = mb.BassetConfig(
                T=T, N_space=N_space, delta_time=dt,
                alpha=alpha, phi0=phi0, beta0=beta0,
                nu0=0.0, vel0=0.0, reac0=0.0,        # no spatial operator
                zetaL=1.0, xiL=0.0,                   # Neumann left
                zetaR=1.0, xiR=0.0,                   # Neumann right
                initial_condition=lambda x: np.zeros_like(x),  # u(x, 0) = 0
                forcing=forcing_func,
                print_eigenvalue=False,
            )
            _, mesh_x, u_num = mb.solve(cfg)
            err = np.sqrt(np.mean((u_num[:, -1] - u_exact(T, mesh_x)) ** 2))
            errors.append(err)
            print(f"[phi0={phi0}] dt = {dt:.4f}, L2 Error = {err:.3e}")

        # The error must (a) be small and (b) decrease at first order.  Both
        # only hold if B carries beta0 independently of phi0.
        self.assertLess(errors[-1], 1e-2)
        for i in range(len(errors) - 1):
            rate = np.log2(errors[i] / errors[i + 1])
            print(f"[phi0={phi0}] Rate dt={dts[i]}->{dts[i+1]} is {rate:.3f}")
            self.assertGreater(rate, 0.9)

    def test_manufactured_solution_pde_spatial(self):
        """
        Full-PDE MMS test exercising the spatial operator L and the BC rows.

        The temporal-only MMS tests zero out nu/vel/reac, so they never check
        the assembly of the diffusion stencil or the Robin boundary treatment.
        Here we use a genuinely space-dependent manufactured solution

            u_exact(x, t) = t^2 * cos(pi x),

        which satisfies homogeneous Neumann conditions exactly at both ends
        (u_x = -pi t^2 sin(pi x) vanishes at x = 0 and x = 1).  We include
        diffusion (nu0 = 1) and solve

            phi0 d_t u + beta0 d_t^alpha u = nu0 u_xx + f,

        with the forcing constructed from the exact solution:
            d_t u       = 2 t cos(pi x)
            d_t^alpha u = 2 / Gamma(3 - alpha) * t^(2 - alpha) cos(pi x)
            u_xx        = -pi^2 t^2 cos(pi x).

        Space and time are refined together (dx ~ dt), so the combined error
        is dominated by the first-order temporal scheme; we therefore assert a
        convergence rate close to 1.  A broken diffusion stencil or BC row
        would destroy this convergence (and inflate the absolute error).
        """
        alpha = 0.5
        beta0 = 0.5
        phi0 = 1.0
        nu0 = 1.0

        u_exact = lambda t, x: (t ** 2) * np.cos(np.pi * x)

        def forcing_func(t, x):
            T_grid, X_grid = np.meshgrid(t, x, indexing='xy')
            cos = np.cos(np.pi * X_grid)
            dt_u = 2.0 * T_grid * cos
            dt_alpha_u = (2.0 / gamma(3.0 - alpha)) * (T_grid ** (2.0 - alpha)) * cos
            u_xx = -(np.pi ** 2) * (T_grid ** 2) * cos
            return phi0 * dt_u + beta0 * dt_alpha_u - nu0 * u_xx

        T = 1.0
        # Joint refinement: N_space doubles (dx halves) as dt halves.
        cases = [(21, 0.02), (41, 0.01), (81, 0.005)]
        errors = []
        dts = []

        for N_space, dt in cases:
            cfg = mb.BassetConfig(
                T=T, N_space=N_space, delta_time=dt,
                alpha=alpha, phi0=phi0, beta0=beta0,
                nu0=nu0, vel0=0.0, reac0=0.0,        # diffusion only (symmetric stencil)
                zetaL=1.0, xiL=0.0,                   # Neumann left  (u_x = 0)
                zetaR=1.0, xiR=0.0,                   # Neumann right (u_x = 0)
                initial_condition=lambda x: np.zeros_like(x),  # u(x, 0) = 0
                forcing=forcing_func,
                print_eigenvalue=False,
            )
            _, mesh_x, u_num = mb.solve(cfg)
            err = np.sqrt(np.mean((u_num[:, -1] - u_exact(T, mesh_x)) ** 2))
            errors.append(err)
            dts.append(dt)
            print(f"[PDE] N={N_space}, dt={dt:.4f}, L2 Error = {err:.3e}")

        self.assertLess(errors[-1], 5e-2)
        for i in range(len(errors) - 1):
            rate = np.log2(errors[i] / errors[i + 1])
            print(f"[PDE] Rate (N={cases[i][0]}->{cases[i+1][0]}) is {rate:.3f}")
            self.assertGreater(rate, 0.9)


if __name__ == '__main__':
    unittest.main()
