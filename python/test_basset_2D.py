"""
test_basset_2D.py
=================

Verification suite for the 2D time-fractional Basset transport solver and the
coupled steady Darcy flow solver in ``main_BassetEqn_2D``.

- test_darcy_flux_conservation : the harmonic-mean two-point flux (TPFA)
  pressure solve must produce a discretely divergence-free interface-flux
  field, even across a sharp permeability barrier.  This pins the consistency
  between the pressure operator and the velocity reconstruction.
- test_darcy_velocity_direction : with a left-to-right pressure drop the
  reconstructed velocity must point in +x and be reduced inside a
  low-permeability barrier.
- test_flushing_actually_flushes : the coupled flushing test configuration
  (clean-water Dirichlet inlet) must remove mass from the column; a regression
  guard against the inlet being frozen at the saturated value.
- test_smooth_permeability_reduces_nodal_divergence : smooth_permeability()
  must reduce the nodal central-difference divergence by at least an order of
  magnitude for a 1000:1 contrast, without changing the face-flux conservation.

Run with either:
    python -m unittest test_basset_2D -v
    pytest test_basset_2D.py -v
"""

import unittest
import numpy as np
import main_BassetEqn_2D as m2


def _face_flux_divergence(N, dx, dy, K):
    """
    Discrete divergence of the harmonic-mean interface fluxes (the conservative
    TPFA flux balance).  Returns the maximum |div| over interior cells.

    This is the physically meaningful measure of a divergence-free Darcy field:
    for div(K grad p)=0 the net face flux out of every interior cell must
    vanish, independently of any cell-centred velocity reconstruction.
    """
    p, _, _ = m2.solve_darcy_flow(N, N, dx, dy, K)
    hm = m2._harmonic_mean
    d = np.zeros((N, N))
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            Fxr = -hm(K[i, j], K[i + 1, j]) * (p[i + 1, j] - p[i, j]) / dx
            Fxl = -hm(K[i, j], K[i - 1, j]) * (p[i, j] - p[i - 1, j]) / dx
            Fyt = -hm(K[i, j], K[i, j + 1]) * (p[i, j + 1] - p[i, j]) / dy
            Fyb = -hm(K[i, j], K[i, j - 1]) * (p[i, j] - p[i, j - 1]) / dy
            d[i, j] = (Fxr - Fxl) / dx + (Fyt - Fyb) / dy
    return np.abs(d[1:-1, 1:-1]).max()


class TestDarcy2D(unittest.TestCase):
    def _barrier_field(self, N):
        mx = np.linspace(0.0, 1.0, N)
        X, Y = np.meshgrid(mx, mx, indexing='ij')
        K = np.ones((N, N))
        K[(X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.15 ** 2] = 1e-3
        return mx, X, Y, K

    def test_darcy_flux_conservation(self):
        """Interface-flux divergence must be ~0 for homogeneous AND barrier K."""
        N = 41
        dx = dy = 1.0 / (N - 1)

        K_hom = np.ones((N, N))
        self.assertLess(_face_flux_divergence(N, dx, dy, K_hom), 1e-9)

        _, _, _, K_bar = self._barrier_field(N)
        div_bar = _face_flux_divergence(N, dx, dy, K_bar)
        print(f"[darcy] barrier interface-flux max|div| = {div_bar:.2e}")
        self.assertLess(div_bar, 1e-9)

    def test_darcy_velocity_direction(self):
        """Flow must go inlet->outlet (+x) and slow down inside the barrier."""
        N = 41
        dx = dy = 1.0 / (N - 1)
        _, _, _, K = self._barrier_field(N)
        _, vx, vy = m2.solve_darcy_flow(N, N, dx, dy, K)

        # Net flow is in +x everywhere (no recirculation for this setup).
        self.assertGreater(vx[1:-1, 1:-1].min(), -1e-9)
        # Barrier centre velocity is far smaller than the open-channel value.
        c = N // 2
        v_barrier = np.hypot(vx[c, c], vy[c, c])
        v_open = np.hypot(vx[c, 2], vy[c, 2])   # near the bottom wall, outside barrier
        self.assertLess(v_barrier, 0.1 * v_open)

    def test_flushing_actually_flushes(self):
        """
        Regression test for the inlet initial condition: with a clean-water
        Dirichlet inlet the saturated column must lose mass over time.  If the
        inlet column were initialised to the saturated value (u=1) the scheme
        would freeze it there and no flushing would occur (constant solution).
        """
        N_x = N_y = 21
        cfg = m2.BassetConfig2D(
            T=0.6, delta_time=0.02, N_x=N_x, N_y=N_y,
            nu0=0.01, vel0_x=1.0, vel0_y=0.0,
            zetaL_x=0.0, xiL_x=1.0,   # clean-water Dirichlet inlet (u=0)
            zetaR_x=1.0, xiR_x=0.0,   # Neumann outlet
            zetaL_y=1.0, xiL_y=0.0, zetaR_y=1.0, xiR_y=0.0,
            initial_condition=lambda x, y: m2._saturated_with_clean_inlet(N_x, N_y),
        )
        _, _, _, u = m2.solve_2D(cfg)

        mass0 = u[:, :, 0].sum()
        mass_end = u[:, :, -1].sum()
        print(f"[flush] mass {mass0:.1f} -> {mass_end:.1f}")
        # The inlet must hold the clean-water value, and mass must drop clearly.
        self.assertAlmostEqual(u[0, N_y // 2, -1], 0.0, places=6)
        self.assertLess(mass_end, 0.85 * mass0)

    def test_smooth_permeability_reduces_nodal_divergence(self):
        """
        smooth_permeability() must substantially reduce the nodal divergence
        artefact at a sharp permeability interface while preserving face-flux
        conservation.

        Background: the face-flux field (TPFA) is always conservative to
        machine precision regardless of K sharpness.  However, the nodal
        velocities that the FD advection stencil sees are averages of two
        opposing face fluxes; across a 1000:1 permeability jump these averages
        carry a large discontinuity, measured by the central-difference nodal
        divergence.  Smoothing K before the Darcy solve spreads the transition
        over ~3 cells, reducing the nodal artefact by >10x.  The face-flux
        conservation must remain intact after smoothing.
        """
        N = 41
        dx = dy = 1.0 / (N - 1)
        mx = np.linspace(0.0, 1.0, N)
        X, Y = np.meshgrid(mx, mx, indexing='ij')

        K_sharp = np.ones((N, N))
        K_sharp[(X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.15 ** 2] = 1e-3

        def nodal_divmax(vx, vy):
            d = ((vx[2:, 1:-1] - vx[:-2, 1:-1]) / (2 * dx) +
                 (vy[1:-1, 2:] - vy[1:-1, :-2]) / (2 * dx))
            return np.abs(d).max()

        # Baseline: sharp K (no smoothing)
        _, vx_sharp, vy_sharp = m2.solve_darcy_flow(N, N, dx, dy, K_sharp)
        div_sharp = nodal_divmax(vx_sharp, vy_sharp)

        # After K-smoothing with sigma=1.5
        K_smooth = m2.smooth_permeability(K_sharp, sigma=1.5)
        _, vx_sm, vy_sm = m2.solve_darcy_flow(N, N, dx, dy, K_smooth)
        div_smooth = nodal_divmax(vx_sm, vy_sm)

        print(f"[smooth] nodal max|div| sharp={div_sharp:.2e}  smooth={div_smooth:.2e}  "
              f"ratio={div_sharp / div_smooth:.1f}x")

        # Smoothing must reduce nodal div by at least 10x.
        self.assertGreater(div_sharp / div_smooth, 10.0)

        # Face-flux conservation must be preserved after smoothing.
        self.assertLess(_face_flux_divergence(N, dx, dy, K_smooth), 1e-9)

        # Barrier still clearly lower permeability than the surrounding medium.
        c = N // 2
        self.assertLess(K_smooth[c, c], 0.1 * K_smooth[0, 0])


if __name__ == '__main__':
    unittest.main()
