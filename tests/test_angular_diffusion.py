import unittest
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_jacobi, expit, jv

from solver_loader import load_solver


FP = load_solver()["FP"]


def angular_solver(ipheat=2, weighted=False, model="symmetric", legmod=4):
    solver = FP(
        ctx=SimpleNamespace(tempi=[1.0]), nptu=101, ipheat=ipheat,
        legmod=legmod, kperp=0.0, omega_ci=1.0, power=1.0,
        ispec=0, nbi_enabled=0, icrh_enabled=1, Ns=1,
    )
    solver.kamax = 1
    solver.ksup = 0
    solver.nfvrb = solver.legmod + 1
    solver.dqlu = np.zeros(solver.kamax + 1)
    solver.fjbess = np.zeros(102)
    solver.fctinv = np.ones(solver.kamax + 2)
    solver.dqlw = 2.0
    solver.dql00 = np.zeros((solver.nfvrb, solver.nfvrb))
    solver.dql01 = np.zeros_like(solver.dql00)
    solver.dql11 = np.zeros_like(solver.dql00)
    solver.trapping_model = model
    solver.coelgd_with_weight_optimized(enable_weight=weighted, uc=0.4, a=20.0)
    return solver


class AngularDiffusionTests(unittest.TestCase):
    def test_low_order_unweighted_analytic_integrals(self):
        # P0=1, P1=mu, P2=(3*mu**2-1)/2; integrate the polynomials exactly.
        expected = {
            1: {
                ("qleg00", 0, 0, 0): 4 / 3,
                ("qleg01", 0, 2, 0): 4 / 5,
                ("qleg01", 2, 0, 0): 0,
                ("qleg01", 1, 1, 0): 4 / 15,
                ("qleg11", 1, 1, 0): 4 / 15,
                ("qleg11", 2, 2, 0): 36 / 35,
                ("qleg01", 0, 2, 1): 12 / 35,
                ("qleg11", 2, 2, 1): 4 / 7,
            },
            2: {
                ("qleg00", 0, 0, 0): 16 / 15,
                ("qleg01", 0, 2, 0): 16 / 35,
                ("qleg01", 2, 0, 0): 0,
                ("qleg01", 1, 1, 0): 16 / 105,
                ("qleg11", 1, 1, 0): 16 / 105,
                ("qleg11", 2, 2, 0): 16 / 35,
                ("qleg01", 0, 2, 1): 16 / 105,
                ("qleg11", 2, 2, 1): 16 / 77,
            },
        }
        for ipheat, cases in expected.items():
            solver = angular_solver(ipheat=ipheat)
            for (name, n, m, k), value in cases.items():
                with self.subTest(ipheat=ipheat, coefficient=(name, n, m, k)):
                    self.assertAlmostEqual(getattr(solver, name)[n, m, k], value, delta=2e-8)

    def test_unweighted_parity_and_symmetric_coefficients(self):
        orders = np.arange(5)
        mixed_parity = (orders[:, None] + orders[None, :]) % 2 == 1
        for ipheat in (1, 2):
            solver = angular_solver(ipheat=ipheat)
            for name in ("qleg00", "qleg01", "qleg11"):
                with self.subTest(ipheat=ipheat, coefficient=name):
                    np.testing.assert_allclose(getattr(solver, name)[mixed_parity], 0, atol=1e-12)
            for name in ("qleg00", "qleg11"):
                coefficient = getattr(solver, name)
                np.testing.assert_allclose(coefficient, coefficient.swapaxes(0, 1), atol=1e-14)

    def test_weighted_mixed_parity_matches_independent_quadrature(self):
        # Adaptive quadrature of explicit low-order polynomials is independent of
        # the solver's sampled Simpson integration and Legendre construction.
        cases = (
            ("qleg00", 0, 1, lambda mu: mu),
            ("qleg01", 0, 1, lambda mu: mu),
            ("qleg01", 1, 2, lambda mu: 3 * mu**3),
            ("qleg11", 1, 2, lambda mu: 3 * mu**3),
        )
        for ipheat in (1, 2):
            solver = angular_solver(ipheat=ipheat, weighted=True, model="directional")
            for name, n, m, polynomial in cases:
                reference = quad(
                    lambda mu: expit(20 * (mu - 0.4)) * (1 - mu**2)**ipheat * polynomial(mu),
                    -1, 1, epsabs=1e-12, epsrel=1e-12,
                )[0]
                with self.subTest(ipheat=ipheat, coefficient=(name, n, m)):
                    self.assertGreater(abs(reference), 0.01)
                    self.assertAlmostEqual(getattr(solver, name)[n, m, 0], reference, delta=2e-8)

    def test_power_diagnostic_matches_matrix_energy_moment(self):
        # The odd polynomial f1=u**2 has a nonzero directional-weighted RF
        # contribution. Include both a constant kernel and oscillatory J1².
        for harmonic, xiperp, weighted in ((1, 0.0, False), (1, 0.0, True), (2, 6.0, True)):
            solver = angular_solver(ipheat=harmonic, weighted=weighted, model="directional")
            solver.xiperp = xiperp
            solver.tbu = np.linspace(0, 1, solver.nptu)
            solver.du = solver.tbu[1] - solver.tbu[0]
            solver.inimat(0)
            solver.inimat(1)
            solver.nspele = 1
            solver.anuisp = np.zeros(1)
            solver.tauisp = np.ones(1)
            solver.vthisp = np.ones(1)
            solver.phifun = np.zeros(1)
            solver.expuab = np.zeros(1)
            solver.thefun = np.zeros(1)
            solver.fgamm = np.zeros((solver.nfvrb, 2, solver.nfvrb, 2))
            solver.tbfu = np.zeros((solver.nptu, solver.nfvrb))
            solver.tbdfu = np.zeros_like(solver.tbfu)
            solver.tbfu[:, 1] = solver.tbu**2
            solver.tbdfu[:, 1] = 2 * solver.tbu
            solver.tbdqlk = np.full((solver.nptu, 1), solver.dqlw)
            solver.auxfun = np.zeros(solver.nptu)
            matrix_rate = 0.0
            for ip in range(solver.nptu - 1):
                for gauss, (z, weight) in enumerate(zip(solver.zgaus, solver.wgaus)):
                    u = solver.tbu[ip] + solver.du * z
                    solver.mastfp(ip, gauss)
                    # Exact polynomial fixture at the actual matrix quadrature
                    # points, with the energy test function's derivative 2*u.
                    matrix_rate -= 4 * np.pi * solver.du * weight * 2 * u * (
                        solver.fgamm[0, 1, 1, 0] * u**2
                        + solver.fgamm[0, 1, 1, 1] * 2 * u
                    )
            diagnostic_rate = solver.qlrate(solver.legmod)
            with self.subTest(harmonic=harmonic, xiperp=xiperp, weighted=weighted):
                self.assertAlmostEqual(diagnostic_rate, matrix_rate, delta=1e-12)
                if weighted:
                    self.assertGreater(abs(matrix_rate), 0.01)
                else:
                    self.assertAlmostEqual(matrix_rate, 0.0, delta=1e-12)

    def test_power_integrates_hermite_derivative_dofs_between_zero_endpoints(self):
        solver = angular_solver(ipheat=1)
        solver.nptu = 2
        solver.tbu = np.array([0.0, 1.0])
        solver.du = 1.0
        solver.inimat(0)
        # Values are zero at both ends, but the derivative DOFs define the
        # nonzero quadratic f0=u*(1-u). Its RF moment is known analytically.
        solver.tbfu = np.zeros((2, solver.nfvrb))
        solver.tbdfu = np.zeros_like(solver.tbfu)
        solver.tbdfu[:, 0] = [1.0, -1.0]
        expected = 4 * np.pi * solver.dqlw * (4 / 3) * (3 / 20)
        self.assertAlmostEqual(solver.qlrate(0), expected, delta=1e-9)

    def test_high_order_derivatives_match_independent_jacobi_integral(self):
        # P'_n = (n+1) P_(n-1)^(1,1) / 2 provides an independent stable basis.
        solver = angular_solver(weighted=True, model="directional", legmod=61)
        reference = quad(
            lambda mu: expit(20 * (mu - 0.4)) * (1 - mu**2)**2
            * (mu * 31 * eval_jacobi(60, 1, 1, mu))**2,
            -1, 1, epsabs=1e-10, epsrel=1e-10, limit=300,
        )[0]
        self.assertAlmostEqual(solver.qleg11[61, 61, 0], reference, delta=1e-9)
        self.assertGreater(reference, 3.0)
        self.assertLess(reference, 4.0)

    def test_symmetric_trapping_keeps_both_passing_directions(self):
        solver = angular_solver(weighted=True)
        np.testing.assert_allclose(solver.ql_weights, solver.ql_weights[::-1], atol=1e-14)
        orders = np.arange(solver.legmod + 1)
        odd_pairs = (orders[:, None] + orders[None, :]) % 2 == 1
        solver.xiperp = 6.0
        solver.qlcoef(1.0)
        for name in ("dql00", "dql01", "dql11"):
            np.testing.assert_array_equal(getattr(solver, name)[odd_pairs], 0.0)
        reference = quad(
            lambda mu: expit(20 * (abs(mu) - 0.4)) * (1 - mu**2)
            * jv(1, 6 * np.sqrt(1 - mu**2))**2,
            -1, 1, points=[0], epsabs=1e-12, epsrel=1e-12,
        )[0]
        self.assertAlmostEqual(solver.dql00[0, 0], solver.dqlw * reference, delta=1e-11)

    def test_complete_bessel_kernel_remains_positive_semidefinite(self):
        # These arguments made the old six-term expansion negative. Check the
        # full radial/angular gradient block, not only its diagonal entries.
        for harmonic, argument in ((1, 4.0), (1, 6.0), (2, 6.0), (2, 80.0)):
            solver = angular_solver(ipheat=harmonic, weighted=True, model="directional", legmod=21)
            solver.xiperp = argument
            solver.qlcoef(1.0)
            gradient_block = np.block([
                [solver.dql00, -solver.dql01],
                [-solver.dql01.T, solver.dql11],
            ])
            eigenvalues = np.linalg.eigvalsh(gradient_block)
            with self.subTest(harmonic=harmonic, argument=argument):
                self.assertGreaterEqual(eigenvalues.min(), -1e-12 * max(1.0, eigenvalues.max()))
                reference = quad(
                    lambda mu: expit(20 * (mu - 0.4)) * (1 - mu**2)
                    * jv(harmonic - 1, argument * np.sqrt(1 - mu**2))**2,
                    -1, 1, epsabs=1e-11, epsrel=1e-11, limit=300,
                )[0]
                self.assertAlmostEqual(solver.dql00[0, 0], solver.dqlw * reference, delta=1e-10)

    def test_bessel_table_handles_zero_and_large_arguments(self):
        solver = angular_solver()
        solver.njbess = 120
        solver.rjbess(80.0)
        self.assertGreaterEqual(len(solver.fjbess), 122)
        np.testing.assert_allclose(solver.fjbess[:3], jv(np.arange(3), 80.0), atol=1e-14)
        solver.rjbess(0.0)
        self.assertEqual(solver.fjbess[0], 1.0)
        np.testing.assert_array_equal(solver.fjbess[1:], 0.0)


if __name__ == "__main__":
    unittest.main()
