import unittest
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad, simpson
from scipy.special import expit

from solver_loader import load_solver


FP = load_solver()["FP"]


def angular_solver(ipheat=2, weighted=False):
    solver = FP(
        ctx=SimpleNamespace(tempi=[1.0]), nptu=101, ipheat=ipheat,
        legmod=4, kperp=0.0, omega_ci=1.0, power=1.0,
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
            solver = angular_solver(ipheat=ipheat, weighted=True)
            for name, n, m, polynomial in cases:
                reference = quad(
                    lambda mu: expit(20 * (mu - 0.4)) * (1 - mu**2)**ipheat * polynomial(mu),
                    -1, 1, epsabs=1e-12, epsrel=1e-12,
                )[0]
                with self.subTest(ipheat=ipheat, coefficient=(name, n, m)):
                    self.assertGreater(abs(reference), 0.01)
                    self.assertAlmostEqual(getattr(solver, name)[n, m, 0], reference, delta=2e-8)

    def test_power_diagnostic_matches_matrix_energy_moment(self):
        # With kperp=0 and fundamental heating, D(u) is constant. The fixture's
        # odd component f1=u**2 has a nonzero weighted heating contribution.
        for weighted in (False, True):
            solver = angular_solver(ipheat=1, weighted=weighted)
            solver.tbu = np.linspace(0, 1, solver.nptu)
            solver.du = solver.tbu[1] - solver.tbu[0]
            solver.zgaus = np.array([0.0])
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
            weak_energy = np.zeros(solver.nptu)
            for ip, u in enumerate(solver.tbu):
                solver.mastfp(ip, 0)
                # Apply the energy test function u**2 to the actual n=0 weak
                # operator; its derivative is 2*u, with the heating sign.
                weak_energy[ip] = -2 * u * (
                    solver.fgamm[0, 1, 1, 0] * solver.tbfu[ip, 1]
                    + solver.fgamm[0, 1, 1, 1] * solver.tbdfu[ip, 1]
                )
            matrix_rate = 4 * np.pi * simpson(weak_energy, x=solver.tbu)
            diagnostic_rate = solver.qlrate(solver.legmod)
            with self.subTest(weighted=weighted):
                self.assertAlmostEqual(diagnostic_rate, matrix_rate, delta=1e-12)
                if weighted:
                    self.assertGreater(abs(matrix_rate), 0.1)
                else:
                    self.assertAlmostEqual(matrix_rate, 0.0, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
