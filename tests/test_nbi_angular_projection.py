import unittest

import numpy as np
from numpy.polynomial.legendre import Legendre, legval
from scipy.integrate import quad

from solver_loader import load_solver


class NBIAngularProjectionTests(unittest.TestCase):
    def gaussian_solver(self, order, center=.7, width=.07):
        solver = load_solver()["FP"].__new__(load_solver()["FP"])
        solver.legmod = order
        solver.Ns = 2
        solver.nbi_file = None
        solver.nbi_enabled = False
        solver.nbi_mu_center = center
        solver.nbi_sigma = width
        solver.nbi_gaussian_beam_weights = np.array([1., .25])
        solver.sig_nbi()
        return solver

    def test_gaussian_coefficients_against_independent_adaptive_integrals(self):
        solver = self.gaussian_solver(61)
        for order in (0, 1, 2, 21, 41, 61):
            polynomial = Legendre.basis(order)
            integral, _ = quad(
                lambda mu: np.exp(-.5 * ((mu - .7) / .07)**2) * polynomial(mu),
                -1., 1., epsabs=1e-14, epsrel=1e-12, limit=200,
            )
            expected = (2 * order + 1) / 2 * integral
            self.assertAlmostEqual(solver.sig[0, order], expected, delta=2e-12)
            self.assertAlmostEqual(solver.sig[1, order], .25 * expected, delta=5e-13)

    def test_unresolved_angular_beam_keeps_its_area_and_mean(self):
        solver = self.gaussian_solver(61, center=.713, width=1e-6)
        self.assertAlmostEqual(solver.sig[0, 0], np.sqrt(2 * np.pi) * 1e-6 / 2,
                               delta=2e-18)
        self.assertAlmostEqual(solver.sig[0, 1] / solver.sig[0, 0], 3 * .713, delta=2e-13)

    def test_spectral_negativity_is_not_clipped_and_decreases_with_resolution(self):
        mu = np.linspace(-1., 1., 4001)
        expected = np.exp(-.5 * ((mu - .7) / .07)**2)
        coarse = legval(mu, self.gaussian_solver(21).sig[0])
        fine = legval(mu, self.gaussian_solver(61).sig[0])
        self.assertLess(coarse.min(), -1e-8)
        self.assertLess(np.max(np.abs(fine - expected)), .01 * np.max(np.abs(coarse - expected)))

    def test_external_piecewise_linear_projection_uses_native_knots(self):
        solver = load_solver()["FP"].__new__(load_solver()["FP"])
        solver.legmod = 61
        solver.Ns = 1
        solver.nbi_file = "fixture"
        solver.nbi_enabled = False
        solver.nbi_range_mu = 0
        solver.nbi_native_mu = np.array([-1., -.51, .197, .713, 1.])
        solver.S_nbi_native = np.array([[0., .3, .9, .2, 0.]])
        solver.sig_nbi()
        for order in (0, 1, 2, 21, 41, 61):
            polynomial = Legendre.basis(order)
            integral = sum(quad(
                lambda mu: np.interp(mu, solver.nbi_native_mu, solver.S_nbi_native[0]) * polynomial(mu),
                lower, upper, epsabs=1e-14, epsrel=1e-12, limit=200,
            )[0] for lower, upper in zip(solver.nbi_native_mu[:-1], solver.nbi_native_mu[1:]))
            self.assertAlmostEqual(solver.sig[0, order], (2 * order + 1) / 2 * integral,
                                   delta=2e-12)


if __name__ == "__main__":
    unittest.main()
