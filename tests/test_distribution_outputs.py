import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

from solver_loader import load_solver


FP = load_solver()["FP"]
plot_radial_distribution_with_contours = load_solver()["plot_radial_distribution_with_contours"]


class DistributionOutputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fpql-output-test-")
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "mode"
        self.output.mkdir()
        self.fp = FP.__new__(FP)
        self.fp.current_out_dir = str(self.output)
        self.fp.output_paths = {"read": str(self.output / "read.txt")}

    def distribution_fixture(self):
        fp = self.fp
        fp.ctx = SimpleNamespace(tempi=[1.0])
        fp.ispres = 0
        fp.nptu = 3
        fp.nmu = 3
        fp.tbu = np.array([0.0, 1.0, 2.0])
        fp.tbkev = fp.tbu**2
        fp.xmu = np.array([-1.0, 0.0, 1.0])
        fp.tfecmu = np.array([[0.1, -0.1, 0.0], [1e-8, 1e-8, -1e-12], [1e-20, 1e-30, 1e-40]])
        fp.f_min = 1e-10
        fp.f_max = 1.0
        fp.mu_min = -1.0
        fp.mu_max = 1.0
        fp.nbi_enabled = 0
        fp.icrh_enabled = 1
        fp.ipheat = 2
        fp.pwdens = 1.0
        fp.power = 1.0
        fp.kperp = 10.0
        fp.legmod = 1
        fp.du = 1.0
        fp.redpsi = np.array([1.0])
        fp.fnorm = 2.0
        fp.matrix_relative_residual = 1e-14
        fp.enable_trapping = 1
        fp.trapping_ucr = 0.4
        fp.trapping_a = 20.0
        return fp

    def test_disfun_preserves_raw_coefficients_and_relative_angular_amplitudes(self):
        fp = self.fp
        fp.nptu, fp.nfvrb, fp.legmod = 3, 2, 1
        fp.du = 0.5
        fp.fb = np.tile([1.0, 0.0, 0.25, 0.0], fp.nptu)
        original_solution = fp.fb.copy()
        fp.tbfu = np.zeros((3, 2))
        fp.tbdfu = np.zeros((3, 2))
        with contextlib.redirect_stdout(io.StringIO()):
            fp.disfun(1)
        self.assertAlmostEqual(fp.fnorm, 4 * np.pi / 3)
        np.testing.assert_array_equal(fp.raw_tbfu, np.tile([1.0, 0.25], (3, 1)))
        np.testing.assert_array_equal(fp.raw_tbdfu, np.zeros((3, 2)))
        np.testing.assert_allclose(fp.tbfu * fp.fnorm, fp.raw_tbfu)
        np.testing.assert_allclose(fp.tbdfu * fp.fnorm, fp.raw_tbdfu)
        np.testing.assert_array_equal(fp.fb, original_solution)
        np.testing.assert_allclose(fp.tbfu[:, 1] / fp.tbfu[:, 0], 0.25)

    def test_outdis_preserves_small_and_negative_distribution_values(self):
        fp = self.distribution_fixture()
        for name in ("hrate", "htperp", "htpar", "htemp", "ttailx", "ttailz", "ttblkx", "ttblkz"):
            setattr(fp, name, 1.0)
        for name in ("thrate", "thtprp", "thtpar", "thttot", "tavgtx", "tavgtz", "tavbkx", "tavbkz"):
            setattr(fp, name, np.zeros((1, 1)))
        fp.zdotn = 0.0
        fp.joutql = 1
        fp.ipsplf = np.ones((1, 2), dtype=int)
        fp.tbpl = np.zeros(3)
        fp.ref_maxwell = np.zeros(3)
        fp.idlout = 1
        fp.psinxt = 2.0
        original = fp.tfecmu.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            fp.outdis(1)
        np.testing.assert_array_equal(fp.tfecmu, original)

    def test_heatmap_masks_only_its_display_copy(self):
        fp = self.distribution_fixture()
        original = fp.tfecmu.copy()
        with plt.rc_context(), contextlib.redirect_stdout(io.StringIO()):
            fig, path = fp.plot_f_heatmap(fp.ctx, 1, plot_range="full")
            self.addCleanup(plt.close, fig)
        self.assertTrue(Path(path).is_file())
        np.testing.assert_array_equal(fp.tfecmu, original)

    def test_tail_interpolates_a_crossing_in_the_final_energy_interval(self):
        fp = self.distribution_fixture()
        fp.tbkev = np.array([0.0, 10.0, 20.0])
        fp.tfecmu = np.array([[4.0, 4.0, 4.0], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5]])
        tail_E, _, _, char_E = fp.extract_high_energy_tail_contour(1.0)
        np.testing.assert_allclose(tail_E, 50.0 / 3)
        self.assertAlmostEqual(char_E, 50.0 / 3)

    def test_debug_reports_threshold_and_domain_truncation(self):
        fp = self.distribution_fixture()
        fp.f_min = 1.0
        fp.tbkev = np.array([0.0, 10.0, 20.0])
        fp.tfecmu = np.array([[4.0, 4.0, 4.0], [2.0, 2.0, 2.0], [0.5, 1.5, -0.5]])
        fp.write_debug_info(1)
        debug = (self.output.parent / "debug.txt").read_text(encoding="utf-8")
        self.assertIn("尾端等值线特征", debug)
        self.assertIn("等值线阈值 f_min = 1.000000e+00", debug)
        self.assertIn("能域边界截断 = True", debug)
        self.assertIn("触及边界的 mu = [0.0]", debug)
        self.assertIn("负值网格点数 = 1", debug)
        self.assertIn("fnorm = 2.000000000000e+00", debug)

    def radial_fixture(self, radial_points=1):
        fp = self.fp
        fp.nptu, fp.nmu, fp.nrdpsi = 4001, 5, radial_points
        fp.ispres, fp.nspele, fp.ipheat = 1, 3, 2
        fp.r_E_1 = fp.r_E_2 = 20.0
        fp.ztempi = np.full((3, radial_points), 4.0)
        fp.zdeni = np.tile([[5.335], [0.165], [5.5]], (1, radial_points))
        fp.redpsi = np.linspace(0.1, 1, radial_points)
        fp.xmu = np.linspace(-1, 1, fp.nmu)
        fp.mu_min, fp.mu_max = -1.0, 1.0
        fp.nbi_energies_keV = [30.0]
        velocity = np.linspace(0, np.sqrt(20), fp.nptu)
        maxwellian = np.exp(-velocity**2) / np.pi**1.5
        fp.f_v_mu_psi = np.broadcast_to(
            maxwellian[:, None, None], (fp.nptu, fp.nmu, radial_points)
        ).copy()
        return fp

    def test_energy_spectrum_uses_selected_species_and_phase_space_jacobian(self):
        fp = self.radial_fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            spectrum, energy = fp.integrate_over_radius()
        # Exact isotropic Maxwell energy density per unit mu, with Ti=4 keV.
        ni = 0.165e19
        expected = ni * np.sqrt(energy) * np.exp(-energy / 4) / (np.sqrt(np.pi) * 4**1.5)
        np.testing.assert_allclose(spectrum[:, 2], expected, rtol=3e-5, atol=1.0)
        np.testing.assert_array_equal(spectrum[0], 0)
        density = simpson(simpson(spectrum, x=fp.xmu, axis=1), x=energy)
        self.assertAlmostEqual(density / ni, 1.0, delta=4e-5)
        header = (self.output / "12_energy_spectrum.txt").read_text().splitlines()[:3]
        self.assertIn("Energy_keV", header[0])
        self.assertIn("m^-3 keV^-1", header[0])
        self.assertIn("single radial point", header[1])

    def test_repeated_energy_mapping_preserves_the_original_velocity_distribution(self):
        fp = self.radial_fixture(radial_points=2)
        fp.ztempi[fp.ispres] = [2.0, 4.0]
        original = fp.f_v_mu_psi.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            first, first_energy = fp.integrate_over_radius()
            second, second_energy = fp.integrate_over_radius()
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_energy, second_energy)
        np.testing.assert_array_equal(fp.f_v_mu_psi, original)
        self.assertEqual(fp.radial_weight_model, "cylindrical approximation: dV proportional to r dr")
        self.assertTrue(np.all(fp.energy_spectrum_mu_psi[first_energy > 40, :, 0] == 0))

    def test_explicit_radial_volume_derivative_weights(self):
        fp = self.radial_fixture(radial_points=3)
        fp.redpsi = np.array([0.1, 0.3, 1.0])
        fp.zdeni[fp.ispres] = [1, 2, 4]
        with contextlib.redirect_stdout(io.StringIO()):
            spectrum, energy = fp.integrate_over_radius(dvolume_dpsi=[2, 3, 4])
        # Piecewise-linear integration gives weighted density 8.5/2.95 * 1e19.
        density = simpson(simpson(spectrum, x=fp.xmu, axis=1), x=energy)
        self.assertAlmostEqual(density / ((8.5 / 2.95) * 1e19), 1, delta=4e-5)
        self.assertEqual(fp.radial_weight_model, "explicit dV/dpsi weights")
        with self.assertRaisesRegex(ValueError, "weights"):
            fp.integrate_over_radius(dvolume_dpsi=[2, -1, 4])

    def test_radial_plot_highlights_beam_without_rescaling_data(self):
        fp = self.radial_fixture()
        original = fp.f_v_mu_psi.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            fig, _ = plot_radial_distribution_with_contours(
                fp, show_nbi_contour=False, show_nbi_marker=False,
                show_icrh_contour=False, show_icrh_marker=False, highlight_nbi_peak=True,
            )
        self.addCleanup(plt.close, fig)
        np.testing.assert_array_equal(fp.nbi_f_processed, fp.radial_energy_spectrum)
        np.testing.assert_array_equal(fp.f_v_mu_psi, original)
        self.assertIn('keV', fig.axes[-1].get_ylabel())
        self.assertTrue((self.output / "NBI_Integrated_2D_Heatmap.png").is_file())

    def test_legacy_all_modes_plot_does_not_relabel_one_distribution(self):
        with self.assertRaisesRegex(NotImplementedError, 'run_selected_modes'):
            load_solver()['plot_all_modes'](self.fp, str(self.output))

    def test_outdis_uses_all_ions_and_the_actual_electron_index(self):
        for nspele in (2, 4):
            with self.subTest(nspele=nspele):
                fp = self.distribution_fixture()
                fp.nspele = nspele
                fp.ctx.tempi = [1.0] * nspele
                for name in ("hrate", "htperp", "htpar", "htemp", "ttailx", "ttailz", "ttblkx", "ttblkz"):
                    setattr(fp, name, 1.0)
                for name in ("thrate", "thtprp", "thtpar", "thttot", "tavgtx", "tavgtz", "tavbkx", "tavbkz"):
                    setattr(fp, name, np.zeros((1, 1)))
                fp.ipsplf = np.zeros((1, 2), dtype=int)
                fp.psinxt, fp.dpsxnt = 0.0, 0.1
                fp.zdotn = fp.pwsrce = fp.pwnbi = fp.pwnbi_l = fp.zbalpw = 0.0
                fp.emax, fp.joutql = 10.0, 0
                fp.pwcoll = np.array([0.0, -10.0])
                fp.pwexch = np.array([np.zeros(nspele), np.arange(1, nspele + 1)])
                case_path = self.output / f"case-{nspele}.txt"
                fp.output_paths['case'] = str(case_path)
                with contextlib.redirect_stdout(io.StringIO()):
                    fp.outdis(1)
                columns = np.loadtxt(case_path)
                self.assertEqual(columns[1], sum(range(1, nspele)))
                self.assertEqual(columns[2], nspele)
                log = (self.output / 'read.txt').read_text()
                self.assertIn('二阶矩等效温度', log)


if __name__ == "__main__":
    unittest.main()
