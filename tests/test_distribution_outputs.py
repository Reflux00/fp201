import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import matplotlib.pyplot as plt
import numpy as np

from solver_loader import load_solver


FP = load_solver()["FP"]


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
        fp.fb = np.tile([1.0, 2.0, 0.25, 0.5], fp.nptu)
        original_solution = fp.fb.copy()
        fp.tbfu = np.zeros((3, 2))
        fp.tbdfu = np.zeros((3, 2))
        with contextlib.redirect_stdout(io.StringIO()):
            fp.disfun(1)
        self.assertAlmostEqual(fp.fnorm, 4 * np.pi / 3)
        np.testing.assert_array_equal(fp.raw_tbfu, np.tile([1.0, 0.25], (3, 1)))
        np.testing.assert_array_equal(fp.raw_tbdfu, np.tile([2.0, 0.5], (3, 1)))
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


if __name__ == "__main__":
    unittest.main()
