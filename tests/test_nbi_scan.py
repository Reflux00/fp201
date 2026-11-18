"""Regression checks for the standalone NBI scan against the real solver."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from solver_loader import load_solver


class NBIScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fp201-scan-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.repo = Path(cls.temporary.name)
        (cls.repo / "analysis").mkdir()
        source_repo = Path(__file__).resolve().parents[1]
        shutil.copy2(source_repo / "FPQL_201.py", cls.repo / "FPQL_201.py")
        cls.script = cls.repo / "analysis" / "nbi_power_scan_probe.py"
        shutil.copy2(source_repo / "analysis" / "nbi_power_scan_probe.py", cls.script)
        config = json.loads((source_repo / "int_1.json").read_text())
        config["nptu"] = 81
        config["legmod"] = 5
        config["tempi"] = [4.0, 6.0]
        config["nbi"]["energy_base_keV"] = 60.0
        cls.results = []
        for power in (1.0, 2.0):
            config["nbi"]["pwdens"] = power
            config_path = cls.repo / f"config-{power}.json"
            config_path.write_text(json.dumps(config))
            output = cls.repo / f"scan-{power}"
            completed = subprocess.run(
                [sys.executable, str(cls.script), "--config", str(config_path),
                 "--output-dir", str(output)],
                cwd=cls.repo, capture_output=True, text=True, check=True,
            )
            cls.results.append(json.loads(completed.stdout))

    def test_moved_checkout_uses_local_grid_and_beam(self):
        result = self.results[0]
        self.assertEqual(result["nptu"], 81)
        self.assertEqual(result["legmod"], 5)
        np.testing.assert_allclose(result["beam_energies_keV"], [60.0])
        self.assertGreater(result["background_source_max"], 0)
        for row in result["rows"]:
            self.assertLessEqual(abs(row["peak_energy_keV"] - 60), 3)

    def test_scan_powers_do_not_depend_on_configured_power(self):
        self.assertEqual(self.results[0]["rows"], self.results[1]["rows"])
        self.assertEqual(self.results[1]["configured_power_Wcm3"], 2.0)
        self.assertEqual(self.results[1]["reference_power_Wcm3"], 1.0)

    def test_nonzero_background_matches_direct_two_watt_solve(self):
        namespace = load_solver(str(self.repo / "FPQL_201.py"))
        output = self.repo / "direct"
        output.mkdir()
        old_cwd = Path.cwd()
        try:
            os.chdir(output)
            with contextlib.redirect_stdout(io.StringIO()):
                ctx, solver, _, _, para = namespace["read_config_file"](
                    str(self.repo / "config-2.0.json")
                )
                namespace["ctx"] = ctx
                namespace["ipsii"] = []
                namespace["apply_config_settings"](solver, para)
                solver.nbi_enabled = 1
                solver.icrh_enabled = 0
                solver.ipsii = [1]
                solver.ipsiii = [1]
                solver.info = 1
                solver.arrays = 1
                solver.positions = np.array([1.])
                solver.current_out_dir = str(output)
                solver.output_paths = {
                    key: str(output / f"{key}.txt")
                    for key in ["read", "case", "qleg00", "A_full", "A_banded", "b_vector",
                                "A_out", "b_out", "Pn", "Fv", "Fi", "name"]
                }
                # Production execution independently initializes and assembles
                # the 2 W/cm³ problem, including its nonzero background source.
                solver.t_FP(solver.power, 1, solver.legmod, solver.nptu, joutql_in=0)
            self.assertLess(solver.matrix_relative_residual, 1e-10)
            with np.load(self.repo / "scan-1.0" / "P-2.0.npz") as scanned:
                np.testing.assert_allclose(scanned["raw"], solver.fb, rtol=2e-12, atol=1e-13)
                np.testing.assert_allclose(
                    scanned["normalized_distribution"], solver.tbfu @ solver.plgcmu,
                    rtol=2e-12, atol=1e-13,
                )
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
