"""Small real-solver cases for particle, power and frequency contracts."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from solver_loader import load_solver


REPO = Path(__file__).resolve().parents[1]
CASE_RUNNER = """
import json
from pathlib import Path
import sys

repo, config_path, label, output = sys.argv[1:]
sys.path.insert(0, str(Path(repo) / 'analysis'))
from physics_numerical_probe import run_case

config = json.loads(Path(config_path).read_text())
try:
    result = run_case(config, label, Path(output))
except ValueError as error:
    result = {'error_type': type(error).__name__, 'error': str(error)}
print(json.dumps(result))
"""


class PhysicalBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fp201-physical-balance-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name)
        base = json.loads((REPO / "int_1.json").read_text())
        base.update(nptu=81, legmod=5, ipheat=2, pwdens=0.2,
                    frequency_convention="rf_hz", omega_ci=36.5e6,
                    rf_power_mode="absorbed", plot=False, ne=5.5, nspec=2,
                    ispec=1, azi=[1.0, 1.0], atm=[2.0, 1.0],
                    aconc=[0.97, 0.03], kperp=10.0, Ns=1)
        base["tempe"] = 4.0
        base["tempi"] = [4.0, 4.0]
        base["energy_range"] = {"r_E_1": 200.0, "r_E_2": 60.0}
        base["trapping"] = {"enabled": 1, "model": "symmetric", "ucr": 0.4, "a": 20.0}
        base["input_mode"] = {"case_mode": 0, "ipsii": [1.0]}
        base["nbi"].update(pwdens=1.0, use_external_file=False,
                           energy_base_keV=90.0, energies_keV=[90.0],
                           power_profile_file=None, energy_spread_ratio=0.01,
                           gaussian={"mu_center": 0.7, "sigma": 0.07})
        specifications = {
            "maxwell": dict(fnbi=0, ficrh=0),
            "nbi": dict(fnbi=1, ficrh=0),
            "icrh": dict(fnbi=0, ficrh=1),
            "synergy": dict(fnbi=1, ficrh=1),
            "reference": dict(fnbi=0, ficrh=1, rf_power_mode="maxwell_reference"),
            "zero_rf": dict(fnbi=0, ficrh=1, pwdens=0.0),
            "excess_nbi": dict(fnbi=1, ficrh=0),
            "cyclotron_hz": dict(fnbi=0, ficrh=1, frequency_convention="cyclotron_hz",
                                 omega_ci=18.25e6),
            "cyclotron_rad_s": dict(fnbi=0, ficrh=1, frequency_convention="cyclotron_rad_s",
                                    omega_ci=2 * np.pi * 18.25e6),
        }
        for label, updates in specifications.items():
            config = json.loads(json.dumps(base))
            config.update(updates)
            if label == "excess_nbi":
                config["nbi"]["pwdens"] = 10.0
            (cls.output / f"{label}.json").write_text(json.dumps(config))

        def run(label):
            result = subprocess.run(
                [sys.executable, "-c", CASE_RUNNER, str(REPO),
                 str(cls.output / f"{label}.json"), label, str(cls.output)],
                cwd=cls.output, capture_output=True, text=True, check=True,
            )
            return label, json.loads(result.stdout)

        # Legacy relative diagnostics and module globals stay isolated in each
        # process. The cases do not share mutable solver instances or cwd.
        with ThreadPoolExecutor(max_workers=3) as executor:
            cls.results = dict(executor.map(run, specifications))

    def test_unforced_maxwell_retains_density_and_temperature(self):
        result = self.results["maxwell"]
        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["fnorm"], 1.0, delta=1e-11)
        self.assertAlmostEqual(result["htemp_keV"], 4.0, delta=2e-4)
        self.assertAlmostEqual(result["htpar_keV"], 4.0, delta=2e-4)
        self.assertAlmostEqual(result["htperp_keV"], 4.0, delta=2e-4)
        self.assertLess(result["maxwell_max_absolute_error"], 2e-5)

    def test_particle_and_energy_balance_close_for_all_heating_modes(self):
        for label in ("maxwell", "nbi", "icrh", "synergy"):
            result = self.results[label]
            with self.subTest(mode=label):
                self.assertNotIn("error", result)
                self.assertLess(abs(result["nbi_net_particle_source"]), 1e-12)
                self.assertLess(abs(result["original_origin_row_residual_4pi"]), 1e-9)
                self.assertLess(abs(result["bulk_power_residual_Wcm3"]), 1e-9)
                self.assertLess(result["density_residual"], 1e-11)
                self.assertLess(result["matrix_relative_residual"], 1e-10)
        self.assertAlmostEqual(self.results["nbi"]["nbi_injected_power_Wcm3"], 1.0, delta=1e-10)

    def test_final_absorbed_power_is_fixed_with_and_without_nbi(self):
        for label in ("icrh", "synergy"):
            with self.subTest(mode=label):
                result = self.results[label]
                self.assertAlmostEqual(result["hrate_Wcm3"], 0.2, delta=0.2e-7)
                self.assertLess(result["power_relative_residual"], 1e-7)

    def test_reference_power_mode_does_not_force_final_absorption(self):
        result = self.results["reference"]
        self.assertNotIn("error", result)
        self.assertGreater(abs(result["hrate_Wcm3"] - 0.2), 1e-4)
        self.assertLess(abs(result["bulk_power_residual_Wcm3"]), 1e-9)

    def test_zero_rf_stays_zero_and_high_nbi_rejects_negative_core(self):
        zero = self.results["zero_rf"]
        self.assertNotIn("error", zero)
        self.assertEqual(zero["ICRH_input_Wcm3"], 0.0)
        self.assertEqual(zero["hrate_Wcm3"], 0.0)
        excessive = self.results["excess_nbi"]
        self.assertEqual(excessive.get("error_type"), "ValueError")
        self.assertIn("各向同性分布为负", excessive.get("error", ""))

    def test_frequency_conventions_give_the_same_physical_solution(self):
        with np.load(self.output / "icrh" / "solution.npz") as baseline:
            reference = baseline["normalized"]
            for label in ("cyclotron_hz", "cyclotron_rad_s"):
                with self.subTest(convention=label):
                    self.assertNotIn("error", self.results[label])
                    with np.load(self.output / label / "solution.npz") as equivalent:
                        np.testing.assert_allclose(equivalent["normalized"], reference,
                                                   rtol=1e-10, atol=1e-12)

    def test_sampled_moment_integrates_the_last_interval_for_even_grid_size(self):
        solver_type = load_solver()["FP"]
        solver = solver_type.__new__(solver_type)
        # A constant distribution with the u² volume measure integrates to
        # U³/3; both odd and even point counts must cover the full domain.
        for count in (4, 80, 81):
            with self.subTest(points=count):
                actual = solver.qlmomu(np.ones(count), 2.0 / (count - 1), count, 0)
                self.assertAlmostEqual(actual, 8.0 / 3.0, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
