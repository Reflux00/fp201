"""Reject nonphysical JSON and external radial profiles at their input boundaries."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from solver_loader import load_solver


REPO = Path(__file__).resolve().parents[1]
PROFILE_RUNNER = """
import contextlib
import io
import json
from pathlib import Path
import sys
import numpy as np

repo, config_path = sys.argv[1:]
sys.path.insert(0, str(Path(repo) / 'tests'))
from solver_loader import load_solver
namespace = load_solver()
baseline = {
    'P': [0.2, 0.2], 'N0': [0.5335, 0.5335], 'N1': [0.0165, 0.0165],
    'Ne': [0.55, 0.55], 'T0': [4.0, 4.0], 'T1': [4.0, 4.0], 'Te': [4.0, 4.0],
}
cases = {
    'valid': {}, 'zero_power': {'P': [0.0, 0.0]},
    'negative_power': {'P': [-0.001, -0.001]},
    'zero_temperature': {'T0': [0.0, 4.0]},
    'zero_electron_density': {'Ne': [0.0, 0.55]},
    'zero_selected_density': {'N0': [0.0, 0.5335]},
    'negative_ion_density': {'N1': [-0.01, 0.0165]},
    'nan_temperature': {'T1': [float('nan'), 4.0]},
    'nonneutral': {'Ne': [0.6, 0.55]},
    'wrong_length': {'P': [0.2]},
    'rho_outside': {}, 'rho_repeated': {},
}
results = {}
with contextlib.redirect_stdout(io.StringIO()):
    for name, changes in cases.items():
        ctx, solver, _, _, config = namespace['read_config_file'](config_path)
        namespace['apply_config_settings'](solver, config)
        solver.nrdpsi = 2
        solver.ipsii = [0.2, 0.4]
        solver.positions = np.array([0.2, 0.4])
        if name == 'rho_outside':
            solver.positions = np.array([-0.1, 0.4])
        if name == 'rho_repeated':
            solver.positions = np.array([0.2, 0.2])
        solver.arrays = {**baseline, **changes}
        solver.output_paths = {'read': str(Path.cwd() / (name + '.log'))}
        try:
            solver.inifpq()
        except ValueError as error:
            results[name] = {'error_type': type(error).__name__, 'error': str(error)}
        else:
            results[name] = {'rf_power': solver.pwredh[0].tolist()}
print(json.dumps(results))
"""


class InputValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="fp201-input-validation-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.base = json.loads((REPO / "int_1.json").read_text())
        cls.base.update(nptu=21, legmod=5, nspec=2, ispec=1, ipheat=2,
                        fnbi=0, ficrh=1, pwdens=0.2, Ns=1, ne=5.5,
                        azi=[1.0, 1.0], atm=[2.0, 1.0], aconc=[0.97, 0.03],
                        tempi=[4.0, 4.0], tempe=4.0, plot=False)
        cls.base["input_mode"] = {"case_mode": 0, "ipsii": [0.2]}
        cls.base["nbi"].update(Ns=1, energies_keV=[90.0], use_external_file=False)
        cls.config_path = cls.directory / "config.json"
        cls.config_path.write_text(json.dumps(cls.base))
        # inifpq has a legacy relative diagnostic write. Isolate the whole
        # profile acceptance/rejection check without changing test-runner cwd.
        completed = subprocess.run(
            [sys.executable, "-c", PROFILE_RUNNER, str(REPO), str(cls.config_path)],
            cwd=cls.directory, capture_output=True, text=True, check=True,
        )
        cls.profiles = json.loads(completed.stdout)
        cls.namespace = load_solver()

    def read_config(self, config):
        path = self.directory / "candidate.json"
        path.write_text(json.dumps(config))
        with contextlib.redirect_stdout(io.StringIO()):
            return self.namespace["read_config_file"](str(path))

    def test_negative_rf_profile_cannot_bypass_valid_json(self):
        result = self.profiles["negative_power"]
        self.assertEqual(result.get("error_type"), "ValueError")
        self.assertIn("RF 功率剖面不能为负", result["error"])

    def test_zero_rf_profiles_are_preserved(self):
        self.assertEqual(self.profiles["valid"], {"rf_power": [0.2, 0.2]})
        self.assertEqual(self.profiles["zero_power"], {"rf_power": [0.0, 0.0]})

    def test_profile_temperature_density_shape_and_quasineutrality(self):
        errors = {
            "zero_temperature": "必须为正",
            "zero_electron_density": "必须为正",
            "zero_selected_density": "被求解离子的径向密度必须为正",
            "negative_ion_density": "离子密度不能为负",
            "nan_temperature": "全部有限",
            "nonneutral": "准中性",
            "wrong_length": "等长",
            "rho_outside": "严格递增",
            "rho_repeated": "严格递增",
        }
        for name, message in errors.items():
            with self.subTest(profile=name):
                self.assertEqual(self.profiles[name].get("error_type"), "ValueError")
                self.assertIn(message, self.profiles[name]["error"])

    def test_json_beam_count_must_match_explicit_energies(self):
        config = json.loads(json.dumps(self.base))
        config["nbi"]["Ns"] = 2
        with self.assertRaisesRegex(ValueError, "energies_keV.*Ns"):
            self.read_config(config)

    def test_json_zero_power_is_legal_and_negative_power_is_rejected(self):
        config = json.loads(json.dumps(self.base))
        config["pwdens"] = 0.0
        _, solver, _, _, _ = self.read_config(config)
        self.assertEqual(solver.power, 0.0)
        config["pwdens"] = -0.001
        with self.assertRaisesRegex(ValueError, "pwdens.*非负"):
            self.read_config(config)

    def test_json_rejects_invalid_radial_coordinates(self):
        for radii in ([-0.1], [1.1], [float("nan")], [0.4, 0.2], [0.2, 0.2]):
            config = json.loads(json.dumps(self.base))
            config["input_mode"] = {"case_mode": 1, "ipsii": radii}
            with self.subTest(radii=radii):
                with self.assertRaisesRegex(ValueError, "ipsii"):
                    self.read_config(config)


if __name__ == "__main__":
    unittest.main()
