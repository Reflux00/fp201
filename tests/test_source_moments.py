import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class SourceMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Legacy initialization writes a relative diagnostic file. A subprocess
        # keeps those writes and stdout handling out of the test runner's state.
        probe = """
import contextlib
import io
import json
from pathlib import Path
import sys
import numpy as np

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / 'tests'))
from solver_loader import load_solver

namespace = load_solver()
config = json.loads((repo / 'int_1.json').read_text())
config['nptu'] = 81
config['legmod'] = 4
config['fnbi'] = 1
config['ficrh'] = 0
config['Ns'] = 1
config['tempe'] = 4.0
config['tempi'] = [4.0, 4.0]
config['energy_range']['r_E_2'] = 60.0
config['ipheat'] = 2
config['nbi'].update({
    'pwdens': 1.0, 'energies_keV': [90.0], 'energy_spread_ratio': 0.1,
    'use_external_file': False,
})
Path('config.json').write_text(json.dumps(config))
with contextlib.redirect_stdout(io.StringIO()):
    ctx, solver, _, _, parameters = namespace['read_config_file']('config.json')
    namespace['ctx'] = ctx
    namespace['ipsii'] = []
    namespace['apply_config_settings'](solver, parameters)
    solver.nrdpsi = 1
    solver.nphi = ctx.nphi
    solver.ipsii = [1]
    solver.ipsiii = [1]
    solver.info = 1
    solver.arrays = 1
    solver.positions = np.array([1.0])
    solver.joutql = 1
    solver.current_out_dir = str(Path.cwd())
    solver.output_paths = {
        key: str(Path.cwd() / (key + '.txt'))
        for key in ['read', 'case', 'qleg00', 'A_full', 'A_banded', 'b_vector',
                    'A_out', 'b_out', 'Pn', 'Fv', 'Fi', 'name']
    }
    solver.inifpq()
    solver.coelgd_with_weight_optimized(enable_weight=True, uc=0.4, a=20.0)
    solver.iniqln(solver.ispec, 0)
    solver.inimat(0)
    solver.inicol(solver.ispec, 1)
    solver.iniqln(solver.ispec, 1)
    solver.inimat(1)
    solver.dnbi = 1.7
    solver.matrix()
    first = [solver.nbi_particle_source_moment, solver.nbi_source_power]
    solver.matrix()
    repeated = [solver.nbi_particle_source_moment, solver.nbi_source_power]
    solver.nbi_enabled = 0
    solver.matrix()
    disabled = [solver.nbi_particle_source_moment, solver.nbi_source_power]
    solver.nbi_enabled = 1
    solver.sig *= 2
    solver.matrix()
    doubled = [solver.nbi_particle_source_moment, solver.nbi_source_power]

print(json.dumps({
    'first': first, 'repeated': repeated, 'disabled': disabled, 'doubled': doubled,
    'zemxw': solver.zemxw, 'du': solver.du,
}))
"""
        with tempfile.TemporaryDirectory(prefix="fp201-source-moment-test-") as directory:
            result = subprocess.run(
                [sys.executable, "-c", probe, str(REPO)], cwd=directory,
                check=True, capture_output=True, text=True,
            )
            cls.result = json.loads(result.stdout)

    def test_assembled_source_matches_gaussian_moments(self):
        # The unit-integral Gaussian has <u²> = mean² + sigma². Its tails
        # lie more than twelve standard deviations from the finite boundaries.
        speed_squared = 90.0 / 4.0
        variance = speed_squared * 0.1**2 / 4
        self.assertGreater(variance**0.5, 1.5 * self.result["du"])
        expected_particle = 2 * (1.0 / 90.0) / speed_squared * 1.7
        expected_power = expected_particle * self.result["zemxw"] * (speed_squared + variance)
        self.assertAlmostEqual(self.result["first"][0], expected_particle, delta=expected_particle * 1e-8)
        self.assertAlmostEqual(self.result["first"][1], expected_power, delta=expected_power * 1e-8)

    def test_reassembly_resets_moments_and_disabling_nbi_clears_them(self):
        self.assertEqual(self.result["repeated"], self.result["first"])
        self.assertEqual(self.result["disabled"], [0.0, 0.0])
        for actual, initial in zip(self.result["doubled"], self.result["first"]):
            self.assertAlmostEqual(actual, 2 * initial, delta=abs(initial) * 1e-12)


if __name__ == "__main__":
    unittest.main()
