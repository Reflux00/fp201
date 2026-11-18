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
        probe = r"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import numpy as np

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / 'tests'))
from solver_loader import load_solver
namespace = load_solver()
base = json.loads((repo / 'int_1.json').read_text())
base.update(nptu=41, legmod=4, fnbi=1, ficrh=0, Ns=1, tempe=4.0, tempi=[4.0, 4.0])
base['energy_range']['r_E_2'] = 60.0
base['nbi'].update(pwdens=1.0, energies_keV=[90.0], energy_spread_ratio=.01,
                   use_external_file=False, power_profile_file=None)

def initialize(config, radial_position=1.0):
    Path('config.json').write_text(json.dumps(config))
    ctx, solver, _, _, parameters = namespace['read_config_file']('config.json')
    namespace['ctx'] = ctx
    namespace['ipsii'] = []
    namespace['apply_config_settings'](solver, parameters)
    solver.nrdpsi = 1
    solver.nphi = ctx.nphi
    solver.ipsii = [radial_position]
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
    solver.iniqln(solver.ispec, 0)
    solver.inimat(0)
    solver.inicol(solver.ispec, 1)
    solver.inimat(1)
    return solver

def assembled(solver):
    # Recover the physical source row before Dirichlet replacement. This lets
    # the constant/energy test functions measure the real assembled RHS.
    original_bound = solver.boundf
    origin = []
    def capture_origin():
        origin.append(float(solver.uvp[0, 0]))
        original_bound()
    solver.boundf = capture_origin
    try:
        solver.matrix()
    finally:
        solver.boundf = original_bound
    rhs = solver.fb.reshape(solver.nptu, solver.nfvrb, 2)[:, 0, :].copy()
    rhs[0, 0] = origin[0]
    return {
        'injected_power': solver.nbi_injected_power,
        'sink_power': solver.nbi_sink_power,
        'net_power': solver.nbi_source_power,
        'particles_in': solver.nbi_particle_source_moment,
        'particles_out': solver.nbi_particle_sink_moment,
        'particles_net': solver.nbi_net_particle_source_moment,
        'physical_rate': solver.nbi_particle_injection_rate,
        'rhs_particle_moment': float(4 * np.pi * rhs[:, 0].sum()),
        'rhs_power': float(4 * np.pi * solver.zemxw * (
            np.dot(solver.tbu**2, rhs[:, 0]) + np.dot(2*solver.tbu, rhs[:, 1]))),
    }

rows = []
with contextlib.redirect_stdout(io.StringIO()):
    scenarios = [
        ('default_width_coarse', 41, 5.5, [90.0], .01, 1.0),
        ('default_width_fine', 201, 5.5, [90.0], .01, 1.0),
        ('unresolved_narrow', 41, 2.75, [90.0], 1e-6, 1.0),
        ('delta', 41, 11.0, [90.0], 0.0, 1.0),
        ('three_beams', 41, 5.5, [90.0, 45.0, 30.0], .01, 1.0),
        ('zero_power', 41, 5.5, [90.0], .01, 0.0),
        ('endpoint_delta', 41, 5.5, [240.0], 0.0, 1.0),
    ]
    for label, points, density, energies, spread, power in scenarios:
        config = copy.deepcopy(base)
        config.update(nptu=points, ne=density, Ns=len(energies))
        config['nbi'].update(energies_keV=energies, energy_spread_ratio=spread, pwdens=power)
        solver = initialize(config)
        row = assembled(solver)
        row.update(label=label, energies=energies, power=power, temperature=4.0,
                   ion_density_cm3=density*.97*1e13, nu_ref=solver.anuref)
        rows.append(row)

    solver = initialize(copy.deepcopy(base))
    solver.dnbi = 1.7
    first = assembled(solver)
    repeated = assembled(solver)
    solver.sig *= 2
    doubled = assembled(solver)
    solver.nbi_enabled = 0
    disabled = assembled(solver)

    config = copy.deepcopy(base)
    config['nbi']['energies_keV'] = [239.0]
    try:
        initialize(config)
        truncated_rejected = False
    except ValueError as error:
        truncated_rejected = '8 sigma' in str(error)

    solver = initialize(copy.deepcopy(base))
    solver.ztempi[-1, 0] = 2.0
    solver.inicol(solver.ispec, 1)
    local = solver.anucol[-1]
    config = copy.deepcopy(base)
    config['tempe'] = 2.0
    reference = initialize(config)
    local_temperature = {
        'stored': solver.ctx.tempe,
        'local_collision': local,
        'reference_collision': reference.anucol[-1],
    }

    # A true solve tests the energy-moment identity, not just agreement among
    # independently reported source fields.
    balances = []
    for electron_temperature in (4.0, 2.0):
        config = copy.deepcopy(base)
        config['tempe'] = electron_temperature
        solver = initialize(config)
        solver.solve_stationary(1)
        collisional_power = -solver.zemxw * solver.exchcl(-1)
        species_sum = sum(solver.exchcl(species) for species in range(solver.nspele))
        balances.append({
            'Te': electron_temperature,
            'energy_residual': collisional_power + solver.pwsrce + solver.nbi_source_power,
            'species_sum_error': species_sum - solver.exchcl(-1),
        })

    Path('power_profile.txt').write_text('rho power_Wcm3\n0.0 1.0\n0.5 2.0\n1.0 3.0\n')
    mu = np.linspace(-1., 1., 101)
    angular0 = np.exp(-.5 * ((mu + .5) / .07)**2)
    angular1 = np.exp(-.5 * ((mu - .5) / .07)**2)
    angular_data = np.column_stack((mu, angular0, angular1))
    np.savetxt('angular_source.txt', angular_data,
               header='Mu rho_0 rho_1', comments='')
    config = copy.deepcopy(base)
    config['nbi']['power_profile_file'] = 'power_profile.txt'
    config['nbi']['source_file'] = 'angular_source.txt'
    config['nbi']['use_external_file'] = True
    solver = initialize(config, radial_position=.25)
    profile_power = solver.nbi_injected_power
    angular_first_moment = solver.sig[0, 1] / solver.sig[0, 0]
    invalid_angular_sources = []
    for header, data in [
        ('Mu rho_1 rho_0', angular_data),
        ('Mu rho_0.5 rho_1', angular_data),
        ('Mu rho_0 rho_1', angular_data[::-1]),
        ('Mu unknown unknown', angular_data),
    ]:
        np.savetxt('angular_source.txt', data, header=header, comments='')
        try:
            initialize(config, radial_position=.25)
            invalid_angular_sources.append(False)
        except ValueError:
            invalid_angular_sources.append(True)

print(json.dumps(dict(rows=rows, first=first, repeated=repeated, doubled=doubled,
                     disabled=disabled, truncated_rejected=truncated_rejected,
                     local_temperature=local_temperature, balances=balances,
                     profile_power=profile_power, angular_first_moment=angular_first_moment,
                     invalid_angular_sources=invalid_angular_sources)))
"""
        with tempfile.TemporaryDirectory(prefix="fp201-source-moment-test-") as directory:
            result = subprocess.run(
                [sys.executable, "-c", probe, str(REPO)], cwd=directory,
                check=True, capture_output=True, text=True,
            )
            cls.result = json.loads(result.stdout)

    def test_physical_particle_and_energy_input_and_balanced_thermal_sink(self):
        for row in self.result["rows"]:
            with self.subTest(row["label"]):
                # The fixture gives equal power to each beam. These expected
                # values come from P/E and 3T/2 per removed thermal particle.
                per_beam_power = row["power"] / len(row["energies"])
                physical_rate = sum(per_beam_power / (energy * 1.602176634e-16)
                                    for energy in row["energies"])
                normalized_rate = physical_rate / (row["ion_density_cm3"] * row["nu_ref"])
                sink_power = physical_rate * 1.5 * row["temperature"] * 1.602176634e-16
                self.assertAlmostEqual(row["injected_power"], row["power"], delta=2e-12)
                self.assertAlmostEqual(row["sink_power"], sink_power, delta=2e-12)
                self.assertAlmostEqual(row["net_power"], row["power"] - sink_power, delta=2e-12)
                self.assertAlmostEqual(row["particles_in"], normalized_rate, delta=2e-14)
                self.assertAlmostEqual(row["particles_out"], -normalized_rate, delta=2e-14)
                self.assertAlmostEqual(row["physical_rate"], physical_rate,
                                       delta=max(1., physical_rate) * 2e-12)
                self.assertAlmostEqual(row["particles_net"], 0., delta=2e-17)
                self.assertAlmostEqual(row["rhs_particle_moment"], 0., delta=2e-17)
                self.assertAlmostEqual(row["rhs_power"], row["power"] - sink_power, delta=2e-12)

    def test_reassembly_multiplier_and_disabling_nbi(self):
        self.assertEqual(self.result["repeated"], self.result["first"])
        for key in ("injected_power", "sink_power", "particles_in", "physical_rate"):
            self.assertAlmostEqual(self.result["doubled"][key], 2 * self.result["first"][key],
                                   delta=max(1., abs(self.result["first"][key])) * 2e-12)
        self.assertAlmostEqual(self.result["first"]["injected_power"], 1.7, delta=2e-12)
        self.assertTrue(all(value == 0 for value in self.result["disabled"].values()))

    def test_energy_domain_truncation_is_rejected(self):
        self.assertTrue(self.result["truncated_rejected"])

    def test_local_electron_temperature_updates_collision_coefficient(self):
        result = self.result["local_temperature"]
        self.assertEqual(result["stored"], 2.0)
        self.assertAlmostEqual(result["local_collision"], result["reference_collision"], delta=1e-12)

    def test_collision_energy_moment_closes_the_actual_steady_equation(self):
        for row in self.result["balances"]:
            with self.subTest(Te=row["Te"]):
                self.assertAlmostEqual(row["energy_residual"], 0.0, delta=2e-11)
                self.assertAlmostEqual(row["species_sum_error"], 0.0, delta=2e-14)

    def test_radial_power_profile_uses_coordinates(self):
        self.assertAlmostEqual(self.result["profile_power"], 1.5, delta=2e-12)

    def test_angular_source_uses_the_same_radial_coordinate(self):
        # Equal-integral beams at mu=-.5 (rho=0) and +.5 (rho=1)
        # have <mu>=-.25 at rho=.25, hence Legendre S1/S0=3<mu>.
        self.assertAlmostEqual(self.result["angular_first_moment"], -.75, delta=2e-10)
        self.assertTrue(all(self.result["invalid_angular_sources"]))


if __name__ == "__main__":
    unittest.main()
