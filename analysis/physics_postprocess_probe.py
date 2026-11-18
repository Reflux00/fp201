"""Audit actual radial postprocessing and zero-power initialization, without editing the solver."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from solver_loader import load_solver


def run_probe():
    namespace = load_solver()
    FP = namespace["FP"]
    result = {}
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="fp201-postprocess-audit-") as directory:
        try:
            os.chdir(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                # Exercise the production radial integration on an exact normalized
                # Maxwellian. Its particle density is known independently.
                fp = FP.__new__(FP)
                fp.nptu, fp.nmu, fp.nrdpsi = 201, 3, 1
                fp.ispres, fp.nspele, fp.ipheat = 1, 3, 2
                fp.r_E_1, fp.r_E_2 = 200.0, 60.0
                fp.ztempi = np.full((3, 1), 4.0)
                fp.zdeni = np.array([[5.335], [0.165], [5.5]])
                fp.redpsi = np.array([1.0])
                velocity = np.linspace(0, np.sqrt(60), fp.nptu)
                maxwellian = np.exp(-velocity**2) / np.pi**1.5
                fp.f_v_mu_psi = np.broadcast_to(
                    maxwellian[:, None, None], (fp.nptu, fp.nmu, 1)
                ).copy()
                first, energy = fp.integrate_over_radius()
                second, _ = fp.integrate_over_radius()
                index = int(np.argmin(abs(energy - 4.0)))
                result["radial_postprocessing"] = {
                    "selected_species": "second ion (3 percent concentration)",
                    "selected_ion_density_m3": float(fp.zdeni[1, 0] * 1e19),
                    "electron_density_used_m3": float(fp.zdeni[-1, 0] * 1e19),
                    "density_factor_error": float(fp.zdeni[-1, 0] / fp.zdeni[1, 0]),
                    "sample_energy_keV": float(energy[index]),
                    "first_call_value": float(first[index, 1]),
                    "second_call_value": float(second[index, 1]),
                    "second_over_first": float(second[index, 1] / first[index, 1]),
                    "energy_zero_value": float(first[0, 1]),
                    "correct_dndE_dmu_at_zero": 0.0,
                }

                parameters = json.loads((REPO / "int_1.json").read_text())
                parameters.update(nptu=41, legmod=4, pwdens=0.0, fnbi=0, ficrh=1)
                config = Path(directory) / "config.json"
                config.write_text(json.dumps(parameters))
                ctx, solver, _, _, parameters = namespace["read_config_file"](str(config))
                namespace["ctx"], namespace["ipsii"] = ctx, []
                namespace["apply_config_settings"](solver, parameters)
                solver.nrdpsi, solver.nphi = 1, ctx.nphi
                solver.ipsii, solver.ipsiii = [1], [1]
                solver.info, solver.arrays = 1, 1
                solver.positions = np.array([1.0])
                solver.joutql = 0
                solver.current_out_dir = directory
                solver.output_paths = {
                    key: str(Path(directory) / (key + ".txt"))
                    for key in ["read", "case", "qleg00", "A_full", "A_banded", "b_vector",
                                "A_out", "b_out", "Pn", "Fv", "Fi", "name"]
                }
                solver.inifpq()
                result["zero_icrh_input"] = {
                    "input_Wcm3": 0.0,
                    "power_after_initialization": float(solver.power),
                    "harmonic_power_profile": solver.pwredh.tolist(),
                }
        finally:
            os.chdir(original_cwd)
    return result


if __name__ == "__main__":
    result = run_probe()
    print(json.dumps(result, indent=2))
