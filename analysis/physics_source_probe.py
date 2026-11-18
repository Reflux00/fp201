"""Read-only probes of source units and collision coefficients in the real solver."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.special import erf


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from solver_loader import load_solver


def initialize(config, directory):
    namespace = load_solver()
    path = directory / "config.json"
    path.write_text(json.dumps(config))
    ctx, solver, _, _, parameters = namespace["read_config_file"](str(path))
    namespace["ctx"] = ctx
    namespace["ipsii"] = []
    namespace["apply_config_settings"](solver, parameters)
    solver.nrdpsi = 1
    solver.nphi = ctx.nphi
    solver.ipsii = [1]
    solver.ipsiii = [1]
    solver.info = 1
    solver.arrays = 1
    solver.positions = np.array([1.0])
    solver.joutql = 1
    solver.current_out_dir = str(directory)
    solver.output_paths = {
        key: str(directory / (key + ".txt"))
        for key in ["read", "case", "qleg00", "A_full", "A_banded", "b_vector",
                    "A_out", "b_out", "Pn", "Fv", "Fi", "name"]
    }
    solver.inifpq()
    solver.iniqln(solver.ispec, 0)
    solver.inimat(0)
    solver.inicol(solver.ispec, 1)
    solver.inimat(1)
    return solver


def source_moments(solver):
    particle = power = 0.0
    for element in range(solver.nfelem):
        for point, weight in zip(solver.zgaus, solver.wgaus):
            u = solver.tbu[element] + solver.du * point
            solver.fokker(u)
            solver.source(u)
            term = 4 * np.pi * solver.du * weight * solver.source_nbi[0] * solver.dnbi
            particle += term
            power += term * solver.zemxw * u**2
    return {"particle_moment": particle, "power_Wcm3": power}


def run():
    base = json.loads((REPO / "int_1.json").read_text())
    base["legmod"] = 4
    base["ficrh"] = 0
    base["nbi"]["use_external_file"] = False
    base["nbi"]["power_profile_file"] = None
    base["nbi"]["pwdens"] = 1.0
    result = {"source_scaling": []}
    old_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="fp201-physics-source-") as temp:
        directory = Path(temp)
        try:
            os.chdir(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                for density, energy in [(2.75, 90), (5.5, 90), (11, 90), (5.5, 45)]:
                    config = copy.deepcopy(base)
                    config["ne"] = density
                    config["nbi"]["energies_keV"] = [energy]
                    solver = initialize(config, directory)
                    result["source_scaling"].append({
                        "ne_1e19_m3": density, "beam_keV": energy,
                        "declared_power_Wcm3": solver.pwnbi,
                        "anuref_per_s": solver.anuref,
                        "zemxw_Wcm3": solver.zemxw,
                        **source_moments(solver),
                    })

                solver = initialize(copy.deepcopy(base), directory)
                samples = np.r_[np.geomspace(1e-3, .49, 100), np.linspace(.5, 10, 1000)]
                actual = np.array([solver.fperfu(float(u)) for u in samples])
                exact = erf(samples) - 2 / np.sqrt(np.pi) * samples * np.exp(-samples**2)
                result["collision_function"] = {
                    "max_absolute_error": float(np.max(np.abs(actual - exact))),
                    "max_relative_error": float(np.max(np.abs((actual - exact) / exact))),
                }
                flux = []
                for u in np.linspace(.01, 8, 400):
                    solver.fokker(u)
                    f = np.exp(-u*u) / np.pi**1.5
                    flux.append(u * solver.psicol * (-2*u*f) / 2 + u*u*solver.psitau*f)
                result["equal_temperature_maxwell_flux_max"] = float(np.max(np.abs(flux)))

                solver.ztempi[-1, 0] = 2.0
                solver.inicol(solver.ispec, 1)
                stale = float(solver.anucol[-1])
                collision_power = -solver.zemxw * solver.exchcl(-1)
                result["unequal_temperature"] = {
                    "local_Te_keV": solver.ctx.tempi[-1], "ctx_tempe_keV": solver.ctx.tempe,
                    "electron_collision_coefficient_stale": stale,
                    "maxwell_collision_power_Wcm3": collision_power,
                    "background_source_power_Wcm3": solver.pwsrce,
                    "sum_Wcm3": collision_power + solver.pwsrce,
                }
                # This changes only the diagnostic fixture, never solver source.
                solver.ctx.tempe = 2.0
                solver.inicol(solver.ispec, 1)
                correct = float(solver.anucol[-1])
                result["unequal_temperature"].update({
                    "electron_collision_coefficient_local": correct,
                    "relative_coefficient_error": stale / correct - 1,
                })
        finally:
            os.chdir(old_cwd)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
