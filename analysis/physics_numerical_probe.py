"""Audit real FE solutions and the original origin row before Dirichlet replacement."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.linalg import solve_banded

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from solver_loader import load_solver


def run_case(config, label, output):
    namespace = load_solver()
    case_dir = output / label
    case_dir.mkdir(parents=True, exist_ok=True)
    config_path = case_dir / "config.json"
    config_path.write_text(json.dumps(config))
    previous = Path.cwd()
    started = time.monotonic()
    try:
        os.chdir(case_dir)
        with (case_dir / "probe.log").open("w") as log, contextlib.redirect_stdout(log):
            ctx, fp, _, _, para = namespace["read_config_file"](str(config_path))
            namespace["ctx"] = ctx
            namespace["ipsii"] = []
            namespace["apply_config_settings"](fp, para)
            fp.nrdpsi = 1
            fp.nphi = ctx.nphi
            fp.ipsii = [1]
            fp.ipsiii = [1]
            fp.info = 1
            fp.arrays = 1
            fp.positions = np.array([1.0])
            fp.joutql = 0
            fp.current_out_dir = str(case_dir)
            fp.output_paths = {key: str(case_dir / (key + ".txt")) for key in
                               ["read", "case", "qleg00", "A_full", "A_banded", "b_vector",
                                "A_out", "b_out", "Pn", "Fv", "Fi", "name"]}
            fp.inifpq()
            fp.coelgd_with_weight_optimized(enable_weight=bool(fp.enable_trapping),
                                            uc=fp.trapping_ucr, a=fp.trapping_a)
            fp.iniqln(fp.ispec, 0)
            fp.inimat(0)
            fp.inicol(fp.ispec, 1)
            fp.iniqln(fp.ispec, 1)
            fp.inimat(1)
            fp.solve_stationary(1)
            fp.dfparm(1, fp.power_iterations)
            minimum_index = np.unravel_index(np.argmin(fp.tfecmu), fp.tfecmu.shape)
            _, _, char_mu, char_energy = fp.extract_high_energy_tail_contour(1e-7)
            result = {
                "label": label, "nptu": fp.nptu, "legmod": fp.legmod,
                "configuration": config,
                "emax_keV": float(fp.tbkev[-1]), "du": fp.du,
                "NBI_input_Wcm3": fp.pwdens_nbi if fp.nbi_enabled else 0.0,
                "ICRH_input_Wcm3": fp.pwdens,
                "hrate_Wcm3": fp.hrate, "fnorm": fp.fnorm, "zemxw": fp.zemxw,
                "nbi_source_power_raw_Wcm3": fp.nbi_source_power,
                "nbi_particle_source_raw": fp.nbi_particle_source_moment,
                "original_origin_row_residual_4pi": fp.particle_balance_residual,
                "nbi_injected_power_Wcm3": fp.nbi_injected_power,
                "nbi_sink_power_Wcm3": fp.nbi_sink_power,
                "nbi_net_particle_source": fp.nbi_net_particle_source_moment,
                "density_residual": fp.density_residual,
                "power_relative_residual": fp.power_relative_residual,
                "matrix_relative_residual": fp.matrix_relative_residual,
                "bulk_power_residual_Wcm3": fp.bulk_power_residual,
                "collisional_power_Wcm3": fp.pwcoll[1],
                "min_distribution": float(fp.tfecmu.min()),
                "minimum_energy_keV": float(fp.tbkev[minimum_index[0]]),
                "minimum_mu": float(fp.xmu[minimum_index[1]]),
                "negative_grid_count": int(np.count_nonzero(fp.tfecmu < 0)),
                "raw_f0_at_first_nodes": fp.raw_tbfu[:5, 0].tolist(),
                "raw_f0_at_u1": float(np.interp(1.0, fp.tbu, fp.raw_tbfu[:, 0])),
                "normalized_f0_at_u1": float(np.interp(1.0, fp.tbu, fp.tbfu[:, 0])),
                "htemp_keV": fp.htemp, "htpar_keV": fp.htpar, "htperp_keV": fp.htperp,
                "last_energy_f_max": float(fp.tfecmu[-1].max()),
                "last_energy_above_1e7": int(np.count_nonzero(fp.tfecmu[-1] >= 1e-7)),
                "tail_threshold": 1e-7, "char_mu": float(char_mu), "char_E_keV": float(char_energy),
                "seconds": time.monotonic() - started,
            }
            if not fp.nbi_enabled and not fp.icrh_enabled:
                reference = np.exp(-fp.tbu**2) / np.pi**1.5
                result["maxwell_max_absolute_error"] = float(np.max(np.abs(fp.tbfu[:, 0] - reference)))
            np.savez_compressed(case_dir / "solution.npz", raw=fp.raw_tbfu,
                                normalized=fp.tbfu, distribution=fp.tfecmu,
                                energy=fp.tbkev, mu=fp.xmu, derivatives=fp.tbdfu)
        (case_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["nbi-grid", "modes", "angular", "domain", "maxwell", "extra"], default="nbi-grid")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (args.output_dir or Path(tempfile.mkdtemp(prefix="fp201-physics-"))).resolve()
    base = json.loads((REPO / "int_1.json").read_text())
    cases = []
    if args.suite == "nbi-grid":
        for nptu in [101, 201, 401, 801]:
            config = json.loads(json.dumps(base))
            config.update(nptu=nptu, legmod=5, fnbi=1, ficrh=0)
            config["nbi"]["energy_spread_ratio"] = 0.1
            cases.append((f"nbi-n{nptu}", config))
    elif args.suite == "modes":
        for name, nbi, icrh in [("nbi", 1, 0), ("icrh", 0, 1), ("synergy", 1, 1)]:
            config = json.loads(json.dumps(base))
            config.update(fnbi=nbi, ficrh=icrh)
            cases.append((name, config))
    elif args.suite == "maxwell":
        for nptu in [101, 201, 401]:
            config = json.loads(json.dumps(base))
            config.update(nptu=nptu, legmod=5, fnbi=0, ficrh=0)
            cases.append((f"maxwell-n{nptu}", config))
    elif args.suite == "angular":
        for legmod in [11, 21, 31, 41]:
            config = json.loads(json.dumps(base))
            config.update(legmod=legmod, nptu=101, fnbi=0, ficrh=1)
            cases.append((f"icrh-l{legmod}", config))
    elif args.suite == "domain":
        for emax, nptu in [(240, 201), (540, 301), (960, 401)]:
            config = json.loads(json.dumps(base))
            config.update(nptu=nptu, fnbi=0, ficrh=1)
            config["energy_range"]["r_E_2"] = emax / 4
            cases.append((f"icrh-e{emax}", config))
    elif args.suite == "extra":
        for name, nbi, icrh, legmod in [("maxwell-default", 0, 0, 21),
                                      ("icrh-l41-default", 0, 1, 41),
                                      ("synergy-l41-default", 1, 1, 41)]:
            config = json.loads(json.dumps(base))
            config.update(fnbi=nbi, ficrh=icrh, legmod=legmod)
            cases.append((name, config))
    rows = []
    for label, config in cases:
        result = run_case(config, label, output)
        rows.append(result)
        print(json.dumps(result), flush=True)
        (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
