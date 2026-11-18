"""Scan absolute NBI powers using the current solver and a fixed background."""

import argparse
import ast
import contextlib
import json
import os
from pathlib import Path
import tempfile
import time


def run_scan(config_path, output_dir=None):
    repo = Path(__file__).resolve().parents[1]
    config_path = Path(config_path).resolve()
    probe_dir = (Path(output_dir).resolve() if output_dir is not None
                 else Path(tempfile.mkdtemp(prefix="fp201-nbi-scan-")))
    probe_dir.mkdir(parents=True, exist_ok=True)
    source_path = repo / "FPQL_201.py"
    module = ast.parse(source_path.read_text())
    # The solver has top-level execution and stdout setup; load definitions only.
    module.body = [node for node in module.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)
    )]
    namespace = {"__name__": "fp201_probe"}
    exec(compile(module, str(source_path), "exec"), namespace)
    np = namespace["np"]
    solve_banded = namespace["solve_banded"]
    started = time.monotonic()
    original_cwd = Path.cwd()
    try:
        # Some legacy diagnostic outputs still use paths relative to cwd.
        os.chdir(probe_dir)
        with (probe_dir / "initialization.log").open("w") as log, contextlib.redirect_stdout(log):
            ctx, ssp, out, plot, para = namespace["read_config_file"](str(config_path))
            namespace["ctx"] = ctx
            namespace["ipsii"] = []
            namespace["apply_config_settings"](ssp, para)
            configured_power = ssp.pwdens_nbi
            for name in ("nbi_file", "nbi_power_file"):
                value = getattr(ssp, name)
                if value:
                    setattr(ssp, name, str((config_path.parent / value).resolve()))
            ssp.nbi_enabled = 1
            ssp.icrh_enabled = 0
            ssp.nrdpsi = 1
            ssp.nphi = ctx.nphi
            ssp.ipsii = [1]
            ssp.ipsiii = [1]
            ssp.info = 1
            ssp.arrays = 1
            ssp.positions = np.array([1.], dtype=float)
            ssp.joutql = 1
            ssp.current_out_dir = str(probe_dir)
            ssp.output_paths = {
                key: str(probe_dir / (key + ".txt"))
                for key in ["read", "case", "qleg00", "A_full", "A_banded", "b_vector",
                            "A_out", "b_out", "Pn", "Fv", "Fi", "name"]
            }
            ssp.inifpq()
            # The scan supplies absolute powers; the baseline is always 1 W/cm³,
            # including when the input contains a radial power profile.
            ssp.pwdens_nbi = 1.0
            ssp.nbi_power_profile = None
            ssp.coelgd_with_weight_optimized(
                enable_weight=bool(ssp.enable_trapping), uc=ssp.trapping_ucr, a=ssp.trapping_a
            )
            ssp.iniqln(ssp.ispec, 0)
            ssp.inimat(0)
            ssp.inicol(ssp.ispec, 1)
            ssp.iniqln(ssp.ispec, 1)
            ssp.inimat(1)
            ssp.matrix()
            A = ssp.fmatrx[ssp.iwd:3 * ssp.iwd + 1, :].copy()
            b1 = ssp.fb.copy()
            sig1 = ssp.sig.copy()

            # Assemble the actual background RHS: unequal species temperatures
            # can produce a fixed source in addition to the origin boundary.
            ssp.sig.fill(0.0)
            ssp.matrix()
            b0 = ssp.fb.copy()
            if not np.array_equal(A, ssp.fmatrx[ssp.iwd:3 * ssp.iwd + 1, :]):
                raise RuntimeError("The operator depends on NBI source amplitude; a fixed-matrix scan is invalid")
            ssp.sig = sig1

            op_diffs = []
            for ip in np.unique(np.linspace(0, ssp.nfelem - 1, 4, dtype=int)):
                for gauss_index in range(ssp.ngaus):
                    ssp.sig = sig1
                    ssp.mastfp(ip, gauss_index)
                    gamma1 = ssp.fgamm.copy()
                    ssp.sig = 10 * sig1
                    ssp.mastfp(ip, gauss_index)
                    op_diffs.append(float(np.max(np.abs(ssp.fgamm - gamma1))))
            ssp.sig = sig1

            energies = ssp.tbkev.copy()
            beam_energies = ssp.E_nbi_k.copy()
            energy_mask = np.any(
                np.abs(energies[:, None] - beam_energies) <= 0.05 * beam_energies, axis=1
            )
            if not np.any(energy_mask):
                raise ValueError("No velocity grid point falls within ±5% of a configured NBI beam energy")
            rows = []
            for power in [.1, .5, 1., 2., 5., 10.]:
                rhs = b0 + power * (b1 - b0)
                raw = solve_banded((ssp.iwd, ssp.iwd), A, rhs)
                coefficients = raw.reshape(ssp.nptu, 2 * ssp.nfvrb)[:, 0::2]
                raw_f = coefficients @ ssp.plgcmu
                raw_peak = float(np.max(raw_f[energy_mask, :]))
                ssp.fb = raw.copy()
                ssp.disfun(1)
                normalized_f = ssp.tbfu @ ssp.plgcmu
                window = normalized_f[energy_mask, :]
                i_energy, i_mu = np.unravel_index(np.argmax(window), window.shape)
                actual_energy = energies[np.flatnonzero(energy_mask)[i_energy]]
                rows.append({
                    "power_Wcm3": power,
                    "raw_peak": raw_peak,
                    "fnorm": float(ssp.fnorm),
                    "normalized_peak": float(np.max(window)),
                    "peak_energy_keV": float(actual_energy),
                    "peak_mu": float(ssp.xmu[i_mu]),
                    "f0_boundary": float(raw[0]),
                })
                np.savez_compressed(
                    probe_dir / f"P-{power}.npz", raw=raw, raw_distribution=raw_f,
                    normalized_distribution=normalized_f, energies=energies, mu=ssp.xmu
                )

        result = {
            "probe_dir": str(probe_dir), "config_path": str(config_path),
            "configured_power_Wcm3": configured_power, "reference_power_Wcm3": 1.0,
            "beam_energies_keV": beam_energies.tolist(), "seconds": time.monotonic() - started,
            "nptu": ssp.nptu, "legmod": ssp.legmod, "nmu": ssp.nmu,
            "emax_keV": float(ssp.tbkev[-1]), "Ti": ssp.ctx.tempi,
            "dqlw": float(ssp.dqlw),
            "background_source_max": float(np.max(np.abs(ssp.tab_source))),
            "fb1_mxw": float(ssp.fb1_mxw),
            "operator_coefficient_maxdiff_P1_vs_P10": max(op_diffs), "rows": rows,
        }
        (probe_dir / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "int_1.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_scan(args.config, args.output_dir), indent=2))
