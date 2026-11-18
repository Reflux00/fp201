"""Audit current QL kernels against independent special functions/quadrature.

Run from the repository root with NumPy, SciPy and Matplotlib installed:
    python analysis/physics_ql_probe.py --output analysis/physics_ql_results.json
The solver and input configuration are not modified.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.special import eval_legendre, expit, factorial, jv, legendre, roots_legendre

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from solver_loader import load_solver

FP = load_solver()["FP"]


def kernel_solver(harmonic, order=21):
    solver = FP(SimpleNamespace(tempi=[1.0]), 101, harmonic, order,
                10.0, 36.5e6, 1.0, 0, 0, 1, 1)
    solver.kamax = 6
    solver.ksup = 6
    solver.dqlu = np.zeros(7)
    solver.fjbess = np.zeros(102)
    solver.fctinv = 1.0 / factorial(np.arange(8))
    solver.xiperp = 1.0
    solver.dqlw = 1.0
    solver.dql00 = np.zeros((order + 1, order + 1))
    solver.dql01 = np.zeros_like(solver.dql00)
    solver.dql11 = np.zeros_like(solver.dql00)
    return solver


def stable_legendre_and_mu_derivative(order, points):
    polynomials = np.array([eval_legendre(n, points) for n in range(order + 1)])
    derivatives = np.zeros_like(polynomials)
    interior = np.abs(points) < 1.0
    for n in range(1, order + 1):
        x = points[interior]
        derivatives[n, interior] = n * (
            x * polynomials[n, interior] - polynomials[n - 1, interior]
        ) / (x * x - 1.0)
        derivatives[n, points == 1.0] = n * (n + 1) / 2.0
        derivatives[n, points == -1.0] = (-1)**(n + 1) * n * (n + 1) / 2.0
    return polynomials, points * derivatives


def angular_reference(harmonic, order, quadrature_points):
    points, weights = roots_legendre(quadrature_points)
    polynomials, mu_derivative = stable_legendre_and_mu_derivative(order, points)
    weight = weights * expit(20 * (points - 0.4)) * (1 - points**2)**harmonic
    return {
        "qleg00": np.einsum("i,li,mi->lm", weight, polynomials, polynomials),
        "qleg01": np.einsum("i,li,mi->lm", weight, polynomials, mu_derivative),
        "qleg11": np.einsum("i,li,mi->lm", weight, mu_derivative, mu_derivative),
    }


def audit():
    config = json.loads((REPO / "int_1.json").read_text())
    species = config["ispec"] - 1
    vth_cm_s = 4.37695e7 * np.sqrt(config["tempi"][species] / config["atm"][species])
    xi_code = config["kperp"] * vth_cm_s * 1e-2 / (2 * np.pi * config["omega_ci"])
    umax = np.sqrt(config["energy_range"]["r_E_2" if config["ipheat"] == 2 else "r_E_1"])
    frequency = {
        "kperp_unit_assumed": "m^-1 (consistent with current cm/s conversion)",
        "vth_cm_s": float(vth_cm_s), "xi_code": float(xi_code),
        "xi_if_frequency_is_RF_Hz": float(config["ipheat"] * xi_code),
        "xi_u_max_code": float(xi_code * umax),
        "xi_u_max_if_frequency_is_RF_Hz": float(config["ipheat"] * xi_code * umax),
    }
    trapping = {str(mu): float(expit(20 * (mu - 0.4)))
                for mu in (-1.0, -0.7, -0.4, 0.0, 0.4, 0.7, 1.0)}
    angular = []
    kernels = []
    mu = np.linspace(-1.0, 1.0, 2001)
    for harmonic in (1, 2):
        for order in (21, 31, 41, 61):
            solver = kernel_solver(harmonic, order)
            solver.coelgd_with_weight_optimized(True, 0.4, 20.0)
            reference = angular_reference(harmonic, order, 200)
            refined = angular_reference(harmonic, order, 300)
            row = {"harmonic": harmonic, "legmod": order}
            for name, values in reference.items():
                actual = getattr(solver, name)[:, :, 0]
                row[name + "_max_abs_error"] = float(np.max(np.abs(actual - values)))
                row[name + "_reference_refinement_difference"] = float(np.max(np.abs(refined[name] - values)))
            row["q11_LL_current"] = float(solver.qleg11[order, order, 0])
            row["q11_LL_reference"] = float(reference["qleg11"][order, order])
            x = 0.9
            row["derivative_at_0.9_current"] = float(legendre(order).deriv()(x))
            row["derivative_at_0.9_reference"] = float(order * (
                x * eval_legendre(order, x) - eval_legendre(order - 1, x)
            ) / (x * x - 1.0))
            angular.append(row)
        solver = kernel_solver(harmonic)
        solver.coelgd_with_weight_optimized(True, 0.4, 20.0)
        for argument in (frequency["xi_u_max_code"], 0.4, 1.0, 2.0, 4.0, 6.0, 8.0):
            recommended = int(np.floor((argument - (0.516 if harmonic == 1 else 0.965)) /
                                       (0.465 if harmonic == 1 else 0.530)) + 2)
            solver.ksup = min(max(recommended, 0), 6)
            solver.qlcoef(np.float64(argument))
            reconstructed = (1 - mu**2)**(harmonic - 1) * sum(
                solver.dqlu[k] * mu**(2 * k) for k in range(solver.ksup + 1)
            )
            exact = jv(harmonic - 1, argument * np.sqrt(1 - mu**2))**2
            kernels.append({
                "harmonic": harmonic, "argument": argument,
                "recommended_ksup": recommended, "actual_ksup": solver.ksup,
                "kernel_minimum": float(reconstructed.min()),
                "mu_at_minimum": float(mu[np.argmin(reconstructed)]),
                "kernel_max_abs_error": float(np.max(np.abs(reconstructed - exact))),
                "D00_min_eigenvalue": float(np.linalg.eigvalsh(solver.dql00).min()),
            })
    return {"frequency_contract": frequency, "trapping_weights": trapping,
            "angular_quadrature": angular, "bessel_kernel": kernels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(audit(), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
