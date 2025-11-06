import numpy as np
from scipy.integrate import quad
import pandas as pd
from methods import gn_mod, lm_mod
from functions import *
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import json
import os
import argparse

# H_in, Om_in, Ol_in = 70, 0.3, 0.7
H_in, Om_in, Ol_in = 68, 0.23, 0.65
x0 = [H_in, Om_in, Ol_in]
sigma_flat = 0.03

def residuals(theta):
    #впоследствии пригодится для bootstrap
    z_mu = sne['z'].to_numpy(float)
    mu_obs = sne['mu_obs'].to_numpy(float)
    sigma_mu = sne['sigma_mu'].to_numpy(float)

    z_hz = hz['z'].to_numpy(float)
    H_obs = hz['H_obs'].to_numpy(float)
    sigma_hz = hz['sigma_H'].to_numpy(float)

    z_bao = bao['z'].to_numpy(float)
    Dvrd_obs = bao['Dv_over_rd'].to_numpy(float)
    sigma_bao = bao['sigma'].to_numpy(float)

    r_mu  = (-mu_obs + mu_model(z_mu,  *theta)) / sigma_mu
    r_hz  = (-H_obs + H_model (z_hz,  *theta)) / sigma_hz
    r_bao = (-Dvrd_obs + Dv_over_rd(z_bao, *theta)) / sigma_bao
    #добавление набора с другой чувствительностью по z, ломающего вырождение
    r_flat = np.array([(theta[1] + theta[2] - 1.0) / sigma_flat]) #сильная корреляция Omega_m и Omega_л в случае малых z приводит к появлению 
                                                                  # малых собтсвенных значений для матрицы грама J^T J -> огромная обусловленность
    return np.concatenate([r_mu, r_hz, r_bao, r_flat])

def jacobian_numeric_sne(x):
    eps = 1e-5
    z_in = sne['z'].to_numpy(float)
    sigma_in = sne['sigma_mu'].to_numpy(float)
    n = len(x)
    f0 = mu_model(z_in, *x)
    m = len(f0)
    J_sne = np.zeros((m, n), dtype = float)
    for j in range(n):
        step = eps
        delta = np.zeros(n)
        delta[j] = step
        J_sne[:, j] = (mu_model(z_in, *(np.array(x) + delta)) - mu_model(z_in, *x)) / (step * sigma_in)
    return J_sne

def jacobian_numeric_bao(x):
    eps = 1e-5
    z_in = bao['z'].to_numpy(float)
    sigma_in = bao['sigma'].to_numpy(float)
    n = len(x)
    f0 = Dv_over_rd(z_in, *x)
    m = len(f0)
    J_bao = np.zeros((m, n), dtype = float)
    for j in range(n):
        step = eps
        delta = np.zeros(n)
        delta[j] = step
        J_bao[:, j] = (Dv_over_rd(z_in, *(np.array(x) + delta)) - Dv_over_rd(z_in, *x)) / (step * sigma_in)
    return J_bao

def jacobian_analytical_hz(x):
    H0_in, Om_in, Ol_in = x[0], x[1], x[2]
    z_in = hz['z'].to_numpy(float)
    sigma_in = hz['sigma_H'].to_numpy(float)

    arg = Om_in * (1.0 + z_in)**3 + Ol_in
    E_h = np.sqrt(np.maximum(arg, 1e-12))

    dH_dH0 = E_h
    dH_dOm = H0_in * ((1 + z_in)**3) / (2 * E_h)
    dH_dOl = H0_in * (1.0) / (2 * E_h)
    J_h = np.column_stack([dH_dH0/sigma_in, dH_dOm/sigma_in, dH_dOl/sigma_in])
    return J_h

def jacobian_hybrid(x):
    blocks = []
    J1 = jacobian_numeric_sne(x); blocks.append(J1)
    J2 = jacobian_analytical_hz(x); blocks.append(J2)
    J3 = jacobian_numeric_bao(x); blocks.append(J3)
    J_prior = np.array([[0.0, 1.0 / sigma_flat, 1.0 / sigma_flat]], dtype=float)
    blocks.append(J_prior)
    return np.vstack(blocks) #if blocks else np.zeros((0, 3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""ΛCDM joint fit: SNe + H(z) + 
                                     BAO with GN / LM / SciPy and bootstrap""")
    parser.add_argument("--sne", default="jla_mub_synth.csv")
    parser.add_argument("--hz", default="hz_synth.csv")
    parser.add_argument("--bao", default="bao_synth.csv")
    parser.add_argument("--no-header-sne", action="store_true")
    parser.add_argument("--no-header-hz", action="store_true")
    parser.add_argument("--no-header-bao", action="store_true")
    parser.add_argument("--x0", nargs=3, type=float, 
                        metavar=("H0","Om","Ol"), default=[70.0,0.3,0.7])
    parser.add_argument("--methods", nargs="+", 
                        choices=["gn","lm","scipy","all"], default=["all"])
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sne = load_sne(args.sne, has_header=not args.no_header_sne)
    hz  = load_hz(args.hz,  has_header=not args.no_header_hz)
    bao = load_bao(args.bao, has_header=not args.no_header_bao)

    z_mu = sne['z'].to_numpy(float)
    mu_obs = sne['mu_obs'].to_numpy(float)
    sigma_mu = sne['sigma_mu'].to_numpy(float)

    z_hz = hz['z'].to_numpy(float)
    H_obs = hz['H_obs'].to_numpy(float)
    sigma_hz = hz['sigma_H'].to_numpy(float)

    z_bao = bao['z'].to_numpy(float)
    Dvrd_obs = bao['Dv_over_rd'].to_numpy(float)
    sigma_bao = bao['sigma'].to_numpy(float)

    x0 = list(map(float, args.x0))
    do_gn = ("gn" in args.methods) or ("all" in args.methods)
    do_lm = ("lm" in args.methods) or ("all" in args.methods)
    do_sc = ("scipy" in args.methods) or ("all" in args.methods)

    results = {}
    if do_gn:
        result_theta_gn = gn_mod(residuals, jacobian_hybrid, x0)
        theta_gn = result_theta_gn.x
        # print(theta_gn)
        results["GN"] = result_theta_gn
    if do_lm:
        result_theta_lm = lm_mod(residuals, jacobian_hybrid, x0)
        theta_lm = result_theta_lm.x
        # print(theta_lm)
        results["LM"] = result_theta_lm
    if do_sc:
        result_theta_scipy = least_squares(
                lambda th: residuals(th), x0)
        theta_sc = result_theta_scipy.x
        # print(theta_sc)
        results["SciPy"] = result_theta_scipy

    def _summary_for(theta, res_obj=None):
        J = jacobian_hybrid(theta)
        r = residuals(theta)
        chi2 = float(r @ r)
        ndof = r.size - len(theta)
        cov = np.linalg.inv(J.T @ J)
        errs = np.sqrt(np.diag(cov))
        nfev = int(getattr(res_obj, "nfev", 0))
        return {"theta": theta.tolist(), "theta_err": errs.tolist(),
                "chi2": chi2, "ndof": ndof, "nfev": nfev}

    report = {"sizes": {
        "N_SN": int(len(sne)), "N_H": int(len(hz)), "N_BAO": int(len(bao))
    }}

    if do_gn:
        report["Gauss-Newton"] = _summary_for(results["GN"].x, results["GN"])
    if do_lm:
        report["Levenberg-Marquardt"] = _summary_for(results["LM"].x, results["LM"])
    if do_sc:
        report["SciPy"] = _summary_for(results["SciPy"].x, results["SciPy"])

    if not args.no_plots:
        z_mu_grid = np.linspace(z_mu.min(), z_mu.max(), 400)
        z_hz_grid = np.linspace(z_hz.min(), z_hz.max(), 400)
        z_bao_grid = np.linspace(z_bao.min(), z_bao.max(), 400)

        if do_gn:
            mu_fit_gn = mu_model(z_mu_grid, *theta_gn)
            H_fit_gn  = H_model(z_hz_grid, *theta_gn)
            Dv_fit_gn = Dv_over_rd(z_bao_grid, *theta_gn)
        if do_lm:
            mu_fit_lm = mu_model(z_mu_grid, *theta_lm)
            H_fit_lm  = H_model(z_hz_grid, *theta_lm)
            Dv_fit_lm = Dv_over_rd(z_bao_grid, *theta_lm)
        if do_sc:
            mu_fit_sc = mu_model(z_mu_grid, *theta_sc)
            H_fit_sc  = H_model(z_hz_grid, *theta_sc)
            Dv_fit_sc = Dv_over_rd(z_bao_grid, *theta_sc)

        plt.rcParams['figure.dpi'] = 600

        #μ(z)
        plt.figure(figsize=(7,5))
        plt.title("μ(z)")
        plt.errorbar(z_mu, mu_obs, yerr=sigma_mu, fmt=".", 
                     ms = 3,  alpha=0.75, label="данные")
        if do_gn: plt.plot(z_mu_grid, mu_fit_gn, lw=2, label="модель (GN)")
        if do_lm: plt.plot(z_mu_grid, mu_fit_lm, lw=2, label="модель (LM)")
        if do_sc: plt.plot(z_mu_grid, mu_fit_sc, lw=2, label="модель (SC)")
        plt.xlabel("z"); plt.ylabel("μ")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 
                                                     "plot_mu_z.png"))
        plt.close()

        # H(z)
        plt.figure(figsize=(7,5))
        plt.title("H(z)")
        plt.errorbar(z_hz, H_obs, yerr=sigma_hz, fmt=".", 
                     ms=3, alpha=0.9, zorder=3, label="данные")
        if do_gn: plt.plot(z_hz_grid, H_fit_gn, lw=2, 
                           alpha=0.9, zorder=2, label="модель (GN)")
        if do_lm: plt.plot(z_hz_grid, H_fit_lm, lw=2, 
                           alpha=0.9, zorder=2, label="модель (LM)")
        if do_sc: plt.plot(z_hz_grid, H_fit_sc, lw=2, 
                           alpha=0.9, zorder=2, label="модель (SC)")
        plt.xlabel("z"); plt.ylabel("H, km/s/Mpc")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, 
                                                     "plot_H_z.png"))
        plt.close()

        # BAO: Dv/rd
        plt.figure(figsize=(7,5))
        plt.title("BAO: $D_V/r_d$")
        plt.errorbar(z_bao, Dvrd_obs, yerr=sigma_bao, fmt=".", 
                     ms=7, alpha=0.9, zorder=3, label="данные")
        if do_gn: plt.plot(z_bao_grid, Dv_fit_gn, lw=2, 
                           alpha=0.9, zorder=2, label="модель (GN)")
        if do_lm: plt.plot(z_bao_grid, Dv_fit_lm, lw=2, 
                           alpha=0.9, zorder=2, label="модель (LM)")
        if do_sc: plt.plot(z_bao_grid, Dv_fit_sc, lw=2, 
                           alpha=0.9, zorder=2, label="модель (SC)")
        plt.xlabel("z"); plt.ylabel("$D_V/r_d$")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig((os.path.join(args.outdir, 
                                                      "plot_bao_dvrd.png")))
        plt.close()

    if args.bootstrap > 0:
        B = int(args.bootstrap)
        rng = np.random.default_rng(args.seed)
        def _resample(df):
            idx = rng.integers(len(df), size=len(df))
            return df.iloc[idx].reset_index(drop=True)
        _sne0, _hz0, _bao0 = sne.copy(), hz.copy(), bao.copy()
        boot_gn = []; boot_lm = []; boot_sc = []
        start = results["LM"].x if "LM" in results else x0
        for _ in range(B):
            sne = _resample(_sne0)
            hz = _resample(_hz0)
            bao = _resample(_bao0)
            if do_gn:
                rb = gn_mod(residuals, jacobian_hybrid, start)
                boot_gn.append(rb.x)
            if do_lm:
                rb = lm_mod(residuals, jacobian_hybrid, start)
                boot_lm.append(rb.x)
            if do_sc:
                rb = least_squares(residuals, start, jac=jacobian_hybrid, max_nfev=200)
                boot_sc.append(rb.x)
        sne, hz, bao = _sne0, _hz0, _bao0
        if do_gn: boot_gn = np.asarray(boot_gn, float)
        if do_lm: boot_lm = np.asarray(boot_lm, float)
        if do_sc: boot_sc = np.asarray(boot_sc, float)
        report["bootstrap_meta"] = {"iterations": B, "seed": args.seed}
        if do_gn: report.setdefault("bootstrap", {})["GN"] = {"mean": boot_gn.mean(0).tolist(), 
                                                              "std": boot_gn.std(0, ddof=1).tolist()}
        if do_lm: report.setdefault("bootstrap", {})["LM"] = {"mean": boot_lm.mean(0).tolist(), 
                                                              "std": boot_lm.std(0, ddof=1).tolist()}
        if do_sc: report.setdefault("bootstrap", {})["SciPy"] = {"mean": boot_sc.mean(0).tolist(), 
                                                                 "std": boot_sc.std(0, ddof=1).tolist()}

    with open(os.path.join(args.outdir, "parameters_ext.json"), "w", 
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(os.path.join(args.outdir, "parameters_ext.json"))
