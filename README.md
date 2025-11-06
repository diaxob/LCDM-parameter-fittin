# Project Overview

Joint estimation of flat ΛCDM parameters from **three probes** — Type Ia SNe (μ–z), cosmic-chronometer **H(z)**, and **BAO** (D_V/r_d). We fit (\theta=(H_0,\Omega_m,\Omega_\Lambda)) with a soft flatness prior and compare a custom **Gauss–Newton (GN)**, custom **Levenberg–Marquardt (LM)**, and SciPy’s `least_squares`. The pipeline outputs figures and a JSON report with uncertainties and bootstrap stats.

## Data & Model

* **Inputs:**
  `z, μ_obs, σ_μ` (SNe), `z, H_obs, σ_H` (km/s/Mpc), `z, D_V/r_d, σ` (BAO).
  We assume (r_d) is absorbed in BAO data.
* **Flat ΛCDM:**
  (E(z)=\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}),
  (H(z)=H_0E(z)),
  (D_M(z)=\frac{c}{H_0}\int_0^z \frac{dz'}{E(z')}),
  (D_L=(1+z)D_M), (\mu(z)=5\log_{10}(D_L/10\text{ pc})),
  (D_A=D_L/(1+z)^2), (D_V = \big[(1+z)^2 D_A^2 , (cz/H)\big]^{1/3}).

## Residual Stacking & Prior

We stack normalized residuals:
[
r(\theta)=
\begin{bmatrix}
(\mu_{\rm obs}-\mu_{\rm model})/\sigma_\mu\
(H_{\rm obs}-H_{\rm model})/\sigma_H\
((D_V/r_d)*{\rm obs}-(D_V/r_d)*{\rm model})/\sigma
\end{bmatrix},
\quad
r_{\rm flat}=\frac{\Omega_m+\Omega_\Lambda-1}{\sigma_{\rm flat}}
]
with a small (\sigma_{\rm flat}) (e.g. 0.01–0.03). The Jacobian is a vertical block stack of dataset Jacobians plus the prior row.

## Solvers & Implementation Notes

* **GN (custom):** finite-difference Jacobians for SNe/BAO (small relative step (\epsilon(1+|x_j|))); **column scaling** of (J) (normalize each column), tiny **ridge** on (J^\top J), and **backtracking line search**. These stabilize steps and fix scale disparities among parameters.
* **LM (custom):** adaptive damping (J^\top J+\lambda I), same residuals/Jacobian for parity with GN/SC.
* **SciPy:** `least_squares(residuals, jac=jacobian_hybrid, max_nfev=…)` (trust-region).
* **Analytic vs numeric Jacobians:** analytic derivatives for (H(z)); numeric for (\mu(z)) and (D_V/r_d).
* **Numerical hygiene:** guard against zeros in logs/distances, tight tolerances in `quad`, reasonable `limit`, and unit consistency (km/s/Mpc, Mpc, pc).

## Why Degeneracy Appears — and How We Fix It

Columns of (J) for (\Omega_m) and (\Omega_\Lambda) are **nearly collinear** for typical redshift ranges → the Gram matrix (J^\top J) has **small eigenvalues** (huge condition number). That yields unstable GN steps and inflates ((J^\top J)^{-1}) (large parameter errors).
We break the degeneracy by:

1. **Flatness prior** (r_{\rm flat}) (soft constraint (\Omega_k=0)),
2. **Column scaling** and a tiny ridge,
3. Mixing **complementary probes** (SNe + H(z) + BAO).
   Once the cost surface is “well-conditioned”, GN, LM and SciPy converge to **the same (\hat\theta)**, and bootstrap/covariance errors agree.

## Uncertainty Estimation

* **Covariance:** ( \mathrm{Cov}(\hat\theta)\approx s^2 (J^\top J)^{-1}), (s^2=\chi^2/\mathrm{ndof}), computed stably via **SVD** cutoff for near-singular modes.
* **Bootstrap:** resample each dataset independently; refit with GN/LM/SC to obtain distribution-free std.

## Outputs

* `parameters_ext.json`: (\hat\theta), errors (cov & bootstrap), (\chi^2), dof, nfev.
* `plot_mu_z.png`, `plot_H_z.png`, `plot_bao_dvrd.png`: data + model fits.
* `plot_cost.png`: GN/LM cost vs iteration (few steps are expected after scaling/line search).

## Where Else This Pattern Works

* Joint fits in astrophysics/cosmology: growth (f\sigma_8(z)), (E_G), AP tests, weak-lensing distances.
* Any **nonlinear least-squares** with multi-probe stacking, mixed analytic/numeric Jacobians, strong parameter correlations, need for priors/regularization.
* Engineering & biostatistics: calibration of mechanistic ODE models, pharmacokinetics, system ID with multiple sensors.

Use this repository as a template for robust multi-dataset NLS problems: stack residuals, scale Jacobians, add physically motivated priors, and validate with bootstrap.
