# Joint ΛCDM Parameter Estimation from SNe Ia, H(z) and BAO

A compact pipeline for joint estimation of **flat ΛCDM** parameters from three probes:
**Type Ia SNe** (μ–z), **cosmic chronometers H(z)**, and **BAO** \(D_V/r_d\).
We fit \(\theta=(H_0,\Omega_m,\Omega_\Lambda)\) with a soft **flatness prior**
and compare a custom **Gauss–Newton (GN)**, custom **Levenberg–Marquardt (LM)**,
and SciPy’s `least_squares`. The code outputs figures and a JSON report with
uncertainties and bootstrap statistics.

---

## Data & Model

**Inputs (examples in `data/`):**
- `jla_mub_synth.csv`: `z, mu_obs, sigma_mu`
- `hz_synth.csv`: `z, H_obs, sigma_H` (km/s/Mpc)
- `bao_synth.csv`: `z, Dv_over_rd, sigma` (dimensionless)

**Flat ΛCDM:**
\[
E(z)=\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda},\quad
H(z)=H_0 E(z),\quad
D_M(z)=\frac{c}{H_0}\int_0^z \frac{dz'}{E(z')}
\]
\[
D_L=(1+z)\,D_M,\quad
\mu(z)=5\log_{10}\!\big(D_L/10~\mathrm{pc}\big),\quad
D_A=D_L/(1+z)^2,\quad
D_V=\Big[(1+z)^2 D_A^2 \cdot \frac{cz}{H(z)}\Big]^{1/3}
\]
BAO observable: \(D_V(z)/r_d\) (with \(r_d\) fixed consistently with the data).

---

## Residual Stacking & Prior

We stack normalized residuals:
\[
r(\theta)=
\begin{bmatrix}
(\mu_{\rm obs}-\mu_{\rm model})/\sigma_\mu\\
(H_{\rm obs}-H_{\rm model})/\sigma_H\\
\big((D_V/r_d)_{\rm obs}-(D_V/r_d)_{\rm model}\big)/\sigma
\end{bmatrix}
\]
and add a soft **flatness prior**:
\[
r_{\rm flat}=\frac{\Omega_m+\Omega_\Lambda-1}{\sigma_{\rm flat}},\qquad
\sigma_{\rm flat}\in[0.01,\,0.03].
\]
The full Jacobian is a vertical block stack of dataset Jacobians (numeric for SNe/BAO,
analytic for \(H(z)\)) plus one prior row.

---

## Solvers & Implementation Notes

- **Gauss–Newton (custom):**
  - finite-difference Jacobians for SNe/BAO with a small *relative* step \(\epsilon(1+|x_j|)\)
  - **column scaling** of \(J\) (normalize each column)
  - tiny **ridge** on \(J^\top J\)
  - **backtracking line search**
- **Levenberg–Marquardt (custom):** adaptive damping \(J^\top J+\lambda I\).
- **SciPy:** `least_squares(residuals, jac=jacobian_hybrid)` (trust-region).
- **Numerical hygiene:** avoid zeros in logs/distances, tight `quad` tolerances/`limit`,
  unit consistency (km/s/Mpc, Mpc, pc).

---

## Why Degeneracy Appears — and How We Fix It

Columns of \(J\) for \(\Omega_m\) and \(\Omega_\Lambda\) are often **nearly collinear**
over typical redshift ranges. The Gram matrix \(J^\top J\) then has **small eigenvalues**
(huge condition number), which inflates \((J^\top J)^{-1}\) and destabilizes steps.
We fix this by:
1. **Flatness prior** \(r_{\rm flat}\) (soft \(\Omega_k=0\)),
2. **Column scaling** and a tiny ridge,
3. Combining **complementary probes** (SNe + H(z) + BAO).
After conditioning improves, GN, LM and SciPy converge to **the same \(\hat\theta\)** and
bootstrap/covariance errors agree.

---

## Uncertainty Estimation

- **Covariance:** \( \mathrm{Cov}(\hat\theta)\approx s^2 (J^\top J)^{-1}\), with
  \(s^2=\chi^2/\mathrm{ndof}\), computed stably via **SVD** cutoff of near-singular modes.
- **Bootstrap:** independent resampling of each dataset; refit with GN/LM/SC to obtain
  distribution-free standard deviations.

---

## Usage

```bash
python cosmology_pp.py \
  --sne data/jla_mub_synth.csv --no-header-sne \
  --hz  data/hz_synth.csv \
  --bao data/bao_synth.csv \
  --methods all \
  --bootstrap 200 \
  --seed 42 \
  --outdir results
````

**Outputs (in `results/`):**

* `parameters_ext.json` — fitted parameters (GN/LM/SC), (\chi^2), dof, uncertainties, bootstrap
* `plot_mu_z.png` — SNe Hubble diagram
* `plot_H_z.png` — H(z) fit
* `plot_bao_dvrd.png` — BAO (D_V/r_d) fit
* `plot_cost.png` — cost vs iteration (GN/LM)

---

## Where Else to Use This Pattern

* Joint cosmology fits: growth (f\sigma_8(z)), (E_G), AP tests, weak-lensing distances.
* General **nonlinear least-squares** with multi-probe stacking, mixed analytic/numeric Jacobians,
  strong parameter correlations, and the need for priors/regularization.
* Engineering/biostatistics: calibration of mechanistic ODE models, PK/PD, multi-sensor system ID.

---

## Requirements

* Python ≥ 3.10
* `numpy`, `scipy`, `pandas`, `matplotlib`
