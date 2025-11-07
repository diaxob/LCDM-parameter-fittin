# Joint ΛCDM Parameter Estimation from SNe Ia, H(z) and BAO

A compact pipeline for joint estimation of **flat ΛCDM** parameters from three probes:
**Type Ia SNe** (μ–z), **cosmic chronometers H(z)**, and **BAO** ($D_V/r_d$).
We fit $\theta=(H_0,\Omega_m,\Omega_\Lambda)$ with a soft flatness prior
and compare a custom **Gauss–Newton (GN)**, custom **Levenberg–Marquardt (LM)**,
and SciPy’s `least_squares`. The code outputs figures and a JSON report with
uncertainties and bootstrap statistics.

---

## Data & Model

**Inputs (examples in `data/`):**
- `jla_mub_synth.csv`: `z, mu_obs, sigma_mu`
- `hz_synth.csv`: `z, H_obs, sigma_H` (km/s/Mpc)
- `bao_synth.csv`: `z, Dv_over_rd, sigma` (dimensionless)

**Flat ΛCDM**

Inline definitions: $E(z)$, $H(z)=H_0E(z)$, $D_L=(1+z)D_M$, $\mu(z)=5\log_{10}(D_L/10\,\mathrm{pc})$, $D_A=D_L/(1+z)^2$, $D_V$ as below.

Block equations (leave blank lines above/below for GitHub to render):

$$
E(z)=\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}
$$

$$
H(z)=H_0\,E(z)
$$

$$
D_M(z)=\frac{c}{H_0}\int_0^z \frac{dz'}{E(z')}
$$

$$
D_L(z)=(1+z)\,D_M(z),\qquad
\mu(z)=5\log_{10}\\Big(\frac{D_L(z)}{10\,\mathrm{pc}}\Big)
$$

$$
D_A(z)=\frac{D_L(z)}{(1+z)^2}
$$

$$
D_V(z)=\Big[(1+z)^2 D_A^2(z)\cdot \frac{c z}{H(z)}\Big]^{1/3}
$$

BAO observable: $D_V(z)/r_d$ (with $r_d$ fixed consistently with the data).

---

## Residual Stacking & Prior

Normalized residuals:

$$
r(\theta)=
\begin{bmatrix}
\frac{\mu_{\rm obs}-\mu_{\rm model}}{\sigma_\mu}\\
\frac{H_{\rm obs}-H_{\rm model}}{\sigma_H}\\
\frac{(D_V/r_d)_{\rm obs}-(D_V/r_d)_{\rm model}}{\sigma}
\end{bmatrix}
$$

Soft flatness prior (helps condition $J^\top J$):

$$
r_{\rm flat}=\frac{\Omega_m+\Omega_\Lambda-1}{\sigma_{\rm flat}},\qquad
\sigma_{\rm flat}\in[0.01,\,0.03].
$$

The full Jacobian is a vertical stack of dataset Jacobians (numeric for SNe/BAO,
analytic for $H(z)$) plus one prior row.

---

## Solvers & Implementation Notes

- **Gauss–Newton (custom):**
  relative FD step $\epsilon(1+|x_j|)$; **column scaling** of $J$;
  tiny ridge on $J^\top J$; **backtracking line search**.
- **Levenberg–Marquardt (custom):** adaptive damping $J^\top J+\lambda I$.
- **SciPy:** `least_squares(residuals, jac=jacobian_hybrid)`.

**Numerical hygiene:** avoid zeros in logs/distances, tight `quad` tolerances/`limit`,
unit consistency (km/s/Mpc, Mpc, pc).

---

## Why Degeneracy Appears — and Fix

Columns for $\Omega_m$ and $\Omega_\Lambda$ can be nearly collinear over typical $z$,
so $J^\top J$ gets small eigenvalues (large condition number). We add the flatness prior,
scale $J$, and combine SNe+H(z)+BAO. After conditioning improves, GN, LM and SciPy give
nearly identical $\hat{\theta}$; bootstrap and covariance errors agree.

---

## Uncertainty Estimation

Covariance via SVD-stabilized inverse of $J^\top J$:
$\mathrm{Cov}(\hat\theta)\approx s^2 (J^\top J)^{-1}$ with $s^2=\chi^2/\mathrm{ndof}$.
Bootstrap: independent resampling of each dataset; refit with GN/LM/SciPy.

---

## Usage

```bash
python cosmology_pp.py \
  --sne data/jla_mub_synth.csv --no-header-sne \
  --hz  data/hz_synth.csv \
  --bao data/bao_synth.csv \
  --methods all \
  --bootstrap 200 --seed 42 \
  --outdir results
Outputs (results/): parameters_ext.json, plot_mu_z.png, plot_H_z.png,
plot_bao_dvrd.png, plot_cost.png.

---

## Where Else to Use
Joint cosmology fits (growth $f\sigma_8$, AP tests, weak-lensing distances) and any
nonlinear least-squares with multi-probe stacking, strong parameter correlations,
and priors/regularization.
