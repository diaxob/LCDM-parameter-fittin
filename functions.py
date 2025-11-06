import numpy as np
from scipy.integrate import quad
import pandas as pd

#валидация и загрузка данных
def validate(df, name = 'DATA', z_col = 'z'):
    #nan/inf
    if df.isna().any().any():
        raise ValueError(f"{name}: найдены NaN")
    if not np.isfinite(df.to_numpy(dtype = float)).all():
        raise ValueError(f"{name}: Найдены Inf/нечисловые значения")
    #z-диапазоны
    z = df[z_col]
    if (z < 0).any():
        raise ValueError(f"{name}: есть отрицательные z")
    if z.max() > 3.0:
        print(f"[warn] {name}: max z = {z.max():.2f} (проверь диапазон)")

def load_sne(path, has_header = True):
    if has_header:
        df = pd.read_csv(path)
        df = df[['z', 'mu_obs', 'sigma_mu']]
    else:
        df = pd.read_csv(path, header = None, names = ['z', 'mu_obs', 'sigma_mu'])

    validate(df, name = 'SNE', z_col = 'z')
    if (df['sigma_mu'] <= 0).any():
        raise ValueError('SNE: sigma_mu должны быть >0')
    
    return df.reset_index(drop=True)
    
def load_hz(path, has_header = True):
    if has_header:
        df = pd.read_csv(path)
        df = df[['z', 'H_obs', 'sigma_H']]
    else:
        df = pd.read_csv(path, header = None, names = ['z', 'H_obs', 'sigma_H'])

    validate(df, name = 'H(z)', z_col = 'z')
    if (df['H_obs'] < 30).any() or (df['H_obs'] > 300).any():
        print('[warn] H_obs выходит за [30,300] — проверь единицы (km/s/Mpc)')
    if (df['sigma_H'] <= 0).any():
        raise ValueError('H(z): sigma_H должны быть > 0')
    
    return df.reset_index(drop = True)

def load_bao(path, has_header = True):
    if has_header:
        df = pd.read_csv(path)
        df = df[['z', 'Dv_over_rd', 'sigma']]
    else:
        df = pd.read_csv(path, header = None, names = ['z', 'Dv_over_rd', 'sigma'])

    validate(df, name = 'BAO', z_col = 'z')
    if (df['sigma'] <= 0).any():
        raise ValueError('BAO: sigma должны быть > 0')
    
    return df.reset_index(drop = True)

#константы и функции
c = 299792.458 #km/s
rd = 147.09 #Mpc

def E(z, Om, Ol):
    z = np.asarray(z, dtype=float)
    arg = Om * (1.0 + z)**3 + Ol
    return np.sqrt(np.maximum(arg, 1e-12))  # защита от отриц/нуля

def Dm(z, H0, Om, Ol):
    zf = float(z)
    if zf <= 0.0:
        return 0.0
    integrand = lambda k: 1.0 / E(k, Om, Ol)
    integral, _ = quad(integrand, 0.0, zf, epsabs=1e-7, epsrel=1e-7, limit=100)
    return (c / H0) * integral

def Dl(z, H0, Om, Ol):
    return (1.0 + z) * Dm(z, H0, Om, Ol)

def Da(z, H0, Om, Ol):
    return Dl(z, H0, Om, Ol) / (1 + z)**2

def mu_model(z_arr, H0, Om, Ol):
    DL = np.array([Dl(z, H0, Om, Ol) for z in z_arr], dtype=float)
    DL = np.maximum(DL, 1e-12)  # защитный порог для нуля
    return 5.0 * np.log10(DL) + 25.0

def H_model(z_arr, H0, Om, Ol):
    return np.array([H0 * E(z, Om, Ol) for z in z_arr])

def Dv_over_rd(z_arr, H0, Om, Ol):
    z_arr = np.asarray(z_arr, dtype=float)
    Hz = H_model(z_arr, H0, Om, Ol)          # >0 в ΛCDM
    Hz = np.maximum(Hz, 1e-12)               # защита от числ. нуля
    DA = np.array([Da(z, H0, Om, Ol) for z in z_arr], dtype=float)
    term = ((1.0 + z_arr)**2) * (DA**2) * (c * z_arr / Hz)
    term = np.maximum(term, 0.0)             # защита от -0.0/-ε
    DV = term**(1.0/3.0)
    return DV / rd