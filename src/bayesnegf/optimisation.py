import numpy as np
import iv as IV
from . import transmission as tran

def prior_transform(u, N):
    x = np.array(u)

    t_scale = -10.00 + (u[0] * 6.00)
    BO = 0.4 + (u[1] * 0.2)
    
    t1 = BO * t_scale
    t2 = (1 - BO) * t_scale

    # Calculates half the HL gap assuming neutral half filled compound setting eps=0 for simplicity and WLOG
    H = tran.SSH(N, 0, t1, t2)
    evals = np.linalg.eigvalsh(H)
    homo_idx = (N // 2) - 1
    lumo_idx = N // 2
    half_gap = (evals[lumo_idx] - evals[homo_idx]) / 2.0

    x[0] = BO
    x[1] = t_scale

    if half_gap < 1e-6:
        half_gap = 1e-6

    x[2] = -half_gap + (u[2] * (2.0 * half_gap))
    x[3] = 0.35 + u[3] * 0.3       # alpha: [0.35, 0.65]

    # log_uniform sampling for couplings
    gamma_upper = -2
    gamma_lower = -0.001
    log_gammaL = np.log10(abs(gamma_lower)) + (u[4] * (np.log10(abs(gamma_upper)) - np.log10(abs(gamma_lower))))
    log_gammaR = np.log10(abs(gamma_lower)) + (u[5] * (np.log10(abs(gamma_upper)) - np.log10(abs(gamma_lower))))

    x[4] = -(10**log_gammaL)
    x[5] = -(10**log_gammaR)

    return x

def log_likelihood(params, V_exp, I_exp, sigma_exp, N, E_start, E_end):

    BO, t_scale, eps, alpha, gammaL, gammaR = params
    t1 = BO * t_scale
    t2 = (1 - BO) * t_scale

    try:
        H = tran.SSH(N, eps, t1, t2)

        I_theory = np.zeros_like(V_exp)

        for i, V in enumerate(V_exp):
            # Theoretical current in nA at 300 K
            I_theory[i] = IV.current(N, H, gammaL, gammaR, V, E_start, E_end, alpha=alpha, g=1.0, kT=0.0256, E_points=500, starkON=True) / 1e-9

  
    except Exception:
        return -np.inf

    log_I_exp = np.log10(np.abs(I_exp) + 1e-3)
    log_I_theory = np.log10(np.abs(I_theory) + 1e-3)

    sigma2 = sigma_exp**2
    log_L = -0.5 * np.sum((log_I_exp - log_I_theory) ** 2 / sigma2 + np.log(sigma2))

    if np.isnan(log_L):
        return -np.inf

    return log_L