import numpy as np
from . import transmission as tran

# Generates a current value for a given voltage
def current(N, H, gamma_L, gamma_R, V_applied, E_start, E_end, alpha=0.5, g=1.0, kT=0.025, E_points=200, starkON=True):
    
    # Chemical potentials of leads fixed symmetric, Stark handles asymmetry.
    V = V_applied * g
    mu_L = V / 2
    mu_R = -V / 2


    Energies, T_E = tran.transmission(N, E_start, E_end, H, gamma_L, gamma_R, V, alpha, E_points, g, starkON)

    f_L = tran.fermi_dirac(Energies, mu_L, kT)
    f_R = tran.fermi_dirac(Energies, mu_R, kT)

    I = T_E * (f_L - f_R)
    dE = Energies[1] - Energies[0]
    Integral = np.trapezoid(I, dx=dE)

    G0 = 7.748092e-5 # units: S

    current = G0 * Integral

    return current


