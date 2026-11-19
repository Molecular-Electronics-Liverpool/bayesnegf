import numpy as np
import scipy.linalg as la

# Fermi-Dirac distribution
def fermi_dirac(E, mu, kT=0.025):
    exponent = np.clip((E - mu) / kT, -500, 500)
    return 1.0 / (np.exp(exponent) + 1.0)

# Oligoyne SSH Hamiltonian
def SSH(N, eps, t_1, t_2):
    H = np.zeros((N,N))
    np.fill_diagonal(H, eps)

    for i in range(N-1):
        if i % 2 ==0:
            beta = t_1
        else:
            beta = t_2

        H[i,i+1] = beta
        H[i+1,i] = beta

    return H

# Degenerate form of Hamiltonian for orthogonal pi systems - see Jorgensen et al. J. Phys. Chem. A, 2021 - Not used here.
def SSHdual(N, eps, t_1, t_2):
    H_diag = SSH(N, eps, t_1, t_2)
    H_off_diag = np.zeros_like(H_diag)
    H = np.block([
        [H_diag, H_off_diag],
        [H_off_diag, H_diag]
        ])
    return H

# Stark Shift Matrix
def stark(N, V_applied, alpha=0.5, g=1.0):

    V = V_applied * g
    shift = (0.5 - alpha) * V

    diagonals = np.linspace(V/2, -V/2, N) + shift
    stark_shift = np.diag(diagonals)

    return stark_shift

# Self energy matrix under the WBL
def sigma(N, gamma_L, gamma_R):
    sigma = np.zeros((N,N), dtype=complex)
    sigma[0,0] = -1j * np.abs(gamma_L)
    sigma[-1,-1] += -1j * np.abs(gamma_R)
    return sigma

def transmission(N, E_start, E_end, H, gamma_L, gamma_R, V, alpha, E_points, g=1.0, starkON=True):
    eta = 1e-9
    if starkON:
        H_v = H + stark(N, V, alpha, g)
    else:
        H_v = H
    Sigma = sigma(N, gamma_L, gamma_R)
    energy_range = np.linspace(E_start, E_end, E_points)
    
    # Vectorised implementation for efficiency
    E_3D = energy_range[:, np.newaxis, np.newaxis] * np.eye(N)
    kernels = E_3D + (1j * eta) * np.eye(N) - H_v - Sigma
    GR_all = np.linalg.inv(kernels)
    gr_0_last_all = GR_all[:, 0, -1]
    Transmission_E = 4 * np.abs(gamma_L) * np.abs(gamma_R) * (np.abs(gr_0_last_all) ** 2)

    return energy_range, Transmission_E

# Sacrifices avoiding Py loops in favour of solving using a banded matrix algorithm. Should go from O(N^3) to O(N) for long chains.
# Native python Thomas algorithm would be slow in python. Useful for long chains. Py overhead slowed it down too much for this work.
def transmission_banded(N, E_start, E_end, H, gamma_L, gamma_R, V, alpha, E_points, g=1.0, starkON=True):
    eta = 1e-9
    if starkON:
        H_v = H + stark(N, V, alpha, g)
    else:
        H_v = H
    Sigma = sigma(N, gamma_L, gamma_R)

    energy_range = np.linspace(E_start, E_end, E_points)
    Transmission_E = np.zeros(E_points)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html
    ab = np.zeros((3, N), dtype=complex)
    ab[0, 1:] = -np.diag(H_v, k=1)      #Upper diagonal bit k=0 (default) is diagonal
    ab[2, :-1] = -np.diag(H_v, k=-1)    #Lower diagonal bit
    main_diag = -np.diag(H_v) - np.diag(Sigma) + (1j * eta)

    # vector to extract the last col of inv Green func
    b = np.zeros(N, dtype=complex)
    b[-1] = 1.0 #set last element = 1.0

    for i, E in enumerate(energy_range):
        ab[1, :] = E + main_diag
        x = la.solve_banded((1, 1), ab, b)
        Transmission_E[i] = 4 * np.abs(gamma_L) * np.abs(gamma_R) * (np.abs(x[0]) ** 2)

    return energy_range, Transmission_E
