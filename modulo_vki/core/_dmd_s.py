import os
import numpy as np
from numpy import linalg as LA
from ..utils._utils import switch_svds


def dmd_s(D_1, D_2, n_Modes, F_S,
          SAVE_T_DMD: bool = False,
          FOLDER_OUT: str = './',
          svd_solver: str = 'svd_sklearn_truncated',
          verbose=True):
    """
    Compute the Dynamic Mode Decomposition (DMD) using the PIP algorithm.

    This implementation follows the Penland & Sardeshmukh PIP approach and
    recovers the same modes as the exact DMD of Tu et al. (2014).

    Parameters
    ----------
    D_1 : ndarray, shape (n_features, n_time-1)
        First snapshot matrix (columns 0 to n_t-2 of the full data).
    D_2 : ndarray, shape (n_features, n_time-1)
        Second snapshot matrix (columns 1 to n_t-1 of the full data).
    n_Modes : int
        Number of DMD modes to compute.
    F_S : float
        Sampling frequency in Hz.
    SAVE_T_DMD : bool, optional
        If True, save time‐series DMD results to disk. Default is False.
    FOLDER_OUT : str, optional
        Directory in which to save outputs when SAVE_T_DMD is True. Default is './'.
    svd_solver : str, optional
        SVD solver to use for the low‐rank approximation. Default is
        'svd_sklearn_truncated'.

    Returns
    -------
    Phi_D : ndarray, shape (n_features, n_Modes)
        Complex spatial DMD modes.
    Lambda_D : ndarray, shape (n_Modes,)
        Complex eigenvalues of the reduced propagator.
    freqs : ndarray, shape (n_Modes,)
        Frequencies (Hz) associated with each DMD mode.
    a0s : ndarray, shape (n_Modes,)
        Initial amplitudes (coefficients) of the DMD modes.
    """ 
    
    Phi_P, Psi_P, Sigma_P = switch_svds(D_1, n_Modes, svd_solver)
    if verbose:
        print('SVD of D1 rdy')
    Sigma_inv = np.diag(1 / Sigma_P)
    dt = 1 / F_S
    # %% Step 3: Compute approximated propagator
    P_A = LA.multi_dot([np.transpose(Phi_P), D_2, Psi_P, Sigma_inv])
    if verbose:
        print('reduced propagator rdy')

    # %% Step 4: Compute eigenvalues of the system
    Lambda, Q = LA.eig(P_A) # not necessarily symmetric def pos! Avoid eigsh, eigh
    freqs = np.imag(np.log(Lambda)) / (2 * np.pi * dt)
    if verbose:
        print(' lambdas and freqs rdy')

    # %% Step 5: Spatial structures of the DMD in the PIP style
    Phi_D = LA.multi_dot([D_2, Psi_P, Sigma_inv, Q])
    if verbose:
        print('Phi_D rdy')

    # %% Step 6: Compute the initial coefficients
    # a0s=LA.lstsq(Phi_D, D_1[:,0],rcond=None)
    a0s = LA.pinv(Phi_D).dot(D_1[:, 0])
    if verbose:
        print('Sigma_D rdy')

    if SAVE_T_DMD:
        os.makedirs(FOLDER_OUT + "/DMD/", exist_ok=True)
        print("Saving DMD results")
        np.savez(FOLDER_OUT + '/DMD/dmd_decomposition',
                 Phi_D=Phi_D, Lambda=Lambda, freqs=freqs, a0s=a0s)

    return Phi_D, Lambda, freqs, a0s
