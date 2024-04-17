import os
import numpy as np
from numpy import linalg as LA
from ..utils._utils import switch_svds


def dmd_s(D_1, D_2, n_Modes, F_S,
          SAVE_T_DMD=False,
          FOLDER_OUT='./',
          svd_solver: str = 'svd_sklearn_truncated'):
    """
    This method computes the Dynamic Mode Decomposition (DMD) using hte PIP algorithm from Penland.    
    
    :param D_1: np.array 
           First portion of the data, i.e. D[:,0:n_t-1]            
    :param D_2: np.array
           Second portion of the data, i.e. D[:,1:n_t]
    :param Phi_P, Psi_P, Sigma_P: np.arrays
           POD decomposition of D1
    :param F_S: float
           Sampling frequency in Hz
    :param FOLDER_OUT: str
           Folder in which the results will be saved (if SAVE_T_DMD=True)
    :param K: np.array
           Temporal correlation matrix
    :param SAVE_T_POD: bool
           A flag deciding whether the results are saved on disk or not. If the MEMORY_SAVING feature is active, it is switched True by default.
    :param n_Modes: int
           number of modes that will be computed
    :param svd_solver: str,
           svd solver to be used 
          
     
    :return1 Phi_D: np.array. 
          DMD's complex spatial structures
    :return2 Lambda_D: np.array. 
          DMD Eigenvalues (of the reduced propagator)
    :return3 freqs: np.array. 
          Frequencies (in Hz, associated to the DMD modes)
    :return4 a0s: np.array. 
          Initial Coefficients of the Modes
    """   
    
    Phi_P, Psi_P, Sigma_P = switch_svds(D_1, n_Modes, svd_solver)
    print('SVD of D1 rdy')
    Sigma_inv = np.diag(1 / Sigma_P)
    dt = 1 / F_S
    # %% Step 3: Compute approximated propagator
    P_A = LA.multi_dot([np.transpose(Phi_P), D_2, Psi_P, Sigma_inv])
    print('reduced propagator rdy')

    # %% Step 4: Compute eigenvalues of the system
    Lambda, Q = LA.eig(P_A) # not necessarily symmetric def pos! Avoid eigsh, eigh
    freqs = np.imag(np.log(Lambda)) / (2 * np.pi * dt)
    print(' lambdas and freqs rdy')

    # %% Step 5: Spatial structures of the DMD in the PIP style
    Phi_D = LA.multi_dot([D_2, Psi_P, Sigma_inv, Q])
    print('Phi_D rdy')

    # %% Step 6: Compute the initial coefficients
    # a0s=LA.lstsq(Phi_D, D_1[:,0],rcond=None)
    a0s = LA.pinv(Phi_D).dot(D_1[:, 0])
    print('Sigma_D rdy')

    if SAVE_T_DMD:
        os.makedirs(FOLDER_OUT + "/DMD/", exist_ok=True)
        print("Saving DMD results")
        np.savez(FOLDER_OUT + '/DMD/dmd_decomposition',
                 Phi_D=Phi_D, Lambda=Lambda, freqs=freqs, a0s=a0s)

    return Phi_D, Lambda, freqs, a0s
