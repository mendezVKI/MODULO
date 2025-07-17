import os
import numpy as np
from ..utils._utils import switch_eigs


def Temporal_basis_POD(K, SAVE_T_POD=False, FOLDER_OUT='./', 
                       n_Modes=10,eig_solver: str = 'eigh'):
    """
    This method computes the POD basis. For some theoretical insights, you can find the theoretical background of the proper orthogonal decomposition in a nutshell here: https://youtu.be/8fhupzhAR_M

    :param FOLDER_OUT: str. Folder in which the results will be saved (if SAVE_T_POD=True)
    :param K: np.array. Temporal correlation matrix
    :param SAVE_T_POD: bool. A flag deciding whether the results are saved on disk or not. If the MEMORY_SAVING feature is active, it is switched True by default.
    :param n_Modes: int. Number of modes that will be computed
    :param svd_solver: str. Svd solver to be used throughout the computation
    :return: Psi_P: np.array. POD's Psis
    :return: Sigma_P: np.array. POD's Sigmas
    """
    # Solver 1: Use the standard SVD
    # Psi_P, Lambda_P, _ = np.linalg.svd(K)
    # Sigma_P = np.sqrt(Lambda_P)

    # Solver 2: Use randomized SVD ############## WARNING #################
    # if svd_solver.lower() == 'svd_sklearn_truncated':
    #     svd = TruncatedSVD(n_Modes)
    #     svd.fit_transform(K)
    #     Psi_P = svd.components_.T
    #     Lambda_P = svd.singular_values_
    #     Sigma_P = np.sqrt(Lambda_P)
    # elif svd_solver.lower() == 'svd_numpy':
    #     Psi_P, Lambda_P, _ = np.linalg.svd(K)
    #     Sigma_P = np.sqrt(Lambda_P)
    # elif svd_solver.lower() == 'svd_sklearn_randomized':
    #     Psi_P, Lambda_P, _ = svds_RND(K, n_Modes)
    #     Sigma_P = np.sqrt(Lambda_P)
    # elif svd_solver.lower() == 'svd_scipy_sparse':
    #     Psi_P, Lambda_P, _ = svds(K, k=n_Modes)
    #     Sigma_P = np.sqrt(Lambda_P)

    print("diagonalizing K....")
    Psi_P, Sigma_P = switch_eigs(K, n_Modes, eig_solver)
    

    if SAVE_T_POD:
        os.makedirs(FOLDER_OUT + "/POD/", exist_ok=True)
        print("Saving POD temporal basis")
        np.savez(FOLDER_OUT + '/POD/temporal_basis', Psis=Psi_P, Sigmas=Sigma_P)

    return Psi_P, Sigma_P
