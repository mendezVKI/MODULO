import os
from sys import platform as _platform
import numpy as np


# import jax.numpy as jnp


def Temporal_basis_POD(K, SAVE_T_POD=False, FOLDER_OUT='./'):
    """
    This method computes the POD basis. For some theoretical insights, you can find
    the theoretical background of the proper orthogonal decomposition in a nutshell here:

    https://youtu.be/8fhupzhAR_M
    --------------------------------------------------------------------------------------------------------------------
    Parameters:
    ----------

    :param FOLDER_OUT: str
            Folder in which the results will be saved (if SAVE_T_POD=True)
    :param K: np.array
            Temporal correlation matrix
    :param SAVE_T_POD: bool
            A flag deciding whether the results are saved on disk or not. If the MEMORY_SAVING feature is active, it is
            switched True by default.
    --------------------------------------------------------------------------------------------------------------------
    Returns:
    --------

    :return: Psi_P: np.array
            POD Psis
    :return: Sigma_P: np.array
            POD Sigmas
    """

    if not (_platform == "linux") or (_platform == "linux2"):
        '''
        Checking the os. If you're using windows you should think to switch it over a Linux machine or a WSL2
        '''

        Psi_P, Lambda_P, _ = np.linalg.svd(K)
        Sigma_P = np.sqrt(Lambda_P)
    else:
        pass
        # Psi_P, Lambda_P, _ = jnp.linalg.svd(K)
        # Sigma_P = np.sqrt(Lambda_P)

    if SAVE_T_POD:
        os.makedirs(FOLDER_OUT + "/POD/", exist_ok=True)
        print("Saving POD temporal basis")
        np.savez(FOLDER_OUT + '/POD/temporal_basis', Psis=Psi_P, Lambdas=Lambda_P, Sigmas=Sigma_P)

    return Psi_P, Sigma_P
