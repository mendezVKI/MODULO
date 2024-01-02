import os
import numpy as np


# import jax.numpy as jnp
from sklearn.decomposition import TruncatedSVD
# For efficient linear algebra
from numpy import linalg as LA
# For Timing
import time



def dmd_s(D_1,D_2,n_Modes, F_S, SAVE_T_DMD=False, FOLDER_OUT='./'):
    """
    This method computes the Dynamic Mode Decomposition (DMD). 

    https://youtu.be/8fhupzhAR_M
    --------------------------------------------------------------------------------------------------------------------
    Parameters:
    ----------

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
            A flag deciding whether the results are saved on disk or not. If the MEMORY_SAVING feature is active, it is
            switched True by default.
    :param n_Modes: int
           number of modes that will be computed
    --------------------------------------------------------------------------------------------------------------------
    Returns:
    --------

        :return Phi_D: np.array
                DMD Psis

        :return Lambda_D: np.array
                DMD Eigenvalues (of the reduced propagator)

        :return freqs: np.array
                Frequencies (in Hz, associated to the DMD modes)
                
         :return a0s: np.array
                Initial Coefficients of the Modes   
    """
   
    #TODO This currently does a POD of D_1. The memory saving feature
    # should be implemented in this step.
    svd = TruncatedSVD(n_Modes)
    print('Computing the SVD of the D1')
    start = time.time()
    X_transformed = svd.fit_transform(D_1)
    #%% Step 2: compute the SVD of D1 (randomized algorithm)
    Phi_P = X_transformed / svd.singular_values_
    Psi_P = svd.components_.T
    Sigma_P=svd.singular_values_
    end = time.time()
    print('SVD computed in '+ str(end - start)+'s')
   
    
    Sigma_inv=np.diag(1/Sigma_P); dt=1/F_S
    #%% Step 3: Compute approximated propagator
    P_A=LA.multi_dot([np.transpose(Phi_P),D_2,\
                     Psi_P,Sigma_inv])
        
    #%% Step 4: Compute eigenvalues of the system
    Lambda, Q = LA.eig(P_A) 
    freqs=np.imag(np.log(Lambda))/(2*np.pi*dt)
    
    #%% Step 5: Spatial structures of the DMD 
    Phi_D=LA.multi_dot([D_2,Psi_P,Sigma_inv,Q]) 
    #%% Step 6: Compute the initial coefficients 
    #a0s=LA.lstsq(Phi_D, D_1[:,0],rcond=None)
    a0s=LA.pinv(Phi_D).dot(D_1[:,0])   
    
    if SAVE_T_DMD:
        os.makedirs(FOLDER_OUT + "/DMD/", exist_ok=True)
        print("Saving DMD results")
        np.savez(FOLDER_OUT + '/DMD/dmd_decomposition',
                  Phi_D=Phi_D, Lambda=Lambda, freqs=freqs,a0s=a0s)

    return Phi_D, Lambda, freqs, a0s
