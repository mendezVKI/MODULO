import os

import numpy as np
from scipy.signal import firwin  # To create FIR kernels
#from scipy.sparse.linalg import svds
from tqdm import tqdm

from modulo.utils._utils import conv_m, switch_eigs


def temporal_basis_mPOD(K, Nf, Ex, F_V, Keep, boundaries, MODE='reduced', dt=1,
                        FOLDER_OUT: str = "./",                        
                        MEMORY_SAVING: bool = False,
                        SAT: int = 100,
                        n_Modes=10,
                        eig_solver: str = 'svd_sklearn_randomized'):
    '''
    This function computes the PSIs for the mPOD. In this implementation, a "dft-trick" is proposed, in order to avoid
    expansive SVDs. Randomized SVD is used instead.
    --------------------------------------------------------------------------------------------------------------------
    Parameters:
    -----------

    :param K: np.array
            Temporal correlation matrix
    :param dt: float
            Time step
    :param Nf: np.array
            Filters
    :param Ex: int
            Extension. Must have Ex >= Nf 
    :param F_V: np.array
            Filter array
    :param Keep: np.array
            vector defining which scale to keep.
    :param boundaries: str
            In order to avoid 'edge effects' if the time correlation matrix is not periodic, several boundary conditions
            can be used.
            Options are (from scipy.ndimage.convolve):
            ‘reflect’ (d c b a | a b c d | d c b a)    The input is extended by reflecting about the edge of the last pixel.
            ‘nearest’ (a a a a | a b c d | d d d d)    The input is extended by replicating the last pixel.
            ‘wrap’ (a b c d | a b c d | a b c d)       The input is extended by wrapping around to the opposite edge.
    :param MODE: str
            As a final step of this algorithm, the orthogonality is imposed via a QR-factorization. This parameter
            define how to perform such factorization, according to numpy.
            Options: this is a wrapper to np.linalg.qr(_, mode=MODE).
            Check numpy's documentation.
            if ‘reduced’ The final basis will not necessarely be full
            if ‘complete’ The final basis will always be full
    :param FOLDER_OUT: str
            This is the directory where intermediate results will be stored if the memory saving is active
            It will be ignored if MEMORY_SAVING=False.
            
              
    :param MEMORY_SAVING: Bool
           If memory saving is active, the results will be saved locally
           Nevertheless, since Psi_M is usually not expensive, it will be returned.        
    :param SAT: int
            Maximum number of modes per scale. 
            The user can decide how many modes to compute; otherwise, modulo set the default SAT=100.
    :param n_Modes: int
          Total number of modes that will be finally exported   
            
    :param eig_solver: str
           This is the eigenvalue solver that will be used. Refer to eigs_swith for the options.      
            
    --------------------------------------------------------------------------------------------------------------------
    Returns:
    --------
    :return PSI_M: np.array
            mPOD PSIs
    '''

    if Ex < np.max(Nf):
        raise RuntimeError("For the mPOD temporal basis computation Ex must be larger than or equal to Nf")

    '''Converting F_V in radiants and initialise number of scales M'''
    Fs = 1 / dt
    F_Bank_r = F_V * 2 / Fs
    M = len(F_Bank_r)

    # Loop over the scales to show the transfer functions
    Psi_M = np.array([])
    Lambda_M = np.array([])
    n_t = K.shape[1]
    
   # if K_S:
   #     Ks = np.zeros((n_t, n_t, M + 1))

    '''DFT-trick below: computing frequency bins.'''
    Freqs = np.fft.fftfreq(n_t) * Fs

    print("Filtering and Diagonalizing H scale: \n")

    '''Filtering and computing eigenvectors'''

    for m in tqdm(range(0, M)):
        # Generate the 1d filter for this
        if m < 1:
            if Keep[m] == 1:
             # Low Pass Filter
             h_A = firwin(Nf[m], F_Bank_r[m], window='hamming')
             # Filter K_LP
             print('\n Filtering Largest Scale')
             K_L = conv_m(K=K, h=h_A, Ex=Ex, boundaries=boundaries)
             # R_K = np.linalg.matrix_rank(K_L, tol=None, hermitian=True)
             '''We replace it with an estimation based on the non-zero freqs the cut off frequency of the scale is '''
             F_CUT = F_Bank_r[m] * Fs / 2
             Indices = np.argwhere(np.abs(Freqs) < F_CUT)
             R_K = np.min([len(Indices), SAT])
             print(str(len(Indices)) + ' Modes Estimated')
             print('\n Diagonalizing Largest Scale')
             Psi_P, Lambda_P = switch_eigs(K_L, R_K, eig_solver) #svds_RND(K_L, R_K)
             Psi_M=Psi_P; Lambda_M=Lambda_P
            else:
             print('\n Scale '+str(m)+' jumped (keep['+str(m)+']=0)')  
            # if K_S:
            #     Ks[:, :, m] = K_L  # First large scale
                         
            # method = signal.choose_conv_method(K, h2d, mode='same')
        elif m > 0 and m < M - 1:
            if Keep[m] == 1:
                # print(m)
                print('\n Working on Scale '+str(m)+'/'+str(M))
                # This is the 1d Kernel for Band pass
                h1d_H = firwin(Nf[m], [F_Bank_r[m], F_Bank_r[m + 1]], pass_zero=False)  # Band-pass
                F_CUT1 = F_Bank_r[m] * Fs / 2
                F_CUT2 = F_Bank_r[m + 1] * Fs / 2
                Indices = np.argwhere((np.abs(Freqs) > F_CUT1) & (np.abs(Freqs) < F_CUT2))
                R_K = np.min([len(Indices), SAT])  # number of frequencies here
                print(str(len(Indices)) + ' Modes Estimated')
                # print('Filtering H Scale ' + str(m + 1) + '/' + str(M))
                K_H = conv_m(K, h1d_H, Ex, boundaries)
                # Ks[:, :, m + 1] = K_H  # Intermediate band-pass
                print('Diagonalizing H Scale ' + str(m + 1) + '/' + str(M))
                # R_K = np.linalg.matrix_rank(K_H, tol=None, hermitian=True)
                Psi_P, Lambda_P = switch_eigs(K_H, R_K, eig_solver) #svds_RND(K_H, R_K)  # Diagonalize scale
                if np.shape(Psi_M)[0]==0: # if this is the first contribute to the basis
                 Psi_M=Psi_P; Lambda_M=Lambda_P
                else:                    
                 Psi_M = np.concatenate((Psi_M, Psi_P), axis=1)  # append to the previous
                 Lambda_M = np.concatenate((Lambda_M, Lambda_P), axis=0)
            else:
                print('\n Scale '+str(m)+' jumped (keep['+str(m)+']=0)')  
      
        else: # this is the case m=M: this is a high pass
            if Keep[m] == 1:
                print('Working on Scale '+str(m)+'/'+str(M))
                # This is the 1d Kernel for High Pass (last scale)
                h1d_H = firwin(Nf[m], F_Bank_r[m], pass_zero=False)
                F_CUT1 = F_Bank_r[m] * Fs / 2
                Indices = np.argwhere((np.abs(Freqs) > F_CUT1))
                R_K = len(Indices)
                R_K = np.min([len(Indices), SAT])  # number of frequencies here
                print(str(len(Indices)) + ' Modes Estimated')
                print('Filtering H Scale ' + str(m + 1) + '/ ' + str(M))
                K_H = conv_m(K, h1d_H, Ex, boundaries)
                # Ks[:, :, m + 1] = K_H  # Last (high pass) scale
                print('Diagonalizing H Scale ' + str(m + 1) + '/ ' + str(M))
                # R_K = np.linalg.matrix_rank(K_H, tol=None, hermitian=True)
                Psi_P, Lambda_P = switch_eigs(K_H, R_K, eig_solver) #svds_RND(K_H, R_K)  # Diagonalize scale
                Psi_M = np.concatenate((Psi_M, Psi_P), axis=1)  # append to the previous
                Lambda_M = np.concatenate((Lambda_M, Lambda_P), axis=0)
            else:
                print('\n Scale '+str(m)+' jumped (keep['+str(m)+']=0)')  

    # Now Order the Scales
    Indices = np.flip(np.argsort(Lambda_M))  # find indices for sorting in decreasing order
    Psi_M = Psi_M[:, Indices]  # Sort the temporal structures
    #print(f"Size psis in mpodtime = {np.shape(Psi_M)}")
    # Now we complete the basis via re-orghotonalization
    print('\n QR Polishing...')
    PSI_M, R = np.linalg.qr(Psi_M, mode=MODE)
    print('Done!')

    if MEMORY_SAVING:
        os.makedirs(FOLDER_OUT + '/mPOD', exist_ok=True)
        np.savez(FOLDER_OUT + '/mPOD/Psis', Psis=PSI_M)
    
    return PSI_M[:,0:n_Modes]
