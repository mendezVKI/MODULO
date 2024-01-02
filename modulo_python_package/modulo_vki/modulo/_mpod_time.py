import os
import sys

import numpy as np
from scipy.signal import firwin  # To create FIR kernels
#from scipy.sparse.linalg import svds
from tqdm import tqdm

from modulo._utils import conv_m, svds_RND


def temporal_basis_mPOD(K, Nf, Ex, F_V, Keep, boundaries, MODE, dt,
                        FOLDER_OUT: str = "./",
                        K_S: bool = False,
                        MEMORY_SAVING: bool = False,
                        SAT: int = 100,
                        n_Modes=10):
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
            Extension. Ex >= Nf must hold
    :param F_V: np.array
            Filter array
    :param Keep: np.array
            To keep
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
            Options
            ‘reduced’ In this case the final basis will not necessarely be full
            ‘complete’ In this case the final basis will always be full
    :param SAT: int
            Maximum number of modes per scale. The user can decide how many modes to compute, otherwise, modulo will
            impose SAT=100.
    :param n_Modes: int
          Total number of modes that will be finally exported   
            
    --------------------------------------------------------------------------------------------------------------------
    Returns:
    --------
    :return PSI_M: np.array
            mPOD PSIs
    '''

    if Ex < np.max(Nf):
        sys.exit("For the mPOD temporal basis computation Ex must be larger or equal to Nf")

    '''Converting F_V in radiants and initialise number of scales M'''
    Fs = 1 / dt
    F_Bank_r = F_V * 2 / Fs
    M = len(F_Bank_r)

    # Loop over the scales to show the transfer functions
    Psi_M = np.array([])
    Lambda_M = np.array([])
    n_t = K.shape[1]
    if K_S:
        Ks = np.zeros((n_t, n_t, M + 1))

    '''DFT-trick below: computing frequency bins.'''
    Freqs = np.fft.fftfreq(n_t) * Fs

    print("Filtering and Diagonalizing H scale: \n")

    '''Filtering and computing eigenvectors'''

    for m in tqdm(range(0, M)):
        # Generate the 1d filter for this
        if m < 1:
            # Low Pass Filter
            h_A = firwin(Nf[m], F_Bank_r[m], window='hamming')
            # h_A2d=np.outer(h_A,h_A) # Create 2D Kernel
            # First Band Pass Filter
            # Create 1D Kernel
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
            Psi_P, Lambda_P = svds_RND(K_L, R_K)
            Psi_M = Psi_P  # In the first scale we take it as is
            Lambda_M = Lambda_P
            # TODO : this scale should also be 'skippable'

            if K_S:
                Ks[:, :, m] = K_L  # First large scale

            # Construct first band pass
            if M > 1:
                h1d_H = firwin(Nf[m], [F_Bank_r[m], F_Bank_r[m + 1]], pass_zero=False)  # Band-pass
                F_CUT1 = F_Bank_r[m] * Fs / 2
                F_CUT2 = F_Bank_r[m + 1] * Fs / 2
                Indices = np.argwhere((np.abs(Freqs) > F_CUT1) & (np.abs(Freqs) < F_CUT2))
                R_K = np.min([len(Indices), SAT])  # number of frequencies here
                print(str(len(Indices)) + ' Modes Estimated')
            else:
                h1d_H = firwin(Nf[m], F_Bank_r[m], pass_zero=False)  # Band-pass
                F_CUT1 = F_Bank_r[m] * Fs / 2
                Indices = np.argwhere(np.abs(Freqs) > F_CUT1)
                R_K = np.min([len(Indices), SAT])  # number of frequencies here
                print(str(len(Indices)) + ' Modes Estimated')

            # print('Filtering H Scale ' + str(m + 1) + '/' + str(M))
            K_H = conv_m(K, h1d_H, Ex, boundaries)
            # Ks[:, :, m + 1] = K_H  # First band pass
            # print('Diagonalizing H Scale ' + str(m + 1) + '/' + str(M))
            # R_K = np.linalg.matrix_rank(K_H, tol=None, hermitian=True)
            Psi_P, Lambda_P = svds_RND(K_H, R_K)  # Diagonalize scale
            Psi_M = np.concatenate((Psi_M, Psi_P), axis=1)  # append to the previous
            Lambda_M = np.concatenate((Lambda_M, Lambda_P), axis=0)
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
                # print('Diagonalizing H Scale ' + str(m + 1) + '/' + str(M))
                # R_K = np.linalg.matrix_rank(K_H, tol=None, hermitian=True)
                Psi_P, Lambda_P = svds_RND(K_H, R_K)  # Diagonalize scale
                Psi_M = np.concatenate((Psi_M, Psi_P), axis=1)  # append to the previous
                Lambda_M = np.concatenate((Lambda_M, Lambda_P), axis=0)
            else:
                print('\n Scale '+str(m)+' jumped (keep=0)')
        else:
            if Keep[m] == 1:
                print('Working on Scale '+str(m)+'/'+str(M))
                # This is the 1d Kernel for High Pass (last scale)
                h1d_H = firwin(Nf[m], F_Bank_r[m], pass_zero=False)
                F_CUT1 = F_Bank_r[m] * Fs / 2
                Indices = np.argwhere((np.abs(Freqs) > F_CUT1))
                R_K = len(Indices)
                R_K = np.min([len(Indices), SAT])  # number of frequencies here
                print(str(len(Indices)) + ' Modes Estimated')
                # print('Filtering H Scale ' + str(m + 1) + '/ ' + str(M))
                K_H = conv_m(K, h1d_H, Ex, boundaries)
                # Ks[:, :, m + 1] = K_H  # Last (high pass) scale
                # print('Diagonalizing H Scale ' + str(m + 1) + '/ ' + str(M))
                # R_K = np.linalg.matrix_rank(K_H, tol=None, hermitian=True)
                Psi_P, Lambda_P = svds_RND(K_H, R_K)  # Diagonalize scale
                Psi_M = np.concatenate((Psi_M, Psi_P), axis=1)  # append to the previous
                Lambda_M = np.concatenate((Lambda_M, Lambda_P), axis=0)
            else:
                print('\n Scale '+str(m)+' jumped (keep=0)')

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
        np.savez(FOLDER_OUT + 'Psis', Psis=PSI_M)

    return PSI_M[:,0:n_Modes]
