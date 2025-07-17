import os
import numpy as np
from scipy.signal import firwin  # To create FIR kernels
from tqdm import tqdm
from modulo_vki.utils import conv_m, conv_m_2D, switch_eigs


def temporal_basis_mPOD(K, Nf, Ex, F_V, Keep, 
                        boundaries, MODE='reduced', 
                        dt=1,FOLDER_OUT: str = "./", 
                        MEMORY_SAVING: bool = False,SAT: int = 100,
                        n_Modes=10, eig_solver: str = 'svd_sklearn_randomized', conv_type: str = '1d', verbose: bool = True):
    '''
    This function computes the PSIs for the mPOD. In this implementation, a "dft-trick" is proposed, in order to avoid
    expansive SVDs. Randomized SVD is used by default for the diagonalization.
    
    :param K: 
        np.array  Temporal correlation matrix
    :param dt: float.   
        1/fs, the dt between snapshots. Units in seconds.
    :param Nf: 
        np.array. Vector collecting the order of the FIR filters used in each scale. It must be of size len(F_V) + 1, where the first element defines 
        the low pass filter, and the last one is the high pass filter. The rest are for the band-pass filters.
    :param Ex: int.
        Extension at the boundaries of K to impose the boundary conditions (see boundaries). It must be at least as Nf, and of size len(F_V) + 1, where the first
        element is the low pass filter, and the last one is the high pass filter. The rest are band-pass filters.
    :param F_V: np.array. 
        Frequency splitting vector, containing the frequencies of each scale (see article). If the time axis is in seconds, these frequencies are in Hz.
    :param Keep: np.array. 
        Vector defining which scale to keep.
    :param boundaries: str -> {'nearest', 'reflect', 'wrap' or 'extrap'}. 
        In order to avoid 'edge effects' if the time correlation matrix is not periodic, several boundary conditions can be used. Options are (from scipy.ndimage.convolve):
        ‘reflect’ (d c b a | a b c d | d c b a)    The input is extended by reflecting about the edge of the last pixel.
        ‘nearest’ (a a a a | a b c d | d d d d)    The input is extended by replicating the last pixel.
        ‘wrap’ (a b c d | a b c d | a b c d)       The input is extended by wrapping around to the opposite edge.
    :param MODE: tr -> {‘reduced’, ‘complete’, ‘r’, ‘raw’}
        As a final step of this algorithm, the orthogonality is imposed via a QR-factorization. This parameterd define how to perform such factorization, according to numpy.
        Options: this is a wrapper to np.linalg.qr(_, mode=MODE). Check numpy's documentation.
        if ‘reduced’ The final basis will not necessarely be full. If ‘complete’ The final basis will always be full
    :param FOLDER_OUT: str. 
        This is the directory where intermediate results will be stored if the memory saving is active.It will be ignored if MEMORY_SAVING=False.              
    :param MEMORY_SAVING: Bool. 
        If memory saving is active, the results will be saved locally.  Nevertheless, since Psi_M is usually not expensive, it will be returned.        
    :param SAT: int.
        Maximum number of modes per scale. The user can decide how many modes to compute; otherwise, modulo set the default SAT=100.
    :param n_Modes: int. 
        Total number of modes that will be finally exported   
    :param eig_solver: str. 
        This is the eigenvalue solver that will be used. Refer to eigs_swith for the options.
    :param conv_type: {'1d', '2d'}
        If 1d, compute K_hat applying 1d FIR filters to the columns and then rows of the extended K.
        More robust against windowing effects but more expensive (useful for modes that are slow compared to the observation time).
        If 2d, compute K_hat applying a 2d FIR filter on the extended K.
    :return PSI_M: np.array. 
        The mPOD PSIs. Yet to be sorted ! 
    '''

    if Ex < np.max(Nf):
        raise RuntimeError("For the mPOD temporal basis computation Ex must be larger than or equal to Nf")

    #Converting F_V in radiants and initialise number of scales M
    fs = 1.0 / dt
    nyq = fs / 2.0
    
    cuts = np.asarray(F_V) / nyq 
    edges = np.concatenate(([0], cuts, [1]))  
    M = len(edges) - 1
    
    assert len(Nf) == M, "Nf must be of size M+1, where M is the number of scales. Instead got sizes {}, {}".format(len(Nf), M+1)
    assert len(F_V) == M-1, "F_V must be of size M, where M is the number of scales. Instead got sizes {}, {}".format(len(F_V), M)
    assert len(Keep) == M, "Keep must be of size M+1, where M is the number of scales. Instead got sizes {}, {}".format(len(Keep), M+1)
    
    n_t = K.shape[1]
    
    # DFT-trick below: computing frequency bins.
    freqs = np.fft.fftfreq(n_t) #* fs
    
    # init modes accumulators
    Psi_M = np.empty((n_t, 0))
    Sigma_M = np.empty((0,))
    
    def _design_filter_and_mask(m):
        # use normalized firwin in agreement to normalized freqs above
        low, high = edges[m], edges[m+1]
        if m == 0:
            # low-pass
            h = firwin(numtaps=Nf[m], cutoff=high, #fs=Fs,
                       window='hamming', pass_zero=True)
            mask = lambda f: np.abs(f) <= high/2
        elif m == M - 1:
            # high-pass
            h = firwin(numtaps=Nf[m], cutoff=low, #fs=Fs,
                       window='hamming', pass_zero=False)
            mask = lambda f: np.abs(f) >= low/2
        else:
            # band-pass
            h = firwin(numtaps=Nf[m], cutoff=[low, high], #fs=Fs,
                       window='hamming', pass_zero=False)
            mask = lambda f: (np.abs(f) >= low/2) & (np.abs(f) <= high/2)
        return h, mask
    
    for m in range(M):
        if not Keep[m]:
            if verbose:
                print(f"Skipping band {m+1}/{M}")
            continue

        # design filter & mask
        h, mask_fn = _design_filter_and_mask(m)
        band_label = f"{edges[m]*nyq:.2f}–{edges[m+1]*nyq:.2f} Hz"
        if verbose:
            print(f"\nFiltering band {m+1}/{M}: {band_label}")

        # rank estimate
        mask_idxs = mask_fn(freqs)

        R_K = min(np.count_nonzero(mask_idxs), SAT, n_Modes)
        if verbose:
            print(f" → estimating {R_K} modes from {mask_idxs.sum()} freq bins")
        
        if R_K == 0:
            print(f"skipping")
            continue

        # apply filter to correlation matrix
        conv_type = conv_type.lower()
        if conv_type == '2d':
            Kf = conv_m_2D(K, h, Ex, boundaries)
        else:
            Kf = conv_m(K, h, Ex, boundaries)

        # diagonalize
        Psi_P, Sigma_P = switch_eigs(Kf, R_K, eig_solver)

        # append
        Psi_M    = np.hstack((Psi_M,    Psi_P))
        Sigma_M = np.concatenate((Sigma_M, Sigma_P))

    # 5) Sort modes by energy and QR-polish
    order = np.argsort(Sigma_M)[::-1]
    Psi_M = Psi_M[:, order]
    if verbose:
        print("\nQR polishing...")
    PSI_M, _ = np.linalg.qr(Psi_M, mode=MODE)

    Sigma_M = Sigma_M[order] # potentially used if effect of qr polishing is negligible on PSI_M


    
    if MEMORY_SAVING:
        os.makedirs(FOLDER_OUT + '/mPOD', exist_ok=True)
        np.savez(FOLDER_OUT + '/mPOD/Psis', Psis=PSI_M)

    return PSI_M[:, :n_Modes], Sigma_M[:n_Modes]

    

def dft(N_T, F_S, D, FOLDER_OUT, SAVE_DFT=False):
    """
    Computes the Discrete Fourier Transform (DFT) from the provided dataset.

    Note
    ----
    Memory saving feature is currently not supported by this function.

    Parameters
    ----------
    N_T : int
        Number of temporal snapshots.

    F_S : float
        Sampling frequency in Hz.

    D : np.ndarray
        Snapshot matrix.

    FOLDER_OUT : str
        Directory path where results are saved if `SAVE_DFT` is True.

    SAVE_DFT : bool, default=False
        If True, computed results are saved to disk and released from memory.

    Returns
    -------
    Phi_F : np.ndarray
        Complex spatial structures corresponding to each frequency mode.
        
    Sorted_Freqs : np.ndarray
        Frequency bins in Hz, sorted in ascending order.

    SIGMA_F : np.ndarray
        Real amplitudes associated with each frequency mode.
    """
    n_t = int(N_T)
    Freqs = np.fft.fftfreq(n_t) * F_S  # Compute the frequency bins
    
    # FFT along the snapshot axis
    PHI_SIGMA = np.fft.fft(D, axis=1) / np.sqrt(n_t)
    sigma_F = np.linalg.norm(PHI_SIGMA, axis=0)  # Compute the norm of each column
    
    # make phi_F orthonormal 
    Phi_F = PHI_SIGMA / sigma_F
    
    # Sort  
    Indices = np.argsort(-sigma_F)  # find indices for sorting in decreasing order
    Sorted_Sigmas = sigma_F[Indices]  # Sort all the sigmas
    Sorted_Freqs = Freqs[Indices]  # Sort all the frequencies accordingly.
    Phi_F = Phi_F[:, Indices]  # Sorted Spatial Structures Matrix
    SIGMA_F = Sorted_Sigmas  # Sorted Amplitude Matrix (vector)
    
    if SAVE_DFT:
        os.makedirs(FOLDER_OUT + 'DFT', exist_ok=True)
        np.savez(FOLDER_OUT + 'DFT/dft_fitted', Freqs=Sorted_Freqs, Phis=Phi_F, Sigmas=SIGMA_F)
        
    return Phi_F, Sorted_Freqs, SIGMA_F


def Temporal_basis_POD(K, SAVE_T_POD=False, FOLDER_OUT='./', 
                       n_Modes=10,eig_solver: str = 'eigh',verbose=True):
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
    if verbose:
        print("Diagonalizing K...")
    Psi_P, Sigma_P = switch_eigs(K, n_Modes, eig_solver)
    

    if SAVE_T_POD:
        os.makedirs(FOLDER_OUT + "/POD/", exist_ok=True)
        print("Saving POD temporal basis")
        np.savez(FOLDER_OUT + '/POD/temporal_basis', Psis=Psi_P, Sigmas=Sigma_P)

    return Psi_P, Sigma_P 