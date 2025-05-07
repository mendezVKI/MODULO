import os
import numpy as np
from scipy.signal import firwin  # To create FIR kernels
from tqdm import tqdm
from utils import conv_m, switch_eigs


def temporal_basis_mPOD(K, Nf, Ex, F_V, Keep, 
                        boundaries, MODE='reduced', 
                        dt=1,FOLDER_OUT: str = "./", 
                        MEMORY_SAVING: bool = False,SAT: int = 100,
                        n_Modes=10, eig_solver: str = 'svd_sklearn_randomized'):
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
    Lambda_M = np.empty((0,))
    
    def _design_filter_and_mask(m):
        # use normalized firwin in agreement to normalized freqs above
        low, high = edges[m], edges[m+1]
        if m == 0:
            # low-pass
            h = firwin(numtaps=Nf[m], cutoff=high, #fs=Fs,
                       window='hamming', pass_zero=True)
            mask = lambda f: np.abs(f) <= high
        elif m == M - 1:
            # high-pass
            h = firwin(numtaps=Nf[m], cutoff=low, #fs=Fs,
                       window='hamming', pass_zero=False)
            mask = lambda f: np.abs(f) >= low
        else:
            # band-pass
            h = firwin(numtaps=Nf[m], cutoff=[low, high], #fs=Fs,
                       window='hamming', pass_zero=False)
            mask = lambda f: (np.abs(f) >= low) & (np.abs(f) <= high)
        return h, mask
    
    for m in range(M):
        if not Keep[m]:
            print(f"Skipping band {m+1}/{M}")
            continue

        # design filter & mask
        h, mask_fn = _design_filter_and_mask(m)
        band_label = f"{edges[m]:.2f}–{edges[m+1]:.2f} Hz"
        print(f"\nFiltering band {m+1}/{M}: {band_label}")

        # rank estimate
        mask_idxs = mask_fn(freqs)
        R_K = min(np.count_nonzero(mask_idxs), SAT, n_Modes)
        print(f" → estimating {R_K} modes from {mask_idxs.sum()} freq bins")

        # apply filter to correlation matrix
        Kf = conv_m(K, h, Ex, boundaries)

        # diagonalize
        Psi_P, Lambda_P = switch_eigs(Kf, R_K, eig_solver)

        # append
        Psi_M    = np.hstack((Psi_M,    Psi_P))
        Lambda_M = np.concatenate((Lambda_M, Lambda_P))

    # 5) Sort modes by energy and QR-polish
    order = np.argsort(Lambda_M)[::-1]
    Psi_M = Psi_M[:, order]
    print("\nQR polishing...")
    PSI_M, _ = np.linalg.qr(Psi_M, mode=MODE)

    return PSI_M[:, :n_Modes]

    
    
    