import os
from tqdm import tqdm
import numpy as np
import math
from scipy.signal import firwin
from scipy import signal
from sklearn.metrics.pairwise import pairwise_kernels

def CorrelationMatrix(N_T,
                      N_PARTITIONS=1,
                      MEMORY_SAVING=False,
                      FOLDER_OUT='./',
                      SAVE_K=False,
                      D=None,
                      weights=np.array([]),
                      verbose=True):
    """
    Computes the temporal correlation matrix from the provided data matrix.

    If MEMORY_SAVING is active, computation is split into partitions to reduce memory load.
    If data matrix D has been computed using MODULO, parameters like dimensions and number of partitions
    will automatically be inferred.

    Parameters
    ----------
    N_T : int
        Number of temporal snapshots.

    N_PARTITIONS : int, default=1
        Number of partitions for memory-saving computation.
        Inherited automatically if using MODULO's partitioning.

    MEMORY_SAVING : bool, default=False
        Activates partitioned computation of the correlation matrix to reduce memory usage.
        Requires pre-partitioned data according to MODULO's `_data_processing`.

    FOLDER_OUT : str, default='./'
        Output directory where the temporal correlation matrix will be saved if required.

    SAVE_K : bool, default=False
        Flag to save the computed correlation matrix K to disk. Automatically enforced
        if MEMORY_SAVING is active.

    D : np.ndarray, optional
        Data matrix used to compute the correlation matrix. Required if MEMORY_SAVING is False.

    weights : np.ndarray, default=np.array([])
        Weight vector `[w_1, w_2, ..., w_Ns]` defined as `w_i = area_cell_i / area_grid`.
        Needed only for non-uniform grids when MEMORY_SAVING is True.

    Returns
    -------
    K : np.ndarray or None
        Temporal correlation matrix if MEMORY_SAVING is False; otherwise returns None,
        as matrix is managed via disk storage in partitioned computations.
    """

    if not MEMORY_SAVING:
        if verbose:
            print("Computing Temporal correlation matrix K ...")
        K = np.dot(D.T, D)
        if verbose:
            print("Done.")

    else:
        SAVE_K = True
        if verbose:
            print("\n Using Memory Saving feature...")
        K = np.zeros((N_T, N_T))
        dim_col = math.floor(N_T / N_PARTITIONS)

        if N_T % N_PARTITIONS != 0:
            tot_blocks_col = N_PARTITIONS + 1
        else:
            tot_blocks_col = N_PARTITIONS

        for k in tqdm(range(tot_blocks_col)):

            di = np.load(FOLDER_OUT + f"/data_partitions/di_{k + 1}.npz")['di']
            if weights.size != 0:
                di = np.transpose(np.transpose(di) * np.sqrt(weights))

            ind_start = k * dim_col
            ind_end = ind_start + dim_col

            if (k == tot_blocks_col - 1) and (N_T - dim_col * N_PARTITIONS > 0):
                dim_col = N_T - dim_col * N_PARTITIONS
                ind_end = ind_start + dim_col

            K[ind_start:ind_end, ind_start:ind_end] = np.dot(di.transpose(), di)

            block = k + 2

            while block <= tot_blocks_col:
                dj = np.load(FOLDER_OUT + f"/data_partitions/di_{block}.npz")['di']
                if weights.size != 0:
                    dj = np.transpose(np.transpose(dj) * np.sqrt(weights))

                ind_start_out = (block - 1) * dim_col
                ind_end_out = ind_start_out + dim_col

                if (block == tot_blocks_col) and (N_T - dim_col * N_PARTITIONS > 0):
                    dim_col = N_T - dim_col * N_PARTITIONS
                    ind_end_out = ind_start_out + dim_col
                    dj = dj[:, :dim_col]

                K[ind_start:ind_end, ind_start_out:ind_end_out] = np.dot(di.T, dj)

                K[ind_start_out:ind_end_out, ind_start:ind_end] = K[ind_start:ind_end, ind_start_out:ind_end_out].T

                block += 1

                dim_col = math.floor(N_T / N_PARTITIONS)

    if SAVE_K:
        os.makedirs(FOLDER_OUT + '/correlation_matrix', exist_ok=True)
        np.savez(FOLDER_OUT + "/correlation_matrix/k_matrix", K=K)

    return K if not MEMORY_SAVING else None


def spectral_filter(K: np.ndarray, N_o:int, f_c: float) -> np.ndarray:
    """
    Zero‐phase band‐pass filter of the correlation matrix K along its diagonals.
    Used for the SPOD proposed by Sieber  et al.

    Parameters
    ----------
    K : (n_t, n_t) array
        Original temporal correlation matrix.
    N_o : int
        Semi‐order of the FIR filter (true filter length = 2*N_o+1).
    f_c : float
        Normalized cutoff frequency (0 < f_c < 0.5).

    Returns
    -------
    K_F : (n_t, n_t) array
        The filtered correlation matrix.
    """
    n_t = K.shape[0]
    
    # extend K for edge-padding
    K_ext = np.pad(
            K,
            pad_width=((N_o, N_o), (N_o, N_o)),
            mode='constant',           # or 'edge', 'reflect', etc.
            constant_values=0
    )

    # Fill the edges ( a bit of repetition but ok.. )
    # Row-wise, Upper part
    for i in range(0, N_o):
        K_ext[i, i:i + n_t] = K[0, :]

    # Row-wise, bottom part
    for i in range(N_o + n_t, n_t + 2 * N_o):
        K_ext[i, i - n_t + 1:i + 1] = K[-1, :]

        # Column-wise, left part
    for j in range(0, N_o):
        K_ext[j:j + n_t, j] = K[:, 0]

        # Column-wise, right part
    for j in range(N_o + n_t, 2 * N_o + n_t):
        K_ext[j - n_t + 1:j + 1, j] = K[:, -1]

    # K_e = np.zeros((n_t + 2 * N_o, n_t + 2 * N_o))
    # From which we clearly know that:
    # K_e[N_o:n_t + N_o, N_o:n_t + N_o] = K
    
    # create 2D kernel for FIR 
    h1d = firwin(N_o, f_c)
    L = K_ext.shape[0]
    
    pad_l = (L - N_o) // 2
    pad_r = L - N_o - pad_l
    
    # symmetrically padded kernel in 1D
    h1d_pad = np.pad(h1d, (pad_l, pad_r))
    
    # we make it 2D diagonal
    h2d = np.diag(h1d_pad)
    
    # finally filter K_ext and return the trimmed filtered without boundaries
    K_ext_filt = signal.fftconvolve(K_ext, h2d, mode='same')
    K_F = K_ext_filt[N_o : N_o + n_t, N_o : N_o + n_t]
    
    return K_F

def kernelized_K(D, M_ij, k_m, metric, cent, alpha):
    
    n_s, n_t = D.shape 
    
    gamma = - np.log(k_m) / M_ij
    K_zeta = pairwise_kernels(D.T, metric=metric, gamma=gamma) # kernel substitute of the inner product 

    # Center the Kernel Matrix (if cent is True):
    if cent:
        H = np.eye(n_t) - 1 / n_t * np.ones_like(K_zeta)
        K_zeta = H @ K_zeta @ H.T
        
    # add `Ridge term` to enforce strictly pos. def. eigs and well-conditioning
    K_r = K_zeta + alpha * np.eye(n_t)
    
    return K_r 
        
    
    
