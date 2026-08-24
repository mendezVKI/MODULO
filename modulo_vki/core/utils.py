import numpy as np
from scipy import signal
from scipy.sparse.linalg import svds, eigsh
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import eigh
from sklearn.utils.extmath import randomized_svd
import warnings
from scipy.linalg import norm
from scipy.signal.windows import get_window 


def Bound_EXT(S, Ex, boundaries):
    """
   This function computes the extension of a signal for
   filtering purposes

   :param S: The Input signal
   :param Nf: The Size of the Kernel (must be an odd number!)
   :param boundaries: The type of extension:
           ‘reflect’ (d c b a | a b c d | d c b a)    The input is extended by reflecting about the edge of the last pixel.
           ‘nearest’ (a a a a | a b c d | d d d d)    The input is extended by replicating the last pixel.
           ‘wrap’ (a b c d | a b c d | a b c d)       The input is extended by wrapping around to the opposite edge.
           ‘extrap’ Extrapolation                     The input is extended via linear extrapolation.


   """
    # We first perform a zero padding
    # Ex=int((Nf-1)/2) # Extension on each size
    size_Ext = 2 * Ex + len(S)  # Compute the size of the extended signal
    S_extend = np.zeros((int(size_Ext)))  # Initialize extended signal
    S_extend[Ex:int((size_Ext - Ex))] = S;  # Assign the Signal on the zeroes

    if boundaries == "reflect":
        LEFT = np.flip(S[0:Ex])  # Prepare the reflection on the left
        RIGHT = np.flip(S[len(S) - Ex:len(S)])  # Prepare the reflectino on the right
        S_extend[0:Ex] = LEFT
        S_extend[len(S_extend) - Ex:len(S_extend)] = RIGHT
    elif boundaries == "nearest":
        LEFT = np.ones(Ex) * S[0]  # Prepare the constant on the left
        RIGHT = np.ones(Ex) * S[len(S) - 1]  # Prepare the constant on the Right
        S_extend[0:Ex] = LEFT
        S_extend[len(S_extend) - Ex:len(S_extend)] = RIGHT
    elif boundaries == "wrap":
        LEFT = S[len(S) - Ex:len(S)]  # Wrap on the Left
        RIGHT = S[0:Ex]  # Wrap on the Right
        S_extend[0:Ex] = LEFT
        S_extend[len(S_extend) - Ex:len(S_extend)] = RIGHT
    elif boundaries == "extrap":
        LEFT = np.ones(Ex) * S[0]  # Prepare the constant on the left
        RIGHT = np.ones(Ex) * S[len(S) - 1]  # Prepare the constant on the Right
        S_extend[0:Ex] = LEFT
        S_extend[len(S_extend) - Ex:len(S_extend)] = RIGHT
        print('Not active yet, replaced by nearest')
    return S_extend


def conv_m(K, h, Ex, boundaries):
    """
   This function computes the 2D convolution by perfoming 2 sets of 1D convolutions.
   Moreover, we here use the fft with an appropriate extension
   that avoids the periodicity condition.

   :param K: Matrix to be filtered
   :param h: The 1D Kernel of the filter
   :param boundaries: The type of extension:
           ‘reflect’ (d c b a | a b c d | d c b a)    The input is extended by reflecting about the edge of the last pixel.
           ‘nearest’ (a a a a | a b c d | d d d d)    The input is extended by replicating the last pixel.
           ‘wrap’ (a b c d | a b c d | a b c d)       The input is extended by wrapping around to the opposite edge.
           ‘extrap’ Extrapolation                     The input is extended via linear extrapolation.
   """
    # Filter along the raws
    n_t = np.shape(K)[0]
    # Ex=int(n_t/2)
    K_F1 = np.zeros(np.shape(K))
    K_F2 = np.zeros(np.shape(K))
    # K_F=np.zeros(np.shape(K))
    for k in range(0, n_t):
        S = K[:, k]
        S_Ext = Bound_EXT(S, Ex, boundaries)
        S_Filt = signal.fftconvolve(S_Ext, h, mode='valid')
        # Compute where to take the signal
        Ex1 = int((len(S_Filt) - len(S)) / 2)
        # K_F1[k,:]=S_Filt[Ex:(len(S_Filt)-Ex)]
        K_F1[:, k] = S_Filt[Ex1:(len(S_Filt) - Ex1)]
    for k in range(0, n_t):
        S = K_F1[k, :]
        S_Ext = Bound_EXT(S, Ex, boundaries)
        S_Filt = signal.fftconvolve(S_Ext, h, mode='valid')
        # Compute where to take the signal
        Ex1 = int((len(S_Filt) - len(S)) / 2)
        # K_F2[:,k]=S_Filt[Ex:(len(S_Filt)-Ex)]
        K_F2[k, :] = S_Filt[Ex1:(len(S_Filt) - Ex1)]
        # K_F=K_F1+K_F2
    return K_F2

def conv_m_2D(K, h, Ex, boundaries):

    # Extended K
    K_ext = np.pad(K, Ex, mode=boundaries)

    # Filtering matrix
    h_mat = np.outer(np.atleast_2d(h).T, np.atleast_2d(h))

    # Filtering
    K_filt = signal.fftconvolve(K_ext, h_mat, mode='valid')

    # Interior K
    Ex1 = int((len(K_filt) - len(K)) / 2)
    K_F2 = K_filt[Ex1:(len(K_filt) - Ex1), Ex1:(len(K_filt) - Ex1)]

    return K_F2

def _loop_gemm(a, b, c=None, chunksize=100):
    size_i = a.shape[0]
    size_zip = a.shape[1]

    size_j = b.shape[1]
    size_alt_zip = b.shape[0]

    if size_zip != size_alt_zip:
        ValueError("Loop GEMM zip index is not of the same size for both tensors")

    if c is None:
        c = np.zeros((size_i, size_j))

    istart = 0
    for i in range(int(np.ceil(size_i / float(chunksize)))):

        left_slice = slice(istart, istart + chunksize)
        left_view = a[left_slice]

        jstart = 0
        for j in range(int(np.ceil(size_j / float(chunksize)))):
            right_slice = slice(jstart, jstart + chunksize)
            right_view = b[:, right_slice]

            c[left_slice, right_slice] = np.dot(left_view, right_view)
            jstart += chunksize

        istart += chunksize

    return c


def svds_RND(K, R_K):
    """
    Quick and dirty implementation of randomized SVD
    for computing eigenvalues of K. We follow same input/output structure
    as for the svds in scipy
    """
    svd = TruncatedSVD(R_K)
    svd.fit_transform(K)
    Psi_P = svd.components_.T
    Lambda_P = svd.singular_values_
    return Psi_P, Lambda_P


def overlap(array, len_chunk, len_sep=1):
    """
    Returns a matrix of all full overlapping chunks of the input `array`, with a chunk
    length of `len_chunk` and a separation length of `len_sep`. Begins with the first full
    chunk in the array. 
    This function is taken from https://stackoverflow.com/questions/38163366/split-list-into-separate-but-overlapping-chunks
    it is designed to split an array with certain overlaps
    
    :param array: 
    :param len_chunk: 
    :param len_sep: 
    :return array_matrix:
    """

    n_arrays = int(np.ceil((array.size - len_chunk + 1) / len_sep))

    array_matrix = np.tile(array, n_arrays).reshape(n_arrays, -1)

    columns = np.array(((len_sep * np.arange(0, n_arrays)).reshape(n_arrays, -1) + np.tile(
        np.arange(0, len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    rows = np.array((np.arange(n_arrays).reshape(n_arrays, -1) + np.tile(
        np.zeros(len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    return array_matrix[rows, columns]



def switch_svds(A, n_modes, svd_solver='svd_sklearn_truncated'):
    """
    Utility function to switch between different svd solvers
    for the SVD of the snapshot matrix. This is a true SVD solver.

    --------------------------------------------------------------------------------------------------------------------
    Parameters:
    -----------
    :param A: np.array,
        Array of which compute the SVD
    :param n_modes: int,
        Number of modes to be computed. Note that if the `svd_numpy` method is chosen, the full matrix are
        computed, but then only the first n_modes are returned. Thus, it is not more computationally efficient.
    :param svd_solver: str,
        can be:
            'svd_numpy'.
             This uses np.linalg.svd.
             It is the most accurate but the slowest and most expensive.
             It computes all the modes.

            'svd_sklearn_truncated'
            This uses the TruncatedSVD from scikitlearn. This uses either
            svds from scipy or randomized from sklearn depending on the size
            of the matrix. These are the two remaining options. 
            The merit of chosing this is to let sklearn take the decision as to 
            what to use. One prefers to force the usage of any of those with the other two
            options

            'svd_scipy_sparse'
            This uses the svds from scipy.

            'svd_sklearn_randomized',
            This uses the randomized from sklearn.

    Returns
    --------
    :return Psi_P, np.array (N_S x n_modes)
    :return Sigma_P, np.array (n_modes)
    """
    if svd_solver.lower() == 'svd_sklearn_truncated':
        svd = TruncatedSVD(n_modes)
        X_transformed = svd.fit_transform(A)
        Phi_P = X_transformed / svd.singular_values_
        Psi_P = svd.components_.T
        Sigma_P = svd.singular_values_
    elif svd_solver.lower() == 'svd_numpy':
        Phi_P, Sigma_P, Psi_P = np.linalg.svd(A)
        Phi_P = Phi_P[:, 0:n_modes]
        Psi_P = Psi_P.T[:, 0:n_modes]
        Sigma_P = Sigma_P[0:n_modes]
    elif svd_solver.lower() == 'svd_sklearn_randomized':
        Phi_P, Sigma_P, Psi_P = randomized_svd(A, n_modes)
        Psi_P = Psi_P.T
    elif svd_solver.lower() == 'svd_scipy_sparse':
        Phi_P, Sigma_P, Psi_P = svds(A, k=n_modes)
        Psi_P = Psi_P.T
        # It turns out that this does not rank them in decreasing order.
        # Hence we do it manually:
        idx = np.flip(np.argsort(Sigma_P))
        Sigma_P = Sigma_P[idx]
        Phi_P = Phi_P[:, idx]
        Psi_P = Psi_P[:, idx]

    return Phi_P, Psi_P, Sigma_P


def switch_eigs(A, n_modes, eig_solver):
    """
    Utility function to switch between different eig solvers in a consistent way across different
    methods of the package.
    --------------------------------------------------------------------------------------------------------------------
    Parameters:
    -----------
    :param A: np.array,
        Array of which compute the eigenvalues
    :param n_modes: int,
        Number of modes to be computed. Note that if the `svd_numpy` method is chosen, the full matrix are
        computed, but then only the first n_modes are returned. Thus, it is not more computationally efficient.
    :param eig_solver: str,
        can be:
            'svd_sklearn_randomized',
            This uses svd truncated approach, which picks either randomized svd or scipy svds.
            By default, it should pick mostly the first.

            'eigsh' from scipy sparse. This is a compromise between the previous and the following.

            'eigh' from scipy lin alg. This is the most precise, although a bit more expensive

    Returns
    --------
    :return Psi_P, np.array (N_S x n_modes)
    :return Sigma_P, np.array (n_modes)
    """
    if eig_solver.lower() == 'svd_sklearn_randomized':
        Psi_P, Lambda_P = svds_RND(A, n_modes)
    elif eig_solver.lower() == 'eigh':
        n = np.shape(A)[0]
        Lambda_P, Psi_P = eigh(A, subset_by_index=[n - n_modes, n - 1])
        # It turns out that this does not rank them in decreasing order.
        # Hence we do it manually:
        idx = np.flip(np.argsort(Lambda_P))
        Lambda_P = Lambda_P[idx]
        Psi_P = Psi_P[:, idx]
    elif eig_solver.lower() == 'eigsh':
        Lambda_P, Psi_P = eigsh(A, k=n_modes)
        # It turns out that this does not rank them in decreasing order.
        # Hence we do it manually:
        idx = np.flip(np.argsort(Lambda_P))
        Lambda_P = Lambda_P[idx]
        Psi_P = Psi_P[:, idx]
    else:
        raise ValueError('eig_solver must be svd_sklearn_randomized, eigh or eigsh')

    Sigma_P = np.sqrt(Lambda_P)

    return Psi_P, Sigma_P

def segment_and_fft(D: np.ndarray,
                    F_S: float, 
                    L_B: int,
                    O_B: int,
                    n_processes: int = 1) -> tuple:
    """
    Partition the snapshot matrix D into overlapping time windows and compute
    the FFT of each block (optionally in parallel). Auxiliary function for the
    SPOD (Towne approach).

    Parameters
    ----------
    D : ndarray, shape (n_space, n_time)
        Snapshot matrix, where each column is the state at a time instant.
    F_S : float
        Sampling frequency in Hz.
    L_B : int
        Length of each FFT block (window size).
    O_B : int
        Number of samples to overlap between consecutive blocks.
    n_processes : int, optional
        Number of parallel processes for block FFT. If 1, uses vectorized np.fft.
        
    Returns
    -------
    D_hat : ndarray, shape (n_space, n_freq_bins, n_blocks)
        FFT of each block along the time axis, only non-negative frequencies.
    freqs_pos : ndarray, shape (n_freq_bins,)
        Non-negative frequency vector corresponding to the second axis of D_hat.
    """
    
    n_s, n_t = D.shape 
    
    # For this SPOD, assemble a tensor of overlapping chunks, similarly to
    # Welch's method. Its shape is (n_s, n_B, n_P), where n_s is the number of
    # spatial points, n_B is the block length, and n_P is the number of blocks.
    ind = np.arange(n_t)
    indices = overlap(ind, len_chunk=L_B, len_sep=O_B)
    
    n_p, n_b = indices.shape
    
    # D_t = np.zeros((n_s, n_b, n_p), dtype=D.dtype) # we enforce same data type with the main D (memory saving)
    D_t = D[:, indices].transpose(0, 2, 1)
     
    if n_processes > 1:
        print('-------------------------------------------------- \n')
        print('|\t Performing FFT of D_t in parallel on {} processors \t|'.format(n_processes))
        print('-------------------------------------------------- \n')
        
        try:
            from joblib import Parallel, delayed
            
            def _fft_block(block):
                # block is D_t[:, :, k]
                return np.fft.fft(block, n=n_b, axis=1)

            fft_blocks = Parallel(n_jobs=n_processes, backend='threading')(
                delayed(_fft_block)(D_t[:, :, k]) for k in range(n_p)
            )
            # stack back into (n_s, n_b, n_p)
            D_hat_full = np.stack(fft_blocks, axis=2)
        except:
            n_processes = 1
            raise ImportWarning("Parallel library joblib is not installed. Reverting to single process.")
    else:
        D_hat_full = np.fft.fft(D_t, n=n_b, axis=1)
        
    freqs = np.fft.fftfreq(n_b) * F_S 
    pos = freqs >= 0
    
    # only returning positive frequencies and discarding complex conjugates
    freqs_pos = freqs[pos]
    
    return D_hat_full, freqs_pos, n_processes
    
def pod_from_dhat(D_hat: np.ndarray, 
                  n_modes: int, 
                  n_freqs: int,
                  svd_solver: str,
                  n_processes: int=1):
    """
    Compute SPOD modes & energies from the FFT tensor.

    Parameters
    ----------
    D_hat : ndarray, shape (n_s, n_freqs, n_blocks)
        FFT of each block, only nonnegative frequencies.
    n_Modes : int
        Number of modes to extract per frequency.
    n_processes : int
        If >1, parallelize the SVDs over frequencies.

    Returns
    -------
    Phi : ndarray, shape (n_s, n_Modes, n_freqs)
        Spatial SPOD modes.
    Sigma : ndarray, shape (n_Modes, n_freqs)
        Modal energies.
    """
    # n_freqs is n_b but considering only positive freqs.
    
    n_s, n_b, n_p = D_hat.shape

    Phi = np.zeros((n_s, n_modes, n_freqs), dtype=complex)
    Sigma = np.zeros((n_modes, n_freqs))

    if n_processes > 1:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            warnings.warn(
                'joblib not installed: falling back to serial POD'
            )
            n_processes = 1

    if n_processes > 1:
        # def _svd_slice(j):
        #     U,V,Sigma=switch_svds(D_hat[:, j, :], n_modes,
        #                           svd_solver=svd_solver)

        #     return U[:, :n_modes], Sigma[:n_modes] / (n_s * n_b)
        def _svd_slice(j):
            U, s, _ = np.linalg.svd(D_hat[:, j, :], full_matrices=False)
            return U[:, :n_modes], s[:n_modes] / (n_s * n_b) 

        results = Parallel(
            n_jobs=n_processes,
            backend='threading'
        )(
            delayed(_svd_slice)(j) for j in range(n_freqs)
        )
        # joblib keeps same order of outputs
        for j, (Uj, sj) in enumerate(results):
            Phi[:, :, j] = Uj
            Sigma[:, j] = sj
    else:
        for j in range(n_freqs):
            U, s, _ = np.linalg.svd(
                D_hat[:, j, :], full_matrices=False
            )
            Phi[:, :, j] = U[:, :n_modes]
            Sigma[:, j] = s[:n_modes] / (n_s * n_b)

    return Phi, Sigma


def apply_weights(D: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    If `weights` is empty, return D unchanged.
    If `weights` has length n_s, assume 1D grid and tile automatically.
    If `weights` has length 2*n_s, assume 2D grid and use directly.
    Returns D_star = D * sqrt(w) applied column-wise.
    """
    n_s, n_t = D.shape
    if weights.size == 0:
        return D
    w = np.asarray(weights, dtype=D.dtype)
    if w.size == n_s:
        # 1D grid: automatically broadcast each weight to full time series
        w_full = np.repeat(w, n_t//n_s)  # or tile as needed for time dims
    elif w.size == 2*n_s:
        w_full = w
    else:
        raise ValueError(f"weights must be length {n_s} or {2*n_s}, got {w.size}")
    # apply weights: scale each row i by sqrt(w_full[i])
    return (D.T * np.sqrt(w_full)).T


####  FAST MPOD UTILS  ####
def sortScalesByEnergy(F_V, Keep, freq, mode, sourceMat):
    """
    This function sorts scales by decreasing energy.

    NOTE:
    Sorting the scales by decreasing energy increases the chance of correctly recognizing
    and skipping low energy scales further in the algorithm. It also reduces the number of modes
    kept at each scale, since the most energetic scales are processed first.
    It brings close to no additional computational overhead, since scale energies have to be computed
    anyway for the kept scales, and sorting at most a few tens of scales is negligible.
    However, it can bring significant computational savings, especially when high energy
    dynamics is located at high frequencies.

    Parameters
    ----------
    F_V : list of floats >=0
        Frequency splitting vector in ascending order. Must obey F_V[0]=0, F_V[-1]=f_Nyq and F_V[:-1]<f_Nyq. 

    Keep : list of bool
        Which frequency bands to keep.

    freq : 1D np.ndarray
        Array of frequencies generated during fft.

    mode : str
        Whether or not the global cross-spectral density matrix is available.
    
    sourceMat : 2D complex np.ndarray
        Matrix to be processes. Either K_F or D_hat.

    Returns
    -------
    scaleOrder : 1D np.ndarray of int
        Array of indices indicating the ordering of bands with respect to energy.

    E : 1D np.ndarray of float >=0
        Vector of scale energies (unsorted, i.e., in frequency band order).
    
    f : 2D np.ndarray of float >=0
        Matrix collecting starting and ending frequency of scales.
        rows : scales (unsorted, i.e., in frequency band order)
        columns : starting and ending freq of scale, respectively

    indTot : list of 1D np.ndarray of int >0
        List of arrays containing indices corresponding to appropriate scales. Scales in unsorted order.

    scalesToKeep : 1D np.ndarray of int
        Indices of scales to keep.

    Raises
    ------
    This function does not raise any exceptions by itself. See numpy and scipy documentation 
    of the functions used in the code to see, which error messages you may get. 
    """

    # inic
    indTot = []
    scalesToKeep = np.nonzero(Keep == 1)[0]
    noOfKept = scalesToKeep.size
    E = np.zeros(noOfKept)
    
    print("Precomputing scale energies for %d kept scales ..."%noOfKept)
    print("Keeping scales ",scalesToKeep+1)

    absFreq = np.abs(freq)  # Compute absolute values of frequencies

    # Get diagonal of K_F if available
    if mode == "fullK":
        diag = np.diag(sourceMat).real

    f = np.column_stack((F_V[scalesToKeep], F_V[scalesToKeep + 1])) # Starting and ending frequencies of scales

    # Precompute frequency indices and energies for all kept scales 
    # Must be done no matter what so this does not bring any overhead
    i = 0
    for k in scalesToKeep:

        # Get indices of frequencies in current scale
        if F_V[k+1] < np.max(absFreq)-absFreq[1]/2:  # End freq of scale is not Nyquist freq
            indices = np.nonzero((absFreq>=F_V[k])&(absFreq<F_V[k+1]))[0] # Exclude end freq from scale, since it belongs to the next scale
        else:   # End freq is Nyquist freq, include it in the scale
            indices = np.nonzero(absFreq>=F_V[k])[0]
        indTot.append(indices)

        # Scale energies
        if mode == "fullK":
            E[i] = np.sum(diag[indices]) # This is O(N)
        else:
            E[i] = norm(sourceMat[:,indices],ord='fro',check_finite=False)**2   # This is O(Ns*N)

        i += 1

    # Order scales by decreasing energy
    scaleOrder = np.flip(np.argsort(E))


    return scaleOrder, E, f, indTot, scalesToKeep


def taperBlockBounds(indices, winType, taperSize, Nt, fBounds, fs):
    """
    This function computes the mask to taper frequency block edges to reduce Gibbs phenomenon.

    Parameters
    ----------
    indices : 1D np.ndarray of int
        Indices of frequencies in the scale to be extended.
    
    winType : str
        Type of window to be used for tapering.

    taperSize : int > 0
        Size of the tapering region on each side of the frequency block.
        
    Nt : int
        Number of snapshots in data.

    fBounds : 2 element 1D np.ndarray of float
        Frequency bounds of the block in Hz. fBounds[0] = lower bound, fBounds[1] = upper bound.
    
    fs : float > 0
        Sampling frequency of the dataset in Hz.

    Returns
    -------
    newInd : 1D np.ndarray of int
        New indices of frequencies in the extended block.
    
    maskTot : 2D np.ndarray of float >0
        Smoothing mask to be applied to the extended block.

    Raises
    ------
    AssertionError
        When wrong shaped blocks are encountered.

    ValueError
        When invalid frequency block bounds are provided.

    May throw unexpected error messages when the filter does not have local minima.
    See the note in the code for more details.
    """

    window = get_window(winType,2*taperSize,fftbins=False)  # Get window to be used as tapering function
    if window[0] < 1e-9 and window[-1] < 1e-9:  # If window has leading and trailing zeros, crop them out
        window = get_window(winType,2*(taperSize+1),fftbins=False) # Get larger window to ensure we have the right amount of non-zero values after cropping
        window = window[1:-1] # Crop out the leading and trailing zeros of the window
    # Get tapering functions from window    
    tf1 = window[:taperSize]    
    tf2 = window[taperSize:]

    noOfInd = indices.size

    # fs/Nt is the frequency resolution (i.e. the bin width) of the signal. The freq vector is freq = [0, fs/Nt, 2*fs/Nt, ...]
    if fBounds[0] >= 0.5*fs/Nt and fBounds[1] <= fs/2 - 0.5*fs/Nt:  # Block is fully inside frequency domain => bandpass, all sided taper

        assert noOfInd % 2 == 0, "Invalid frequency indices provided for scale %s Hz, expected even number of indices, got odd"%fBounds
        assert noOfInd >= 4*taperSize, "Scale %s Hz has size smaller than smoothing region, cannot create mask."%fBounds

        noOfEntr = noOfInd//2

        mask1 = np.ones(noOfEntr)
        mask1[:taperSize] = tf1
        mask1[-taperSize:] = tf2

        mask1D = np.zeros(noOfInd)
        mask1D[:noOfEntr] = mask1
        mask1D[noOfEntr:] = mask1

    elif fBounds[0] < 0.5*fs/Nt and fBounds[1] > fs/2 - 0.5*fs/Nt:
        raise ValueError("Processed scale spans the whole frequency domain. Please split it into at least two scales or use classical POD.")

    elif fBounds[0] < 0.5*fs/Nt:   # Block containing f=0 is on the matrix borders => lowpass, one sided taper

        assert noOfInd % 2 == 1, "Invalid frequency indices provided for scale %s Hz, expected odd number of indices, got even"%fBounds
        assert noOfInd >= 2*taperSize, "Scale %s Hz has size smaller than smoothing region, cannot create mask."%fBounds

        noOfEntr = noOfInd//2 + 1 # +1 to include the DC component index

        mask1 = np.ones(noOfEntr)
        mask1[-taperSize:] = tf2

        mask2 = np.ones(noOfEntr-1)
        mask2[:taperSize] = tf1

        mask1D = np.zeros(indices.size)
        mask1D[:noOfEntr] = mask1
        mask1D[noOfEntr:] = mask2

    elif fBounds[1] > fs/2 - 0.5*fs/Nt: # Block containing f_Nyq is contiguous in frequency domain => only one set of indices
        #assert (np.diff(indices) == 1).all(), "Invalid frequency indices for scale %s Hz provided, expected contiguous block of indices."%fBounds
        assert noOfInd >= 2*taperSize, "Scale %s Hz has size smaller than smoothing region, cannot create mask."%fBounds

        mask1D = np.ones(indices.size)
        mask1D[:taperSize] = tf1
        mask1D[-taperSize:] = tf2

    else:
        raise ValueError("Invalid frequency block bounds provided.")

    return mask1D
