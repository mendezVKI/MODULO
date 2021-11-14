import numpy as np
from scipy import signal
from sklearn.decomposition import TruncatedSVD


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


def svds_RND(K,R_K):
  '''Quick and dirty implementation of randomized SVD
  for computing eigenvalues of K. We follow same input/output structure
  as for the svds in scipy
  '''
  svd = TruncatedSVD(R_K)
  svd.fit_transform(K)
  Psi_P = svd.components_.T
  Lambda_P =svd.singular_values_
  return Psi_P, Lambda_P
