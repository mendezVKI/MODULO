import numpy as np
from scipy import signal
from scipy.signal import firwin
from ._pod_time import Temporal_basis_POD
from ._pod_space import Spatial_basis_POD

def compute_SPOD_s(D, K, F_S, n_s, n_t,N_o=100, f_c=0.3,n_Modes=10, SAVE_SPOD=True,
                   FOLDER_OUT='./', MEMORY_SAVING=False, N_PARTITIONS=1):
    """
    This method computes the Spectral POD of your data.
    This is the one by Sieber
    et al (https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition/DCD8A6EDEFD56F5A9715DBAD38BD461A)

    :param F_S: float,
           Sampling Frequency [Hz]
    :param N_o: float,
           Semi-Order of the diagonal filter.
           Note that the filter order will be 2 N_o +1 (to make sure it is odd)
    :param f_c: float,
           cut-off frequency of the diagonal filter
    :param n_Modes: float,
           number of modes to be computed
    :param SAVE_SPOD: bool,
           If True, MODULO will save the output in self.FOLDER OUT/MODULO_tmp
    :param FOLDER_OUT: string
           Define where the out will be stored (ignored if SAVE_POD=False)
    :param MEMORY SAVING: bool
           Define if memory saving is active or not (reduntant; to be improved)
           Currently left for compatibility with the rest of MODULO.
    :param N_PARTITIONS: int
           number of partitions (if memory saving = False, it should be 1). 
           (reduntant; to be improved)
           Currently left for compatibility with the rest of MODULO.           
    :return Psi_P: np.array
           SPOD Psis
    :return Sigma_P: np.array
           SPOD Sigmas.
    :return Phi_P: np.array
            SPOD Phis
    """
    # if self.D is None:
    #     D = np.load(self.FOLDER_OUT + '/MODULO_tmp/data_matrix/database.npz')['D']
    #     SAVE_SPOD = True
    #     # TODO : Lorenzo check this stuff
    # else:
    #     D = self.D
    #
    # n_s = self.N_S  # Repeat variable for debugging compatibility
    # n_t = self.N_T
    #
    # print('Computing Correlation Matrix \n')

    # The first step is the same as the POD: we compute the correlation matrix
    # K = CorrelationMatrix(self.N_T, self.N_PARTITIONS, self.MEMORY_SAVING,
    #                       self.FOLDER_OUT, D=self.D)

    # 1. Initialize the extended
    K_e = np.zeros((n_t + 2 * N_o, n_t + 2 * N_o))
    # From which we clearly know that:
    K_e[N_o:n_t + N_o, N_o:n_t + N_o] = K

    # 2. We fill the edges ( a bit of repetition but ok.. )

    # Row-wise, Upper part
    for i in range(0, N_o):
        K_e[i, i:i + n_t] = K[0, :]

    # Row-wise, bottom part
    for i in range(N_o + n_t, n_t + 2 * N_o):
        K_e[i, i - n_t + 1:i + 1] = K[-1, :]

        # Column-wise, left part
    for j in range(0, N_o):
        K_e[j:j + n_t, j] = K[:, 0]

        # Column-wise, right part
    for j in range(N_o + n_t, 2 * N_o + n_t):
        K_e[j - n_t + 1:j + 1, j] = K[:, -1]

    # Now you create the diagonal kernel in 2D
    h_f = firwin(N_o, f_c)  # Kernel in 1D
    # This is also something that must be put in a separate file:
    # To cancel the phase lag we make this non-causal with a symmetric
    # shift, hence with zero padding as equal as possible on both sides
    n_padd_l = round((n_t - N_o) / 2);
    n_padd_r = n_t - N_o - n_padd_l

    h_f_pad = np.pad(h_f, (n_padd_l, n_padd_r))  # symmetrically padded kernel in 1D
    h_f_2 = np.diag(h_f_pad)

    # Finally the filtered K is just
    K_F = signal.fftconvolve(K_e, h_f_2, mode='same')[N_o:n_t + N_o, N_o:n_t + N_o]
    # plt.plot(np.diag(K),'b--'); plt.plot(np.diag(K_F_e),'r')

    # From now on it's just POD:
    Psi_P, Sigma_P = Temporal_basis_POD(K_F, SAVE_SPOD, FOLDER_OUT, n_Modes)
    # but with a normalization aspect to be careful about!  
    Phi_P = Spatial_basis_POD(D, N_T=n_t, PSI_P=Psi_P, Sigma_P=Sigma_P,
                              MEMORY_SAVING=MEMORY_SAVING, FOLDER_OUT=FOLDER_OUT,
                              N_PARTITIONS=N_PARTITIONS,rescale=True)

    return Phi_P, Psi_P, Sigma_P