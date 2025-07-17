import numpy as np
from modulo_vki.utils._utils import overlap
from tqdm import tqdm
import os

from modulo_vki.utils._utils import switch_svds



def compute_SPOD_t(D, F_S, L_B=500, O_B=250,n_Modes=10, SAVE_SPOD=True, FOLDER_OUT='/',
                   possible_svds='svd_sklearn_truncated'):
    """
    This method computes the Spectral POD of your data.
    This is the one by Town 
    et al (https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/spectral-proper-orthogonal-decomposition-and-its-relationship-to-dynamic-mode-decomposition-and-resolvent-analysis/EC2A6DF76490A0B9EB208CC2CA037717)
    
    :param D: array.
      snapshot matrix to decompose, of size N_S,N_T 
    :param F_S: float,
      Sampling Frequency [Hz]
    :param L_B: float,
      Lenght of the chunks
    :param O_B: float,
      Overlapping between blocks in the chunk
    :param n_Modes: float,
      Number of modes to be computed FOR EACH FREQUENCY
    :param SAVE_SPOD: bool,
      If True, MODULO will save the output in FOLDER OUT/MODULO_tmp
    :param possible_svds: str,      
      Svd solver to be used throughout the computation
           
    :return Psi_P_hat: np.array
      Spectra of the SPOD Modes
    :return Sigma_P: np.array
      Amplitudes of the SPOD Modes.
    :return Phi_P: np.array
      SPOD Phis
    :return freq: float
      Frequency bins for the Spectral POD
    """
        
    # if D is None:
    #     D = np.load(FOLDER_OUT + '/MODULO_tmp/data_matrix/database.npz')['D']
    #     SAVE_SPOD = True
    # else:
    #     D = D
    #
    # n_s = N_S  # Repeat variable for debugging compatibility
    # n_t = N_T
    #
    # # First comput the PS in each point (this is very time consuming and should be parallelized)
    # # Note: this can be improved a lot...! ok for the moment
    print('Computing PSD at all points\n')
    N_S,N_T=np.shape(D)

    # Step 1 : Partition the data into blocks ( potentially overlapping)
    Ind = np.arange(N_T)
    Indices = overlap(Ind, len_chunk=L_B, len_sep=O_B)

    N_B = np.shape(Indices)[1]
    N_P = np.shape(Indices)[0]
    print('Partitioned into blocks of length n_B=' + str(N_B))
    print('Number of partitions retained is n_P=' + str(N_P))

    # The frequency bins are thus defined:
    Freqs = np.fft.fftfreq(N_B) * F_S  # Compute the frequency bins
    Keep_IND = np.where(Freqs >= 0)
    N_B2 = len(Keep_IND[0])  # indexes for positive frequencies
    Freqs_Pos = Freqs[Keep_IND]  # positive frequencies

    # Step 2 : Construct the D_hats in each partition
    D_P_hat_Tens = np.zeros((N_S, N_B, N_P))
    print('Computing DFTs in each partition')
    for k in tqdm(range(0, N_P)):  # Loop over the partitions
        D_p = D[:, Indices[k]]  # Take the portion of data
        D_P_hat_Tens[:, :, k] = np.fft.fft(D_p, N_B, 1)

    # This would be the mean over the frequencies
    # D_hat_Mean=np.mean(D_P_hat_Tens,axis=1)

    # Initialize the outputs
    Sigma_SP = np.zeros((n_Modes, N_B2))
    Phi_SP = np.zeros((N_S, n_Modes, N_B2))

    # Step 3: Loop over frequencies to build the modes.
    # Note: you only care about half of these frequencies.
    # This is why you loop over N_B2, not N_B
    print('Computing POD for each frequency')
    for j in tqdm(range(0, N_B2)):
        # Get D_hat of the chunk
        D_hat_f = D_P_hat_Tens[:, j, :]
        # Go for the SVD
 
        U,V,Sigma=switch_svds(D_hat_f,n_Modes,svd_solver=possible_svds)

        Phi_SP[:, :, j] = U
        Sigma_SP[:, j] = Sigma / (N_S * N_B)

    if SAVE_SPOD:
        folder_dir = FOLDER_OUT + '/SPOD_T'
        os.makedirs(folder_dir, exist_ok=True)
        np.savez(folder_dir + '/spod_t.npz', Phi=Phi_SP, Sigma=Sigma_SP, Freqs=Freqs_Pos)

    return Phi_SP, Sigma_SP, Freqs_Pos
