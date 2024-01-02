import os

import numpy as np
from tqdm import tqdm


def dft_fit(N_T, F_S, D, FOLDER_OUT, SAVE_DFT=False):
    """
    :param N_T:
    :param F_S:
    :param D:
    :param FOLDER_OUT:
    :param SAVE_DFT:
    :return:
    """
    n_t = int(N_T)
    Freqs = np.fft.fftfreq(n_t) * F_S  # Compute the frequency bins
    # PSI_F = np.conj(np.fft.fft(np.eye(n_t)) / np.sqrt(n_t))  # Prepare the Fourier Matrix.

    # Method 1 (didactic!)
    # PHI_SIGMA = np.dot(D, np.conj(PSI_F))  # This is PHI * SIGMA

    # Method 2
    PHI_SIGMA = (np.fft.fft(D, n_t, 1)) / (n_t ** 0.5)

    PHI_F = np.zeros((D.shape[0], n_t), dtype=complex)  # Initialize the PHI_F MATRIX
    SIGMA_F = np.zeros(n_t)  # Initialize the SIGMA_F MATRIX

    # Now we proceed with the normalization. This is also intense so we time it
    for r in tqdm(range(0, n_t)):  # Loop over the PHI_SIGMA to normalize
        # MEX = 'Proj ' + str(r + 1) + ' /' + str(n_t)
        # print(MEX)
        SIGMA_F[r] = abs(np.vdot(PHI_SIGMA[:, r], PHI_SIGMA[:, r])) ** 0.5
        PHI_F[:, r] = PHI_SIGMA[:, r] / SIGMA_F[r]

    Indices = np.flipud(np.argsort(SIGMA_F))  # find indices for sorting in decreasing order
    Sorted_Sigmas = SIGMA_F[Indices]  # Sort all the sigmas
    Sorted_Freqs = Freqs[Indices]  # Sort all the frequencies accordingly.
    Phi_F = PHI_F[:, Indices]  # Sorted Spatial Structures Matrix
    SIGMA_F = Sorted_Sigmas  # Sorted Amplitude Matrix (vector)

    if SAVE_DFT:
        os.makedirs(FOLDER_OUT + 'DFT', exist_ok=True)
        np.savez(FOLDER_OUT + 'DFT/dft_fitted', Freqs=Sorted_Freqs, Phis=Phi_F, Sigmas=SIGMA_F)

    return Sorted_Freqs, Phi_F, SIGMA_F
