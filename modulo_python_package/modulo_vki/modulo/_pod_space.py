import math
import os

import numpy as np
from tqdm import tqdm


def Spatial_basis_POD(D, PSI_P, Sigma_P, MEMORY_SAVING,
                      N_T, FOLDER_OUT='./', N_PARTITIONS=1,
                      SAVE_SPATIAL_POD=False):
    """
    Given the temporal basis  now the POD spatial ones are computed
    --------------------------------------------------------------------------------------------------------------------
    Parameters:
    ----------
    :param N_T: int
        Number of temporal snapshots

    :param FOLDER_OUT: str
        Folder in which the results are saved if SAVE_SPATIAL_POD = True

    :param SAVE_SPATIAL_POD: bool
        If True, results are saved on disk and released from memory

    :param N_PARTITIONS: int
        Number of partitions to be loaded. If D has been partitioned using MODULO, this parameter is automatically
        inherited from the main class. To be specified otherwise.

    :param MEMORY_SAVING: bool
        Inherited from main class, if True turns on the MEMORY_SAVING feature, loading the partitions and starting
        the proper algorithm

    :param D: np.array
        Data matrix on which project the temporal basis

    :param PSI_P: np.array
        POD Psis
    :param Sigma_P: np.array
        POD Sigmas
    --------------------------------------------------------------------------------------------------------------------
    Returns:
    --------

    :return Phi_P: np.array
        POD Phis
    """
    R = PSI_P.shape[1]

    if not MEMORY_SAVING:
        N_S = D.shape[0]
        Phi_P = np.zeros((N_S, R))

        # N_S = D.shape[0]
        PHI_P_SIGMA_P = np.dot(D, PSI_P)

        print("Completing POD Modes: \n")

        for i in tqdm(range(0, R)):
            # Normalize the columns of C to get spatial modes
            Phi_P[:, i] = PHI_P_SIGMA_P[:, i] / Sigma_P[i]
        
        return Phi_P  
    
        if SAVE_SPATIAL_POD:
            os.makedirs(FOLDER_OUT + 'POD', exist_ok=True)
            np.savez(FOLDER_OUT + '/POD/pod_spatial_basis',
                     phis=Phi_P, PHI_P_SIGMA_P=PHI_P_SIGMA_P)


    else:
        N_S = np.shape(np.load(FOLDER_OUT + "/data_partitions/di_1.npz")['di'])[0]

        dim_col = math.floor(N_T / N_PARTITIONS)
        dim_row = math.floor(N_S / N_PARTITIONS)
        dr = np.zeros((dim_row, N_T))

        # 1 -- Converting partitions dC to dR
        if N_S % N_PARTITIONS != 0:
            tot_blocks_row = N_PARTITIONS + 1
        else:
            tot_blocks_row = N_PARTITIONS

        if N_T % N_PARTITIONS != 0:
            tot_blocks_col = N_PARTITIONS + 1
        else:
            tot_blocks_col = N_PARTITIONS

        # --- Loading Psi_P
        fixed = 0
        R1 = 0
        R2 = 0
        C1 = 0
        C2 = 0

        for i in range(1, tot_blocks_row + 1):

            if (i == tot_blocks_row) and (N_S - dim_row * N_PARTITIONS > 0):
                dim_row_fix = N_S - dim_row * N_PARTITIONS
                dr = np.zeros((dim_row_fix, N_T))

            for b in range(1, tot_blocks_col + 1):
                di = np.load(FOLDER_OUT + f"/data_partitions/di_{b}.npz")['di']
                if (i == tot_blocks_row) and (N_S - dim_row * N_PARTITIONS > 0) and fixed == 0:
                    R1 = R2
                    R2 = R1 + (N_S - dim_row * N_PARTITIONS)
                    fixed = 1
                elif fixed == 0:
                    R1 = (i - 1) * dim_row
                    R2 = i * dim_row

                if (b == tot_blocks_col) and (N_T - dim_col * N_PARTITIONS > 0):
                    C1 = C2
                    C2 = C1 + (N_T - dim_col * N_PARTITIONS)
                else:
                    C1 = (b - 1) * dim_col
                    C2 = b * dim_col

                np.copyto(dr[:, C1:C2], di[R1:R2, :])

            PHI_SIGMA_BLOCK = np.dot(dr, PSI_P)
            np.savez(FOLDER_OUT + f"/POD/PHI_SIGMA_{i}",
                     phi_sigma=PHI_SIGMA_BLOCK)

        # 3 - Converting partitions R to partitions C and get Sigmas
        dim_col = math.floor(R / N_PARTITIONS)
        dim_row = math.floor(N_S / N_PARTITIONS)
        dps = np.zeros((N_S, dim_col))
        Phi_P = np.zeros((N_S, R))

        if R % N_PARTITIONS != 0:
            tot_blocks_col = N_PARTITIONS + 1
        else:
            tot_blocks_col = N_PARTITIONS

        fixed = 0

        for i in range(1, tot_blocks_col + 1):

            if (i == tot_blocks_col) and (R - dim_col * N_PARTITIONS > 0):
                dim_col_fix = R - dim_col * N_PARTITIONS
                dps = np.zeros((N_S, dim_col_fix))

            for b in range(1, tot_blocks_row + 1):

                PHI_SIGMA_BLOCK = np.load(FOLDER_OUT + f"POD/PHI_SIGMA_{b}.npz")['phi_sigma']

                if (i == tot_blocks_col) and (R - dim_col * N_PARTITIONS > 0) and fixed == 0:
                    R1 = R2
                    R2 = R1 + (R - dim_col * N_PARTITIONS)
                    fixed = 1
                elif fixed == 0:
                    R1 = (i - 1) * dim_col
                    R2 = i * dim_col

                if (b == tot_blocks_col) and (N_S - dim_row * N_PARTITIONS > 0):
                    C1 = C2
                    C2 = C1 + (N_S - dim_row * N_PARTITIONS)
                else:
                    C1 = (b - 1) * dim_row
                    C2 = b * dim_row

                dps[C1:C2, :] = PHI_SIGMA_BLOCK[:, R1:R2]

                # Computing Sigmas and Phis:
                # TODO: np.linalg.norm(dps[:, jj]) to substitute with Sigma (that is already available from _pod_time)
            for j in range(R1, R2):
                jj = j - R1
                Phi_P = dps[:, jj] / np.linalg.norm(dps[:, jj])
                np.savez(FOLDER_OUT + f"POD/phi_{j + 1}", phi_p=Phi_P)

        # Read the temporary files to build Phi_P_Matrix (Lorenzo pls fix this :D)
        # TODO 
        Phi_P_M=np.zeros((N_S,R))
        for j in range(R):
         Phi_P_V=np.load(FOLDER_OUT + f"POD/phi_{j + 1}.npz")['phi_p']   
         Phi_P_M[:,j]=Phi_P_V    
    
        return Phi_P_M
