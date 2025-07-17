import numpy as np
import os
from tqdm import tqdm
import math


def spatial_basis_mPOD(D, PSI_M, N_T, N_PARTITIONS, N_S, MEMORY_SAVING, FOLDER_OUT, SAVE: bool = False,weights: np.array = np.array([])):
    """
    Given the temporal basis of the mPOD now the spatial ones are computed
    
    :param D: 
        Snapshot matrix D: if memory savig is active, this is ignored.    
    :param PSI_M: np.array.: 
        The mPOD temporal basis Psi tentatively assembled from all scales
    :param N_T: int. 
        Number of snapshots
    :param N_PARTITIONS: int. 
        Number of partitions in the memory saving
    :param N_S: int. 
        Number of grid points in space
    :param MEMORY_SAVING: bool. 
        Inherited from main class, if True turns on the MEMORY_SAVING feature, loading the partitions and starting the proper algorithm  
    :param FOLDER_OUT: str. 
        Folder in which the results are saved if SAVE_SPATIAL_POD = True
    :param SAVE_SPATIAL_POD: bool.
        If True, results are saved on disk and released from memory
    :param weights:  np.array
        weight vector [w_i,....,w_{N_s}] where w_i = area_cell_i/area_grid. Only needed if grid is non-uniform & MEMORY_SAVING== True
    :return: Phi_M, Psi_M, Sigma_M: np.arrays. The final (sorted) mPOD decomposition
    """
    
    R1 = 0; R2 = 0
    if MEMORY_SAVING:
        SAVE = True
        os.makedirs(FOLDER_OUT + '/mPOD/', exist_ok=True)
        dim_col = math.floor(N_T / N_PARTITIONS)
        dim_row = math.floor(N_S / N_PARTITIONS)
        dr = np.zeros((dim_row, N_T))

        # 1 --- Converting partitions dC to dR
        if N_S % N_PARTITIONS != 0:
            tot_blocks_row = N_PARTITIONS + 1
        else:
            tot_blocks_row = N_PARTITIONS

        if N_T % N_PARTITIONS != 0:
            tot_blocks_col = N_PARTITIONS + 1
        else:
            tot_blocks_col = N_PARTITIONS

        fixed = 0

        for i in range(1, tot_blocks_row + 1):
            # --- Check if dim_row has to be fixed:
            if i == tot_blocks_row and (N_S - dim_row * N_PARTITIONS > 0):
                dim_row_fix = N_S - dim_row * N_PARTITIONS
                dr = np.zeros((dim_row_fix, N_T))

            for cont in range(1, tot_blocks_col + 1):
                di = np.load(FOLDER_OUT + f"/data_partitions/di_{cont}.npz")['di']

                if i == tot_blocks_row and (N_S - dim_row * N_PARTITIONS > 0) and fixed == 0:
                    R1 = R2
                    R2 = R1 + (N_S - dim_row * N_PARTITIONS)
                    fixed = 1
                elif fixed == 0:
                    R1 = (i - 1) * dim_row
                    R2 = i * dim_row

                # Same as before, but we don't need the variable fixed because if
                #         % the code runs this loop, it will be the last time

                if cont == tot_blocks_col and (N_T - dim_col * N_PARTITIONS > 0):
                    C1 = C2
                    C2 = C1 + (N_T - dim_col * N_PARTITIONS)
                else:
                    C1 = (cont - 1) * dim_col
                    C2 = cont * dim_col

                dr[:, C1:C2] = di[R1:R2, :]

            # 2 --- Computing partitions R of PHI_SIGMA
            PHI_SIGMA_BLOCK = dr @ PSI_M
            np.savez(FOLDER_OUT + f'/mPOD/phi_sigma_{i}', PHI_SIGMA_BLOCK)

        # 3 --- Convert partitions R to partitions C and get SIGMA
        R = PSI_M.shape[1]
        dim_col = math.floor(R / N_PARTITIONS)
        dim_row = math.floor(N_S / N_PARTITIONS)
        dps = np.zeros((N_S, dim_col))
        SIGMA_M = []
        PHI_M = []

        if R % N_PARTITIONS != 0:
            tot_blocks_col = N_PARTITIONS + 1
        else:
            tot_blocks_col = N_PARTITIONS

        fixed = 0

        # Here we apply the same logic of the loop before

        for j in range(1, tot_blocks_col + 1):

            if j == tot_blocks_col and (R - dim_col * N_PARTITIONS > 0):
                dim_col_fix = R - dim_col * N_PARTITIONS
                dps = np.zeros((N_S, dim_col_fix))

            for k in range(1, tot_blocks_row + 1):
                PHI_SIGMA_BLOCK = np.load(FOLDER_OUT + f"/mPOD/phi_sigma_{k}.npz")['arr_0']

                if j == tot_blocks_col and (R - dim_col * N_PARTITIONS > 0) and fixed == 0:
                    R1 = R2
                    R2 = R1 + (R - dim_col * N_PARTITIONS)
                    fixed = 1
                elif fixed == 0:
                    R1 = (j - 1) * dim_col
                    R2 = j * dim_col

                if k == tot_blocks_row and (N_S - dim_row * N_PARTITIONS > 0):
                    C1 = C2
                    C2 = C1 + (N_S - dim_row * N_PARTITIONS)
                else:
                    C1 = (k - 1) * dim_row
                    C2 = k * dim_row

                dps[C1:C2, :] = PHI_SIGMA_BLOCK[:, R1:R2]

            # Getting sigmas and phis
            for z in range(R1, R2):
                zz = z - R1
                if weights.size == 0:
                    SIGMA_M.append(np.linalg.norm(dps[:, zz]))
                else:
                    SIGMA_M.append(np.linalg.norm(dps[:, zz]*np.sqrt(weights)))
                tmp = dps[:, zz] / SIGMA_M[z]
                #print(f'Shape tmp = {np.shape(tmp)}')
                PHI_M.append(tmp)
                np.savez(FOLDER_OUT + f'/mPOD/phi_{z + 1}', tmp)

        Indices = np.argsort(SIGMA_M)[::-1]  # find indices for sorting in decreasing order
        SIGMA_M = np.asarray(SIGMA_M)
        PHI_M = np.asarray(PHI_M).T
        PSI_M = np.asarray(PSI_M)
        Sorted_Sigmas = SIGMA_M[Indices]  # Sort all the sigmas
        Phi_M = PHI_M[:, Indices]  # Sorted Spatial Structures Matrix
        Psi_M = PSI_M[:, Indices]  # Sorted Temporal Structures Matrix
        Sigma_M = Sorted_Sigmas  # Sorted Amplitude Matrix

    else:
        R = PSI_M.shape[1]
        PHI_M_SIGMA_M = np.dot(D, (PSI_M))
        # Initialize the output
        PHI_M = np.zeros((N_S, R))
        SIGMA_M = np.zeros((R))

        for i in tqdm(range(0, R)):
            # print('Completing mPOD Mode ' + str(i))
            # Assign the norm as amplitude
            if weights.size == 0:
                SIGMA_M[i] = np.linalg.norm(PHI_M_SIGMA_M[:, i])
            else:
                SIGMA_M[i] = np.linalg.norm(PHI_M_SIGMA_M[:, i]*np.sqrt(weights))
            # Normalize the columns of C to get spatial modes
            PHI_M[:, i] = PHI_M_SIGMA_M[:, i] / SIGMA_M[i]

        Indices = np.flipud(np.argsort(SIGMA_M))  # find indices for sorting in decreasing order
        Sorted_Sigmas = SIGMA_M[Indices]  # Sort all the sigmas
        Phi_M = PHI_M[:, Indices]  # Sorted Spatial Structures Matrix
        Psi_M = PSI_M[:, Indices]  # Sorted Temporal Structures Matrix
        Sigma_M = Sorted_Sigmas  # Sorted Amplitude Matrix

    if SAVE:
        '''Saving results in MODULO tmp proper folder'''
        os.makedirs(FOLDER_OUT + '/mPOD/', exist_ok=True)
        np.savez(FOLDER_OUT + "/mPOD/sorted_phis", Phi_M)
        np.savez(FOLDER_OUT + "/mPOD/sorted_psis", Psi_M)
        np.savez(FOLDER_OUT + "/mPOD/sorted_sigma", Sorted_Sigmas)

    return Phi_M, Psi_M, Sigma_M
