import math
import numpy as np
from tqdm import tqdm
import os


def Spatial_basis_POD(D, PSI_P, Sigma_P, MEMORY_SAVING, 
                      N_T, FOLDER_OUT='./', N_PARTITIONS=1, SAVE_SPATIAL_POD=False,
                      rescale=False, verbose=True):
    """
    This function computs the POD spatial basis from the temporal basis,

    :param D: np.array. 
         matrix on which to project the temporal basis
    :param PSI_P: np.array. 
          POD's Psis
    :param Sigma_P: np.array. 
          POD's Sigmas
    :param MEMORY_SAVING: bool. 
         Inherited from main class, if True turns on the MEMORY_SAVING feature, loading the partitions and starting the proper algorithm          
    :param N_T: int. 
         Number of temporal snapshots
    :param FOLDER_OUT: str. 
         Folder in which the results are saved if SAVE_SPATIAL_POD = True
    :param N_PARTITIONS: int. 
         Number of partitions to be loaded. If D has been partitioned using MODULO, this parameter is automatically inherited from the main class. To be specified otherwise.

    :param SAVE_SPATIAL_POD: bool. 
         If True, results are saved on disk and released from memory

    :param rescale: bool. 
          If False, the Sigmas are used for the normalization. If True, these are ignored and the normalization is carried out.
          For the standard POD, False is the way to go. 
          However, for other decompositions (eg. the SPOD_s) you must use rescale=True                        

    :return Phi_P: np.array. 
          POD's Phis
    """

    R = PSI_P.shape[1]

    if not MEMORY_SAVING:
        N_S = D.shape[0]

        if rescale:
            # The following is the general normalization approach.
            # not needed for POD for required for SPOD
            Phi_P = np.zeros((N_S, R))
            # N_S = D.shape[0] unused variable
            PHI_P_SIGMA_P = np.dot(D, PSI_P)
            if verbose:
                print("Completing Spatial Structures Modes: \n")

            for i in tqdm(range(0, R)):
                # Normalize the columns of C to get spatial modes
                Phi_P[:, i] = PHI_P_SIGMA_P[:, i] / Sigma_P[i]

        else:
            # We take only the first R modes.
            Sigma_P_t = Sigma_P[0:R]
            Sigma_P_Inv_V = 1 / Sigma_P_t
            # So we have the inverse
            Sigma_P_Inv = np.diag(Sigma_P_Inv_V)
            # Here is the one shot projection:
            Phi_P = np.linalg.multi_dot([D, PSI_P[:, 0:R], Sigma_P_Inv])

            if SAVE_SPATIAL_POD:
                os.makedirs(FOLDER_OUT + 'POD', exist_ok=True)
                np.savez(FOLDER_OUT + '/POD/pod_spatial_basis', phis=Phi_P)
                # removed PHI_P_SIGMA_P=PHI_P_SIGMA_P, not present if not rescale and not needed (?)

        return Phi_P

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
            np.savez(FOLDER_OUT + f"/PHI_SIGMA_{i}",
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

                PHI_SIGMA_BLOCK = np.load(FOLDER_OUT + f"/PHI_SIGMA_{b}.npz")['phi_sigma']

                if (i == tot_blocks_col) and (R - dim_col * N_PARTITIONS > 0) and fixed == 0:
                    R1 = R2
                    R2 = R1 + (R - dim_col * N_PARTITIONS)
                    fixed = 1
                elif fixed == 0:
                    R1 = (i - 1) * dim_col
                    R2 = i * dim_col

                if (b == tot_blocks_row) and (N_S - dim_row * N_PARTITIONS > 0): # Change here
                    C1 = C2
                    C2 = C1 + (N_S - dim_row * N_PARTITIONS)
                else:
                    C1 = (b - 1) * dim_row
                    C2 = b * dim_row

                dps[C1:C2, :] = PHI_SIGMA_BLOCK[:, R1:R2]

            # Computing Sigmas and Phis
            if rescale:
                for j in range(R1, R2):
                    jj = j - R1
                    Sigma_P[jj] = np.linalg.norm(dps[:, jj])
                    Phi_P = dps[:, jj] / Sigma_P[jj]
                    np.savez(FOLDER_OUT + f"/phi_{j + 1}", phi_p=Phi_P)
            else:
                for j in range(R1, R2):
                    jj = j - R1
                    Phi_P = dps[:, jj] / Sigma_P[j] # Change here
                    np.savez(FOLDER_OUT + f"/phi_{j + 1}", phi_p=Phi_P)

        Phi_P_M = np.zeros((N_S, R))
        for j in range(R):
            Phi_P_V = np.load(FOLDER_OUT + f"/phi_{j + 1}.npz")['phi_p']
            Phi_P_M[:, j] = Phi_P_V

        return Phi_P_M


def spatial_basis_mPOD(D, PSI_M, N_T, N_PARTITIONS, N_S, MEMORY_SAVING, FOLDER_OUT, SAVE: bool = False,weights: np.array = np.array([]), SIGMA_TYPE: str = "accurate", SIGMA_M: np.array = np.array([])):
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
    :param Sigma_type : {'accurate', 'fast'}
        If accurate, recompute the Sigmas after QR polishing. Slightly slower than the fast option in which the Sigmas are not recomputed.
    :param SIGMA_M: np.array.:
        The mPOD Sigmas before the QR polishing, tentatively assembled from all scales
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

        # Re-compute the sigma after the qr polishing
        SIGMA_TYPE = SIGMA_TYPE.lower()
        if SIGMA_TYPE != 'fast':
            if SIGMA_TYPE != 'accurate':
                print('Warning: MODULO continues to run although SIGMA_TYPE was wrongly defined. Please set it to \'accurate\' or \'fast\' ')
            if weights.size == 0:
                SIGMA_M = np.linalg.norm(PHI_M_SIGMA_M, axis=0)
            else:
                SIGMA_M = np.linalg.norm(PHI_M_SIGMA_M*np.sqrt(weights), axis=0)


        # Normalize the columns of C to get spatial modes
        PHI_M = PHI_M_SIGMA_M / SIGMA_M

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
