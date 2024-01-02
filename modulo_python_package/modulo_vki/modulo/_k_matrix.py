import os
from tqdm import tqdm
import numpy as np
import math


def CorrelationMatrix(N_T, N_PARTITIONS=1, MEMORY_SAVING=False, FOLDER_OUT='./', SAVE_K=True, D=None):
    """
    This method computes the temporal correlation matrix, given a data matrix as input. It's possible to use memory saving
    then splitting the computing in different tranches if computationally heavy. If D has been computed using MODULO
    then the dimension dim_col and N_PARTITIONS is automatically loaded
    --------------------------------------------------------------------------------------------------------------------
    Parameters
    ----------

    :param N_T: int
                Number of temporal snapshots
    :param D: np.array
                Data matrix
    :param SAVE_K: bool
                If SAVE_K=True, the matrix K is saved on disk. If the MEMORY_SAVING feature is active, this is done
                by default.
    :param MEMORY_SAVING: bool
                If MEMORY_SAVING = True, the computation of the correlation matrix is done by steps. It requires the
                data matrix to be partitioned, following algorithm in MODULO._data_processing.
    :param FOLDER_OUT: str
                Folder in which the temporal correlation matrix will be stored
    :param N_PARTITIONS: int
                Number of partitions to be read in computing the correlation matrix. If _data_processing is used to
                partition the data matrix, this is inherited from the main class
    --------------------------------------------------------------------------------------------------------------------
    Returns
    -------

    :return: K (: np.array) if the memory saving is not active. None type otherwise.
    """

    if not MEMORY_SAVING:
        print("\n Computing Temporal correlation matrix K ...")
        K = np.dot(D.T, D)
        print("\n Done.")

    else:
        SAVE_K = True
        print("\n Using Memory Saving feature...")
        K = np.zeros((N_T, N_T))
        dim_col = math.floor(N_T / N_PARTITIONS)

        if N_T % N_PARTITIONS != 0:
            tot_blocks_col = N_PARTITIONS + 1
        else:
            tot_blocks_col = N_PARTITIONS

        for k in tqdm(range(tot_blocks_col)):

            di = np.load(FOLDER_OUT + f"/data_partitions/di_{k + 1}.npz")['di']

            ind_start = k * dim_col
            ind_end = ind_start + dim_col

            if (k == tot_blocks_col - 1) and (N_T - dim_col * N_PARTITIONS > 0):
                dim_col = N_T - dim_col * N_PARTITIONS
                ind_end = ind_start + dim_col

            K[ind_start:ind_end, ind_start:ind_end] = np.dot(di.transpose(), di)

            block = k + 2

            while block <= tot_blocks_col:
                dj = np.load(FOLDER_OUT + f"/data_partitions/di_{block}.npz")['di']

                ind_start_out = (block - 1) * dim_col
                ind_end_out = ind_start_out + dim_col

                if (block == tot_blocks_col) and (N_T - dim_col * N_PARTITIONS > 0):
                    dim_col = N_T - dim_col * N_PARTITIONS
                    ind_end_out = ind_start_out + dim_col
                    dj = dj[:, :dim_col]

                K[ind_start:ind_end, ind_start_out:ind_end_out] = np.dot(di.transpose(), dj)

                K[ind_start_out:ind_end_out, ind_start:ind_end] = K[ind_start:ind_end, ind_start_out:ind_end_out].T

                block += 1

                dim_col = math.floor(N_T / N_PARTITIONS)

    if SAVE_K:
        os.makedirs(FOLDER_OUT + '/correlation_matrix', exist_ok=True)
        np.savez(FOLDER_OUT + "/correlation_matrix/k_matrix", K=K)

    return K if not MEMORY_SAVING else None
