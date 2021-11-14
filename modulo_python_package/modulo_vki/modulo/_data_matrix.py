import math
import os
import numpy as np


def DataMatrix(database: np.array, FOLDER_OUT: str,
               MEMORY_SAVING: bool = False, N_PARTITIONS: int =1,
               MR: bool = False, SAVE_D: bool = False):
    """
    This method performs pre-processing operations on the data matrix, D:
    - Mean removing: if MR=True, the mean (per each column - i.e.: snapshot at time t_i) is removed;
    - Splitting: if the MEMORY_SAVING=True the data matrix is splitted to optimize memory usage. Moreover, D is dumped
                on disk and removed from the live memory. Finally, if in this condition, also the data type of the
                matrix is self is changed: from float64 -> float32, with the same purpose.

    --------------------------------------------------------------------------------------------------------------------
    Parameters
    ----------

    :param database: np.array
                        data matrix D
    :param FOLDER_OUT: str
                        folder in which the data (partitions and/or data matrix itself) will be eventually saved.
    :param MEMORY_SAVING: bool, optional
                        If True, memory saving feature is activated. Passed through __init__
    :param N_PARTITIONS: int
                        In memory saving environment, this parameter refers to the number of partitions to be applied
                        to the data matrix. If the number indicated by the user is not a multiple of the N_T
                        i.e.: if (N_T % N_PARTITIONS) !=0 - then an additional partition is introduced, that contains
                        the remaining columns
    :param MR: bool, optional
                        If True, it removes the mean (per column) from each snapshot
    :param SAVE_D: bool, optional
                    If True, the matrix D is saved into memory. If the Memory Saving feature is active, this is performed
                    by default.

    Returns
    -------
    :return: D if memory saving feature is active. Otherwise, D matrix is saved on disk and this method returns None
    """
    N_S = int(np.shape(database)[0])
    N_T = int(np.shape(database)[1])

    if MR:
        '''Removing mean from data matrix'''

        print("Removing the mean from D ...")
        D_MEAN = np.mean(database, 1)  # Temporal average (along the columns)
        D_Mr = database - np.array([D_MEAN, ] * N_T).transpose()  # Mean Removed
        print("Computing the mean-removed D ... ")
        np.copyto(database, D_Mr)
        del D_Mr

    if MEMORY_SAVING:
        '''Converting D into float32, applying partitions and saving all.'''
        SAVE_D = True
        database = database.astype('float32', casting='same_kind')
        os.makedirs(FOLDER_OUT + "/data_partitions/", exist_ok=True)
        print("Memory Saving feature is active. Partitioning Data Matrix...")
        if N_T % N_PARTITIONS != 0:
            dim_col = math.floor(N_T / N_PARTITIONS)

            columns_to_part = dim_col * N_PARTITIONS
            splitted_tmp = np.hsplit(database[:, :columns_to_part], N_PARTITIONS)
            for ii in range(1, len(splitted_tmp) + 1):
                np.savez(FOLDER_OUT + f"/data_partitions/di_{ii}", di=splitted_tmp[ii - 1])

            np.savez(FOLDER_OUT + f"/data_partitions/di_{N_PARTITIONS + 1}",
                     di=database[:, columns_to_part:])
        else:
            splitted_tmp = np.hsplit(database, N_PARTITIONS)
            for ii in range(1, len(splitted_tmp) + 1):
                np.savez(FOLDER_OUT + f"/data_partitions/di_{ii}", di=splitted_tmp[ii - 1])

        print("\n Data Matrix has been successfully splitted. \n")


    if SAVE_D:
        '''Saving data matrix in FOLDER_OUT'''

        os.makedirs(FOLDER_OUT + "./data_matrix", exist_ok=True)
        print(f"Saving the matrix D in {FOLDER_OUT}")
        np.savez(FOLDER_OUT + 'data_matrix/database', D=database, n_t=N_T, n_s=N_S)

    return database if not MEMORY_SAVING else None
