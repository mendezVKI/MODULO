import numpy as np
import os
from tqdm import tqdm
import math


class ReadData:
    """
    A MODULO helper class for input data.  ReadData allows to load the data directly before using MODULO, and
    hence assembling the data matrix D from data.
    """


    def __init__(self):
        pass


    @classmethod
    def _data_processing(cls,D: np.array, FOLDER_OUT: str='./',
                   N_PARTITIONS: int = 1,
                   MR: bool = False, SAVE_D: bool = False,
                   FOLDER_IN: str = './', filename: str = '',
                   h: int = 0, f: int = 0, c: int = 0,
                   N: int = 0, N_S: int = 0, N_T: int = 0):
        """
        First, if the D matrix is not provided, this method attempts to load the data and assembles the D matrix.
        Then, it performs pre-processing operations on the data matrix, D. if MR=True, the mean (per each column - i.e.: snapshot at time t_i) is removed;
        If the MEMORY_SAVING=True the data matrix is splitted to optimize memory usage. Moreover, D is stored on disk and removed from the live memory.
        Finally, if in this condition, also the data type of the matrix is self is changed: from float64 -> float32, with the same purpose.

        :param D: np.array
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
        :param FOLDER_IN: str, optional. Needed only if database=None
             If the D matrix is not provided (database = None), read it from the path FOLDER_IN
        :param filename: str, optional. Needed only if database=None
             If the database is not provided, read it from the files filename
             The files must be named "filenamexxxx.dat" where x is the number of the file
             that goes from 0 to the number of time steps saved
        :param h: int, optional. Needed only if database=None
             Lines to be skipped from the header of filename
        :param f: int, optional. Needed only if database=None
             Lines to be skipped from the footer of filename
        :param c: int, optional. Needed only if database=None
             Columns to be skipped (for example if the first c columns contain the mesh grid.)
        :param N: int, optional. Needed only if database=None
             Components to be analysed.
        :param N_S:  int, optional. Needed only if database=None
             Number of points in space.
        :param N_T: int, optional. Needed only if database=None
             components to be analysed.

    
        :return:
             There are four possible scenario:
              1. if N_Partitions ==1 and MR = True, return is D,D_MEAN (the mean snapshot!)
              2. if N_Partitions ==1 and MR = False, return is D.
              3. if N_Partitions >1 and MR = True, return is D_MEAN
              4. if N_Partitions >1 and MR=False, return is None
        

        """
        
        if isinstance(D, np.ndarray):  # D was already initialised
            N_S = int(np.shape(D)[0])
            N_T = int(np.shape(D)[1])
            if MR:
                '''Removing mean from data matrix'''

                print("Removing the mean from D ...")
                D_MEAN = np.mean(D, 1)  # Temporal average (along the columns)
                D_Mr = D - np.array([D_MEAN, ] * N_T).transpose()  # Mean Removed
                print("Computing the mean-removed D ... ")
                np.copyto(D, D_Mr)                
                del D_Mr

            if N_PARTITIONS > 1:
                '''Converting D into float32, applying partitions and saving all.'''
                SAVE_D = True
                database = D.astype('float32', casting='same_kind')
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
                os.makedirs(FOLDER_OUT + "/data_matrix", exist_ok=True)
                print(f"Saving the matrix D in {FOLDER_OUT}")
                np.savez(FOLDER_OUT + '/data_matrix/database', D=D.astype('float32', casting='same_kind'), n_t=N_T, n_s=N_S)
        else:  # try to read the data
            print("Data matrix was not provided, reading it from {}".format(FOLDER_IN))
            # First check if the data were saved in the supported format
            try:
                Name = FOLDER_IN + os.sep + filename % (0 + 1) + '.dat'  # Name of the file to read
                # Read data from a file
                DATA = np.genfromtxt(Name, skip_header=h, skip_footer=f)  # Here we have the two colums
            except:
                raise AttributeError(
                    "FOLDER_IN {} does not exist or filename {} has not the good format. Check the help!".format(
                        FOLDER_IN, filename))

            if N_PARTITIONS == 1:  # If you have only one partition (one matrix! )
                D = np.zeros((N_S, N_T))

                print("\n \n Importing data with no partitions... \n \n")

                if MR:
                    print("Mean removal activated")
                    D_MEAN = np.zeros(N_S)

                for k in tqdm(range(0, N_T)):
                    Name = FOLDER_IN + os.sep + filename % (k + 1) + '.dat'  # Name of the file to read
                    # Read data from a file
                    DATA = np.genfromtxt(Name,  # usecols=np.arange(0, 2),
                                         skip_header=h, skip_footer=f)  # Here we have the two colums
                    # Dat = DATA[1:, :]  # Here we remove the first raw, containing the header
                    for ii in range(c, N + c):
                        tmp = DATA[:, ii]
                        if ii == c:
                            V = np.copy(tmp)
                        else:
                            V = np.concatenate([V, tmp], axis=0)
                    if MR:
                        D_MEAN += 1 / N_T * V  # Snapshot contribution to the mean

                    D[:, k] = V  # Reshape and assign

                if MR:
                    print("Removing the mean from D ...")
                    D_Mr = D - D_MEAN.reshape(-1, 1)  # Mean Removed
                    print("Computing the mean-removed D ... ")
                    np.copyto(D, D_Mr)
                    del D_Mr

            elif N_PARTITIONS > 1:  # then we enter in the memory saving loop
                # prepare the folder to store the parittions
                os.makedirs(FOLDER_OUT + "/data_partitions/", exist_ok=True)
                print("Memory Saving feature is active. Partitioning Data Matrix...")

                dim_col = math.floor(N_T / N_PARTITIONS)
                columns_to_part = dim_col * N_PARTITIONS  # These are integer multiples of N_PARTITIONS
                vec = np.arange(0, columns_to_part)
                # This gets the blocks
                splitted_tmp = np.hsplit(vec, N_PARTITIONS)
                if columns_to_part != N_T:
                    print("WARNING: the last " + str(
                        N_T - 1 - splitted_tmp[N_PARTITIONS - 1][-1]) + ' snapshots are not considered')

                if MR:
                    print("Mean removal activated")
                    D_MEAN = np.zeros(N_S)

                for ii in range(1, len(splitted_tmp) + 1):
                    count = 0
                    print('Working on block ' + str(ii) + '/' + str(N_PARTITIONS))
                    D = np.zeros((N_S, len(splitted_tmp[0])))
                    i1 = splitted_tmp[ii - 1][0];
                    i2 = splitted_tmp[ii - 1][-1]  # ranges
                    for k in tqdm(range(i1, i2 + 1)):
                        Name = FOLDER_IN + os.sep + filename % (k + 1) + '.dat'  # Name of the file to read
                        DATA = np.genfromtxt(Name,  # usecols=np.arange(0, 2),
                                             skip_header=h, skip_footer=f)  # Here we have the two colums
                        for nn in range(c, N + c):
                            tmp = DATA[:, nn]
                            if nn == c:
                                V = np.copy(tmp)
                            else:
                                V = np.concatenate([V, tmp], axis=0)

                        if MR:
                            D_MEAN += 1 / N_T * V  # Snapshot contribution to the mean

                        D[:, count] = V  # Reshape and assign
                        count += 1
                    np.savez(FOLDER_OUT + f"/data_partitions/di_{ii}", di=D)
                    print('Partition ' + str(ii) + '/' + str(N_PARTITIONS) + ' saved')

                if MR:
                    print('Reloading the data for removing the mean')
                    for ii in range(1, len(splitted_tmp) + 1):
                        print(f"Mean centering block {ii}")
                        di = np.load(FOLDER_OUT + f"/data_partitions/di_{ii}.npz")['di']
                        di_mr = di - D_MEAN.reshape(-1, 1)  # Mean Removed
                        np.savez(FOLDER_OUT + f"/data_partitions/di_{ii}", di=di_mr)
            else:
                raise TypeError("number of partitions not valid.")

        if (N_PARTITIONS ==1 and MR==True):
         return D, D_MEAN  
        elif (N_PARTITIONS ==1 and MR==False):
         return D
        elif (N_PARTITIONS >1 and MR==True):
         return D_MEAN
        else:
          return None
      
'''
    @classmethod
    def from_xls(cls, filename, **kwargs):
        """
        This class method builds the df from an excel file.

        work

        """
        ## TBD
        return

    @classmethod
    def _from_csv(cls, folder, filename, N, N_S,
                  h: int = 0, f: int = 0,
                  c: int = 0):
        """
        This method imports data (in the specified format) and then assemblies the corresponding
        data matrix, D.

        :param folder: str 
                Folder in which the data is stored
        :param filename: str
                Name of the files to be imported
        :param N number of components: int
                Components to be analysed
        :param h: int 
                Lines to be skipped from header
        :param f: int
                Lines to be skipped from footer
        :param c: int
                Columns to be skipped

        :return: np.array
                Assembled DataMarix

        """
        path, dirs, files = next(os.walk(folder))
        files = [f for f in files if f.endswith('.csv')]
        N_T = len(files)
        D = np.zeros((N_S, N_T))

        print("\n \n Importing data... \n \n")

        for k in tqdm(range(0, N_T)):
            Name = folder + files[k] #os.sep + filename % (k + 1) + '.csv'  # Name of the file to read
            # Read data from a file
            DATA = np.genfromtxt(Name,  # usecols=np.arange(0, 2),
                                 skip_header=h, skip_footer=f)  # Here we have the two colums
            # Dat = DATA[1:, :]  # Here we remove the first raw, containing the header
            for ii in range(c, N + c):
                tmp = DATA[:, ii]
                if ii == c:
                    V = np.copy(tmp)
                else:
                    V = np.concatenate([V, tmp], axis=0)

            D[:, k] = V  # Reshape and assign

        return D

    @classmethod
    def _from_txt(cls, folder, filename, N, N_S,
                  h: int = 0, f: int = 0,
                  c: int = 0):
        """
        This method imports data (in the specified format) and then assemblies the corresponding
        data matrix, D.

        :param folder: str 
                Folder in which the data is stored
        :param filename: str
                Name of the files to be imported
        :param N number of components: int
                Components to be analysed
        :param h: int 
                Lines to be skipped from header
        :param f: int
                Lines to be skipped from footer
        :param c: int
                Columns to be skipped

        :return: np.array
                Assembled DataMarix

        """
        path, dirs, files = next(os.walk(folder))
        N_T = len(files)
        D = np.zeros((N_S, N_T))

        print("\n \n Importing data... \n \n")

        for k in tqdm(range(0, N_T)):
            Name = folder + os.sep + filename % (k + 1) + '.txt'  # Name of the file to read
            # Read data from a file
            DATA = np.genfromtxt(Name,  # usecols=np.arange(0, 2),
                                 skip_header=h, skip_footer=f)  # Here we have the two colums
            # Dat = DATA[1:, :]  # Here we remove the first raw, containing the header
            for ii in range(c, N + c):
                tmp = DATA[:, ii]
                if ii == c:
                    V = np.copy(tmp)
                else:
                    V = np.concatenate([V, tmp], axis=0)

            D[:, k] = V  # Reshape and assign

        return D


'''


#%%

