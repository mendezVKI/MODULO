import numpy as np
import os
from tqdm import tqdm

class ReadData:
    """
    A MODULO helper class for input data.  ReadData allows to load the data directly before using MODULO, and
    hence assemblying the data matrix D from data, if needed.

    """


    def __init__(self):
        pass


    @classmethod
    def _from_dat(cls, folder, filename, N, N_S,
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
            Name = folder + os.sep + filename % (k + 1) + '.dat'  # Name of the file to read
            # Read data from a file
            DATA = np.genfromtxt(Name, #usecols=np.arange(0, 2),
                                 skip_header=h, skip_footer=f)  # Here we have the two colums
            #Dat = DATA[1:, :]  # Here we remove the first raw, containing the header
            for ii in range(c, N + c):
                tmp = DATA[:, ii]
                if ii == c:
                    V = np.copy(tmp)
                else:
                    V = np.concatenate([V, tmp], axis = 0)

            D[:, k] = V  # Reshape and assign

        return D


    @classmethod
    def from_xls(cls, filename, **kwargs):
        """
        This class method builds the df from an excel file.
        :param filename: str
                filename (with path if needed) to the df file.
        :return: constructor for the class.

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
        N_T = len(files)
        D = np.zeros((N_S, N_T))

        print("\n \n Importing data... \n \n")

        for k in tqdm(range(0, N_T)):
            Name = folder + os.sep + filename % (k + 1) + '.csv'  # Name of the file to read
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




