import numpy as np
import os
from tqdm import tqdm
import math


class ReadData:
    """
    A MODULO helper class for input data.  ReadData allows to load the data directly before using MODULO, and
    hence assemblying the data matrix D from data, if needed.

    """


    def __init__(self):
        pass


    @classmethod
    def _from_dat(cls, folder, filename, N, N_S, N_T,
                  h: int = 0, f: int = 0,
                  c: int = 0, N_PARTITIONS: int =1,MR: bool = False):
        """
        This method imports data (in the specified format) and then assemblies the corresponding
        data matrix, D.

        :param folder: str 
                Folder in which the data is stored
        :param filename: str
                Name of the files to be imported
        :param N: int
                Components to be analysed
        :param N_S:  int
                Number of points in space       
        :param N_T: int
                Components to be analysed      
        :param h: int 
                Lines to be skipped from header
        :param f: int
                Lines to be skipped from footer
        :param c: int
                Columns to be skipped (for example if the first c columns
                                       contain the mesh grid.)
        :param N_PARTITIONS: int
               Number of partitions. if =1 , then the matrix will be built
               and returned in output. if >1, then it means that the memory
               saving is active. The code will break the data into blocks
               and stored in a local directory 

        :return: np.array
                Assembled DataMarix

        """
        path, dirs, files = next(os.walk(folder))
        #N_T = len(files) # this is dangerous because the folder could contain
        # also files that are related to the mesh. We give N_T as input!


        if N_PARTITIONS==1: # If you have only one partition (one matrix! )
         D = np.zeros((N_S, N_T))
        
         print("\n \n Importing data with no partitions... \n \n")
     
         if MR:
           print("Mean removal activated")  
           D_MEAN=np.zeros(N_S)


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
            if MR:
             D_MEAN += 1/N_T*V # Snapshot contribution to the mean
           
            D[:, k] = V  # Reshape and assign
        
         if MR:
          print("Removing the mean from D ...")
          D_Mr = D - D_MEAN.reshape(-1,1)  # Mean Removed
          print("Computing the mean-removed D ... ")
          np.copyto(D, D_Mr)
          del D_Mr
        
        
        elif N_PARTITIONS>1: # then we enter in the memory saving loop
          # prepare the folder to store the parittions
          os.makedirs("MODULO_tmp/data_partitions/", exist_ok=True)
          print("Memory Saving feature is active. Partitioning Data Matrix...")
       
          dim_col = math.floor(N_T / N_PARTITIONS)
          columns_to_part = dim_col * N_PARTITIONS # These are integer multiples of N_PARTITIONS
          vec=np.arange(0,columns_to_part)                          
          # This gets the blocks 
          splitted_tmp = np.hsplit(vec, N_PARTITIONS)
          if columns_to_part != N_T :
           print("WARNING: the last "+str(N_T-1-splitted_tmp[N_PARTITIONS-1][-1])+' snapshots are not considered')
          
          if MR:
            print("Mean removal activated")  
            D_MEAN=np.zeros(N_S)
          
          for ii in range(1,len(splitted_tmp)+1):
              count=0   
              print('Working on block '+str(ii)+'/'+str(N_PARTITIONS)) 
              D=np.zeros((N_S,len(splitted_tmp[0])))
              i1=splitted_tmp[ii-1][0]; i2=splitted_tmp[ii-1][-1] # ranges
              for k in tqdm(range(i1,i2+1)):                  
                 Name = folder + os.sep + filename % (k + 1) + '.dat'  # Name of the file to read
                 DATA = np.genfromtxt(Name, #usecols=np.arange(0, 2),
                                      skip_header=h, skip_footer=f)  # Here we have the two colums
                 for nn in range(c, N + c):
                     tmp = DATA[:, nn]
                     if nn == c:
                         V = np.copy(tmp)
                     else:
                         V = np.concatenate([V, tmp], axis = 0)
                 
                 if MR:
                  D_MEAN += 1/N_T*V # Snapshot contribution to the mean
                
            
                 D[:, count] = V  # Reshape and assign
                 count+=1
              np.savez(f"MODULO_tmp/data_partitions/di_{ii}", di=D)            
              print('Partition '+str(ii)+'/'+str(N_PARTITIONS)+' saved')    
                
          if MR:
            print('Reloading the data for removing the mean')  
            for ii in range(1,len(splitted_tmp)+1):
              print(f"Mean centering block {ii}")  
              di = np.load(f"MODULO_tmp/data_partitions/di_{ii}.npz")['di']
              di_mr=di - D_MEAN.reshape(-1,1)  # Mean Removed
              np.savez(f"MODULO_tmp/data_partitions/di_{ii}", di=di_mr)         
        else: 
            raise TypeError("number of partitions not valid.") 
             
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




