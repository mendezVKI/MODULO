import os

import numpy as np

from modulo._data_matrix import DataMatrix
from modulo._k_matrix import CorrelationMatrix
from modulo._mpod_space import spatial_basis_mPOD
from modulo._mpod_time import temporal_basis_mPOD
from modulo._pod_space import Spatial_basis_POD
from modulo._pod_time import Temporal_basis_POD
# Add the DMD (11/11/2021, Miguel)
from modulo._dmd_s import dmd_s
from modulo._dft import dft_fit
import math

from tqdm import tqdm



class MODULO:
    """
    MODULO (MODal mULtiscale pOd) is a software developed at the von Karman Institute to perform Multiscale
    Modal Analysis of numerical and experimental data using the Multiscale Proper Orthogonal Decomposition (mPOD).

    Theoretical foundation can be found at:
    https://arxiv.org/abs/1804.09646

    Presentation of the MODULO framework available here:
    https://arxiv.org/pdf/2004.12123.pdf

    YouTube channel with hands-on tutorials can be found at:
    https://youtube.com/playlist?list=PLEJZLD0-4PeKW6Ze984q08bNz28GTntkR

    All the codes so far assume that the dataset is equally spaced both in space (i.e. along a Cartesian grid)
    and in time. The extension to non-uniformly sampled data will be included in the next release.


    """

    def __init__(self, data: np.array,
                 N_PARTITIONS: int = 1,
                 FOLDER_OUT='./',
                 MEMORY_SAVING: bool = False,
                 n_Modes: int= 10,   
                 SAT: int=100,
                 dtype: str = 'float32'):
        """
        This function initializes the main parameters needed by MODULO.

        Attributes:
       
        :param data: This is the data matrix to factorize. It is a np.array with 
        shape ((N_S, N_T)). If the data has not yet been prepared in the form of a np.array,
        the method ReadData in MODULO can be used (see ReadData).
        :param FOLDER_OUT: Folder in which the output will be stored.
        The output includes the matrices Phi, Sigma and Psi (optional) and temporary files 
        used for some of the calculations (e.g.: for memory saving).

        :param MEMORY_SAVING: Bool param. If True, the Memory Saving feature is activated.
        For details on this feature, see the video of D. Ninni (PhD @ Politecnico of Bari and former STP student at VKI).
        https://youtu.be/LclxO1WTuao

        :param N_PARTITIONS: If memory saving feature is active, this parameter sets the number of partitions
        that will be used to store the data matrices during the computations.
        
        :param n_Modes: Number of Modes to be computed 
        
        
        
        """

        if not isinstance(data, np.ndarray):
            raise TypeError("Please check that your database is in an numpy array format.")

        self.MEMORY_SAVING = MEMORY_SAVING
        
        # Assign the number of modes
        self.n_Modes=n_Modes
        # Max number of modes per scale (only relevant for the mPOD)
        self.SAT=100
        # Check the data type
        self.D = data.astype(dtype)
        # Number of points in time and space 
        self.N_T = data.shape[1]
        self.N_S = data.shape[0]

        if N_PARTITIONS >= self.N_T:
            raise AttributeError("The number of requested partitions is greater of the total columns (N_T). Please,"
                                 "try again.")

        self.N_PARTITIONS = N_PARTITIONS

        self.FOLDER_OUT = FOLDER_OUT + 'MODULO_tmp/'

        os.makedirs(self.FOLDER_OUT, exist_ok=True)

    def _data_processing(self,
                         MR: bool = False,
                         SAVE_D: bool = False):
        """
        This method pre-process the data before running the factorization. 
        If the memory saving option is active, the method ensures the correct splitting of the data matrix
        into the required partitions.
        If the mean removal is desired (MR: True), this method removes the time averaged column from the 
        data matrix D.
        If neither the mean removal nor the memory saving options are active, this method is skipped.

        :param MR: bool
                    if True, the mean field is removed from the data matrix.

        :param SAVE_D: bool
                    if True, the D matrix is saved in the folder decided by the user


        :return: D directly (as class attribute) if Memory saving is not active. 
                 Otherwise, it returns None and the matrix is automatically saved on disk.
        """

        self.D = DataMatrix(self.D, self.FOLDER_OUT, MEMORY_SAVING=self.MEMORY_SAVING,
                            N_PARTITIONS=self.N_PARTITIONS, MR=MR, SAVE_D=SAVE_D)

        return

    def _correlation_matrix(self,
                            SAVE_K: bool = True):
        """
        This method computes the time correlation matrix. Here the memory saving is
        beneficial for large datasets. Since the matrix could be potentially heavy, it is automatically stored on disk to 
        minimize the usage of the RAM. This feature can be deactivated by setting SAVE_K = False. 
        In this case, the correlation matrix is returned to the main class.

        :param SAVE_K: bool
            A flag deciding if the matrix will be stored in the disk (in FOLDER_OUT/MODULO_tmp) or not. 
            Default option is 'True'. This attribute is passed to the class 
            in order to decide if it has to be loaded from disk or not.


        :return K: np.array
                The correlation matrix D^T D (as class attribute) if Memory saving is not active. 
                Otherwise, it returns None and the matrix is automatically saved on disk.

        """

        self.SAVE_K = SAVE_K

        self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS, self.MEMORY_SAVING,
                                   self.FOLDER_OUT, self.SAVE_K, D=self.D)

        return

    def _temporal_basis_POD(self,
                            SAVE_T_POD: bool = True):
        """
        This method computes the temporal structure for the Proper Orthogonal Decomposition (POD) computation. 
        The theoretical background of the POD is briefly recalled here:

        https://youtu.be/8fhupzhAR_M

        The diagonalization of K is computed via Singular Value Decomposition (SVD).
        A speedup is available if the user is on Linux machine, in which case MODULO
        exploits the power of JAX and its Numpy implementation.

        For more on JAX:

        https://github.com/google/jax
        https://jax.readthedocs.io/en/latest/jax.numpy.html

        If the user is on a Win machine, Linux OS can be used using
        the Windows Subsystem for Linux.

        For more on WSL:
        https://docs.microsoft.com/en-us/windows/wsl/install-win10

        :param SAVE_T_POD: bool
                Flag deciding if the results will be stored on the disk. 
                Default value is True, to limit the RAM's usage.
                Note that this might cause a minor slowdown for the loading,
                but the tradeoff seems worthy. 
                This attribute is passed to the MODULO class.


        POD temporal basis are returned if MEMORY_SAVING is not active. Otherwise all the results are saved on disk.

        :return Psi_P: np.array
                POD Psis

        :return Sigma_P: np.array
                POD Sigmas. If needed, Lambdas can be easily computed recalling that: Sigma_P = np.sqrt(Lambda_P)
        """

        if self.MEMORY_SAVING:
            K = np.load(self.FOLDER_OUT + "/correlation_matrix/k_matrix.npz")['K']
            SAVE_T_POD = True
        else:
            K = self.K

        Psi_P, Sigma_P = Temporal_basis_POD(self.K, SAVE_T_POD, 
                                            self.FOLDER_OUT,self.n_Modes)

        del K
        return Psi_P, Sigma_P if not self.MEMORY_SAVING else None

    def _spatial_basis_POD(self, Psi_P, Sigma_P,
                           SAVE_SPATIAL_POD: bool = True):
        """
        This method computes the spatial structure for the Proper Orthogonal Decomposition (POD) computation. 
        The theoretical background of the POD is briefly recalled here:

        https://youtu.be/8fhupzhAR_M
        
        :param Psi_P: np.array
                POD temporal basis
        :param Sigma_P: np.array
                POD Sigmas
        :param SAVE_SPATIAL_POD: bool
                Flag deciding if the results will be stored on the disk. 
                Default value is True, to limit the RAM's usage.
                Note that this might cause a minor slowdown for the loading,
                but the tradeoff seems worthy. 
                This attribute is passed to the MODULO class.

        :return Phi_P: np.array
            POD Phis
            
        """

        self.SAVE_SPATIAL_POD = SAVE_SPATIAL_POD

        if self.MEMORY_SAVING:
            '''Loading temporal basis from disk. They're already in memory otherwise.'''
            Psi_P = np.load(self.FOLDER_OUT + 'POD/temporal_basis.npz')['Psis']
            Sigma_P = np.load(self.FOLDER_OUT + 'POD/temporal_basis.npz')['Sigmas']

        Phi_P = Spatial_basis_POD(self.D, N_T=self.N_T, PSI_P=Psi_P, Sigma_P=Sigma_P,
                                  MEMORY_SAVING=self.MEMORY_SAVING, FOLDER_OUT=self.FOLDER_OUT,
                                  N_PARTITIONS=self.N_PARTITIONS, SAVE_SPATIAL_POD=SAVE_SPATIAL_POD)

        return Phi_P if not self.MEMORY_SAVING else None

    def _temporal_basis_mPOD(self, K, Nf, Ex, F_V, Keep, boundaries, MODE, dt, K_S=False):
        """
        This function computes the temporal structures of each scale in the mPOD, as in step 4 of the algorithm
        ref: Multi-Scale Proper Orthogonal Decomposition of Complex Fluid Flows - M. A. Mendez et al.

        :param K: np.array
                Temporal correlation matrix
        :param Nf: np.array
                Order of the FIR filters that are used to isolate each of the scales
        :param Ex: int
                Extension at the boundaries of K to impose the boundary conditions (see boundaries)
                It must be at least as Nf.
        :param F_V: np.array
                Frequency splitting vector, containing the frequencies of each scale (see article).
                If the time axis is in seconds, these frequencies are in Hz.
        :param Keep: np.array
                Scale keep
        :param boundaries: str -> {'nearest', 'reflect', 'wrap' or 'extrap'}
                Define the boundary conditions for the filtering process, in order to avoid edge effects.
                The available boundary conditions are the classic ones implemented for image processing:
                nearest', 'reflect', 'wrap' or 'extrap'. See also https://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html      
        :param MODE: str -> {‘reduced’, ‘complete’, ‘r’, ‘raw’}
                A QR factorization is used to enforce the orthonormality of the mPOD basis, to compensate
                for the non-ideal frequency response of the filters. 
                The option MODE from np.linalg.qr carries out this operation.

        :return PSI_M: np.array
                Multiscale POD temporal basis

        """

        if self.MEMORY_SAVING:
            K = np.load(self.FOLDER_OUT + "/correlation_matrix/k_matrix.npz")['K']

        PSI_M = temporal_basis_mPOD(K, Nf, Ex, F_V, Keep,
                                    boundaries, MODE, dt, FOLDER_OUT=self.FOLDER_OUT,
                                    n_Modes=self.n_Modes,
                                    MEMORY_SAVING=self.MEMORY_SAVING, K_S=K_S)

        return PSI_M if not self.MEMORY_SAVING else None

    def _spatial_basis_mPOD(self, D, PSI_M, SAVE):
        """
        This function implements the last step of the mPOD algorithm: completing the decomposition. 
        Here we project from psis, to get phis and sigmas

        :param D: np.array
                data matrix
        :param PSI_M: np.array
                temporal basis for the mPOD. Remember that it is not possible to impose both basis matrices
                phis and psis: given one of the two, the other is univocally determined.
        :param SAVE: bool
                if True, MODULO saves the results on disk.

        :return Phi_M: np.array
                mPOD Phis (Matrix of spatial structures)
        :return Psi_M: np.array
                mPOD Psis (Matrix of temporal structures)
        :return Sigma_M: np.array
                mPOD Sigmas (vector of amplitudes, i.e. the diagonal of Sigma_M)

        """

        Phi_M, Psi_M, Sigma_M = spatial_basis_mPOD(D, PSI_M, N_T=self.N_T, N_PARTITIONS=self.N_PARTITIONS,
                                                   N_S=self.N_S, MEMORY_SAVING=self.MEMORY_SAVING,
                                                   FOLDER_OUT=self.FOLDER_OUT,
                                                   SAVE=SAVE)

        return Phi_M, Psi_M, Sigma_M

    def compute_mPOD(self, Nf, Ex, F_V, Keep, boundaries, MODE, dt, SAVE=False):
        """
        This function computes the temporal structures of each scale in the mPOD, as in step 4 of the algorithm
        ref: Multi-Scale Proper Orthogonal Decomposition of Complex Fluid Flows - M. A. Mendez et al.

        :param K: np.array
                Temporal correlation matrix

        :param Nf: np.array
                Order of the FIR filters that are used to isolate each of the scales

        :param Ex: int
                Extension at the boundaries of K to impose the boundary conditions (see boundaries)
                It must be at least as Nf.

        :param F_V: np.array
                Frequency splitting vector, containing the frequencies of each scale (see article).
                If the time axis is in seconds, these frequencies are in Hz.

        :param Keep: np.array
                Scale keep

        :param boundaries: str -> {'nearest', 'reflect', 'wrap' or 'extrap'}
                Define the boundary conditions for the filtering process, in order to avoid edge effects.
                The available boundary conditions are the classic ones implemented for image processing:
                nearest', 'reflect', 'wrap' or 'extrap'. See also https://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html

        :param MODE: str -> {‘reduced’, ‘complete’, ‘r’, ‘raw’}
                A QR factorization is used to enforce the orthonormality of the mPOD basis, to compensate
                for the non-ideal frequency response of the filters.
                The option MODE from np.linalg.qr carries out this operation.
        :param dt: float
                temporal step

        :return Phi_M: np.array
                mPOD Phis (Matrix of spatial structures)
        :return Psi_M: np.array
                mPOD Psis (Matrix of temporal structures)
        :return Sigma_M: np.array
                mPOD Sigmas (vector of amplitudes, i.e. the diagonal of Sigma_M

        """
        K = CorrelationMatrix(self.N_T, self.N_PARTITIONS, self.MEMORY_SAVING,
                                   self.FOLDER_OUT, D=self.D)

        if self.MEMORY_SAVING:
            K = np.load(self.FOLDER_OUT + 'correlation_matrix/k_matrix.npz')['K']

        PSI_M = temporal_basis_mPOD(K=K, Nf=Nf, Ex=Ex, F_V=F_V, Keep=Keep, boundaries=boundaries,
                                    MODE=MODE, dt=dt, FOLDER_OUT=self.FOLDER_OUT, 
                                    n_Modes=self.n_Modes, K_S=False,
                                    MEMORY_SAVING=self.MEMORY_SAVING, SAT=self.SAT)

        Phi_M, Psi_M, Sigma_M = spatial_basis_mPOD(self.D, PSI_M, N_T=self.N_T, N_PARTITIONS=self.N_PARTITIONS,
                                                   N_S=self.N_S, MEMORY_SAVING=self.MEMORY_SAVING,
                                                   FOLDER_OUT=self.FOLDER_OUT,
                                                   SAVE=SAVE)

        return Phi_M, Psi_M, Sigma_M

    def compute_POD(self, SAVE_T_POD: bool = True):
        """
        This method computes the temporal structure for the Proper Orthogonal Decomposition (POD) computation.
        The theoretical background of the POD is briefly recalled here:

        https://youtu.be/8fhupzhAR_M
        
        :return Psi_P: np.array
                POD Psis

        :return Sigma_P: np.array
                POD Sigmas. If needed, Lambdas can be easily computed recalling that: Sigma_P = np.sqrt(Lambda_P)

        :return Phi_P: np.array
                POD Phis
        """
        self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS, self.MEMORY_SAVING,
                                   self.FOLDER_OUT, D=self.D)

        if self.MEMORY_SAVING:
            self.K = np.load(self.FOLDER_OUT + 'correlation_matrix/k_matrix.npz')['K']

        Psi_P, Sigma_P = Temporal_basis_POD(self.K, SAVE_T_POD, 
                                            self.FOLDER_OUT,self.n_Modes)

        Phi_P = Spatial_basis_POD(self.D, N_T=self.N_T, PSI_P=Psi_P, Sigma_P=Sigma_P,
                                  MEMORY_SAVING=self.MEMORY_SAVING, FOLDER_OUT=self.FOLDER_OUT,
                                  N_PARTITIONS=self.N_PARTITIONS)

        return Phi_P, Psi_P, Sigma_P
    
    def compute_DMD(self, SAVE_T_DMD: bool = True,F_S=1):
        """
        This method computes the Dynamic Mode Decomposition of the data
        using the algorithm in https://arxiv.org/abs/1312.0041. 
        See the slides in the DDFM course for more details or see
        https://arxiv.org/abs/2001.01971.
        
        :return Phi_D: np.array
                DMD Psis

        :return Lambda_D: np.array
                DMD Eigenvalues (of the reduced propagator)

        :return freqs: np.array
                Frequencies (in Hz, associated to the DMD modes)
                
         :return a0s: np.array
                Initial Coefficients of the Modes        
                
        """
        
        # If Memory saving is active, we must load back the data 
        if self.MEMORY_SAVING:

         if self.N_T % self.N_PARTITIONS != 0:
            tot_blocks_col = self.N_PARTITIONS + 1
         else:
            tot_blocks_col = self.N_PARTITIONS
         
         # Prepare the D matrix again   
         D=np.zeros((self.N_S,self.N_T))
         R1=0
                  
         print(' \n Reloading D from tmp...')
         for k in tqdm(range(tot_blocks_col)):
            di = np.load(self.FOLDER_OUT + f"/data_partitions/di_{k + 1}.npz")['di']    
            R2=R1+np.shape(di)[1]
            D[:,R1:R2]=di         
            R1=R2
                     
         # Partition the data into D_1 and D_2, then clean memory
         D_1=D[:,0:self.N_T-1] ; D_2=D[:,1:self.N_T]
         # Compute the DMD
         Phi_D, Lambda, freqs, a0s = dmd_s(D_1,D_2,self.n_Modes,F_S)
        
        if not self.MEMORY_SAVING: 
         # Partition the data into D_1 and D_2, then clean memory
         D_1=self.D[:,0:self.N_T-1] ; D_2=self.D[:,1:self.N_T]

         # Compute the DMD
         Phi_D, Lambda, freqs, a0s = dmd_s(D_1,D_2,self.n_Modes,F_S)

        return Phi_D, Lambda, freqs, a0s
    
    def compute_DFT(self, F_S, SAVE_DFT=False):
        """
        This method computes the Discrete Fourier Transform of your data.

        :param F_S: float,
                Sampling Frequency [Hz]
        :param SAVE_DFT: bool,
                If True, MODULO will save the output in self.FOLDER OUT/MODULO_tmp

        :return: Sorted_Freqs: np.array,
                    Sorted Frequencies
        :return Phi_F: np.array,
                    DFT Phis
        :return Sigma_F: np.array,
                    DFT Sigmas
        """
        if self.D is None:
            D = np.load(self.FOLDER_OUT + '/MODULO_tmp/data_matrix/database.npz')['D']
            SAVE_DFT = True
        else:
            D = self.D

        Sorted_Freqs, Phi_F, SIGMA_F = dft_fit(self.N_T, F_S, D, self.FOLDER_OUT, SAVE_DFT=SAVE_DFT)

        return Sorted_Freqs, Phi_F, SIGMA_F
