# Functional ones:
import os
import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import pairwise_kernels
# To have fancy loading bar
from tqdm import tqdm

# All the functions from the modulo package 
from modulo_vki.core._dft import dft_fit
from modulo_vki.core.temporal_structures import dft, temporal_basis_mPOD
from modulo_vki.core._dmd_s import dmd_s
from modulo_vki.core._k_matrix import CorrelationMatrix, spectral_filter, kernelized_K
from modulo_vki.core._mpod_space import spatial_basis_mPOD
# from modulo_vki.core._mpod_time import temporal_basis_mPOD
from modulo_vki.core._pod_space import Spatial_basis_POD
from modulo_vki.core._pod_time import Temporal_basis_POD
from modulo_vki.core._spod_s import compute_SPOD_s
from modulo_vki.core._spod_t import compute_SPOD_t
from modulo_vki.utils._utils import switch_svds

from modulo_vki.utils.read_db import ReadData
from modulo_vki.core.utils import segment_and_fft, pod_from_dhat

class ModuloVKI:
    """
    MODULO (MODal mULtiscale pOd) is a software developed at the von Karman Institute
    to perform Multiscale Modal Analysis using Multiscale Proper Orthogonal Decomposition (mPOD)
    on numerical and experimental data.

    References
    ----------
    - Theoretical foundation:
      https://arxiv.org/abs/1804.09646

    - MODULO framework presentation:
      https://arxiv.org/pdf/2004.12123.pdf

    - Hands-on tutorial videos:
      https://youtube.com/playlist?list=PLEJZLD0-4PeKW6Ze984q08bNz28GTntkR

    Notes
    -----
    MODULO operations assume the dataset is uniformly spaced in both space
    (Cartesian grid) and time. For non-cartesian grids, the user must 
    provide a weights vector `[w_1, w_2, ..., w_Ns]` where `w_i = area_cell_i / area_grid`.
    """

    def __init__(self,
                 data: np.ndarray,
                 N_PARTITIONS: int = 1,
                 FOLDER_OUT: str = './',
                 SAVE_K: bool = False,
                 N_T: int = 100,
                 N_S: int = 200,
                 n_Modes: int = 10,
                 dtype: str = 'float32',
                 eig_solver: str = 'eigh',
                 svd_solver: str = 'svd_sklearn_truncated',
                 weights: np.ndarray = np.array([])):
        """
        Initialize the MODULO analysis.

        Parameters
        ----------
        data : np.ndarray
            Data matrix of shape (N_S, N_T) to factorize. If not yet formatted, use the `ReadData`
            method provided by MODULO. When memory saving mode (N_PARTITIONS > 1) is active,
            set this parameter to None and use prepared partitions instead.

        N_PARTITIONS : int, default=1
            Number of partitions used for memory-saving computation. If set greater than 1,
            data must be partitioned in advance and `data` set to None.

        FOLDER_OUT : str, default='./'
            Directory path to store output (Phi, Sigma, Psi matrices) and intermediate
            calculation files (e.g., partitions, correlation matrix).

        SAVE_K : bool, default=False
            Whether to store the correlation matrix K to disk in
            `FOLDER_OUT/correlation_matrix`.

        N_T : int, default=100
            Number of temporal snapshots. Mandatory when using partitions (N_PARTITIONS > 1).

        N_S : int, default=200
            Number of spatial grid points. Mandatory when using partitions (N_PARTITIONS > 1).

        n_Modes : int, default=10
            Number of modes to compute.

        dtype : str, default='float32'
            Data type for casting input data.

        eig_solver : str, default='eigh'
            Solver for eigenvalue decomposition.

        svd_solver : str, default='svd_sklearn_truncated'
            Solver for Singular Value Decomposition (SVD).

        weights : np.ndarray, default=np.array([])
            Weights vector `[w_1, w_2, ..., w_Ns]` to account for non-uniform spatial grids.
            Defined as `w_i = area_cell_i / area_grid`. Leave empty for uniform grids.
        """

        print("MODULO (MODal mULtiscale pOd) is a software developed at the von Karman Institute to perform "
              "data driven modal decomposition of numerical and experimental data. \n")

        if not isinstance(data, np.ndarray) and N_PARTITIONS == 1:
            raise TypeError(
                "Please check that your database is in an numpy array format. If D=None, then you must have memory saving (N_PARTITIONS>1)")

        if N_PARTITIONS > 1:
            self.MEMORY_SAVING = True
        else:
            self.MEMORY_SAVING = False

            # Assign the number of modes
        self.n_Modes = n_Modes
        # If particular needs, override choice for svd and eigen solve
        self.svd_solver = svd_solver.lower()
        self.eig_solver = eig_solver.lower()
        possible_svds = ['svd_numpy', 'svd_scipy_sparse', 'svd_sklearn_randomized', 'svd_sklearn_truncated']
        possible_eigs = ['svd_sklearn_randomized', 'eigsh', 'eigh']

        if self.svd_solver not in possible_svds:
            raise NotImplementedError("The requested SVD solver is not implemented. Please pick one of the following:"
                                      "which belongs to: \n {}".format(possible_svds))

        if self.eig_solver not in possible_eigs:
            raise NotImplementedError("The requested EIG solver is not implemented. Please pick one of the following: "
                                      " \n {}".format(possible_eigs))

        # if N_PARTITIONS >= self.N_T:
        #     raise AttributeError("The number of requested partitions is greater of the total columns (N_T). Please,"
        #                          "try again.")

        self.N_PARTITIONS = N_PARTITIONS
        self.FOLDER_OUT = FOLDER_OUT
        self.SAVE_K = SAVE_K

        if self.MEMORY_SAVING:
            os.makedirs(self.FOLDER_OUT, exist_ok=True)
        
        # Load the data matrix
        if isinstance(data, np.ndarray):
            # Number of points in time and space
            self.N_T = data.shape[1]
            self.N_S = data.shape[0]
            # Check the data type
            self.D = data.astype(dtype)
        else:
            self.D = None  # D is never saved when N_partitions >1
            self.N_S = N_S  # so N_S and N_t must be given as parameters of modulo
            self.N_T = N_T
        
        '''If the grid is not cartesian, ensure inner product is properly defined using weights.'''
        
        if weights.size == 0:
                print('Modulo assumes you have a uniform grid. If not, please provide weights as parameters.')
        else: 
                if len(weights) == self.N_S:
                        print("The weights you have input have the size of the columns of D \n"
                                  "MODULO has considered that you have already duplicated the dimensions of the weights "
                                  "to match the dimensions of the D columns \n")
                        self.weights = weights
                elif len(weights) == 2 * self.N_S:
                        print("Assuming 2D domain. Automatically duplicating the weights to match the dimension of the D columns \n")
                        self.weights = np.concatenate((weights, weights))
                else:
                        raise AttributeError("Make sure the size of the weight array is twice smaller than the size of D")
                
                if isinstance(data, np.ndarray):
                        # Apply the weights only if D exist.
                        # If not (i.e. N_partitions >1), weights are applied in _k_matrix.py when loading partitions of D
                        self.Dstar = np.transpose(np.transpose(self.D) * np.sqrt(self.weights))
                else:
                        self.Dstar = None
                        
        # # Load and applied the weights to the D matrix
        # if weights.size != 0:
        #     if len(weights) == self.N_S:
        #         print("The weights you have input have the size of the columns of D \n"
        #               "MODULO has considered that you have already duplicated the dimensions of the weights "
        #               "to match the dimensions of the D columns \n")
        #         self.weights = weights
        #     elif 2 * len(weights) == self.N_S:  # 2D computation only
        #         self.weights = np.concatenate((weights, weights))
        #         print("Modulo assumes you have a 2D domain and has duplicated the weight "
        #               "array to match the size of the D columns \n")
        #         print(weights)
        #     else:
        #         raise AttributeError("Make sure the size of the weight array is twice smaller than the size of D")
        #     # Dstar is used to compute the K matrix
        #     if isinstance(data, np.ndarray):
        #         # Apply the weights only if D exist.
        #         # If not (i.e. N_partitions >1), weights are applied in _k_matrix.py when loading partitions of D
        #         self.Dstar = np.transpose(np.transpose(self.D) * np.sqrt(self.weights))
        #     else:
        #         self.Dstar = None
        # else:
        #     print("Modulo assumes you have a uniform grid. "
        #           "If not, please give the weights as parameters of MODULO!")
        #     self.weights = weights
        #     self.Dstar = self.D
        
        pass 



    def _temporal_basis_POD(self,
                            SAVE_T_POD: bool = False):
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

        Psi_P, Sigma_P = Temporal_basis_POD(K, SAVE_T_POD,
                                            self.FOLDER_OUT, self.n_Modes, self.eig_solver)

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

        PSI_M = temporal_basis_mPOD(K=K, Nf=Nf, Ex=Ex, F_V=F_V, Keep=Keep, boundaries=boundaries,
                                    MODE=MODE, dt=dt, FOLDER_OUT=self.FOLDER_OUT,
                                    n_Modes=self.n_Modes, K_S=False,
                                    MEMORY_SAVING=self.MEMORY_SAVING, SAT=self.SAT, eig_solver=self.eig_solver)

        return PSI_M if not self.MEMORY_SAVING else None

    def _spatial_basis_mPOD(self, D, PSI_M, SAVE):
        """
        This function implements the last step of the mPOD algorithm:
        completing the decomposition. Here we project from psis, to get phis and sigmas

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

    def compute_mPOD(self, Nf, Ex, F_V, Keep, SAT, boundaries, MODE, dt, SAVE=False):
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

        :param SAT: Maximum number of modes per scale.
                    Only used for mPOD (max number of modes per scale)

        :param dt: float
                temporal step

        :return Phi_M: np.array
                mPOD Phis (Matrix of spatial structures)
        :return Psi_M: np.array
                mPOD Psis (Matrix of temporal structures)
        :return Sigma_M: np.array
                mPOD Sigmas (vector of amplitudes, i.e. the diagonal of Sigma_M

        """

        print('Computing correlation matrix D matrix...')
        self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS,
                                   self.MEMORY_SAVING,
                                   self.FOLDER_OUT, self.SAVE_K, D=self.Dstar)

        if self.MEMORY_SAVING:
            self.K = np.load(self.FOLDER_OUT + '/correlation_matrix/k_matrix.npz')['K']

        print("Computing Temporal Basis...")

        PSI_M = temporal_basis_mPOD(K=self.K, Nf=Nf, Ex=Ex, F_V=F_V, Keep=Keep, boundaries=boundaries,
                                    MODE=MODE, dt=dt, FOLDER_OUT=self.FOLDER_OUT,
                                    n_Modes=self.n_Modes, MEMORY_SAVING=self.MEMORY_SAVING, SAT=SAT,
                                    eig_solver=self.eig_solver)

        print("Done.")

        if hasattr(self, 'D'):  # if self.D is available:
            print('Computing Phi from D...')
            Phi_M, Psi_M, Sigma_M = spatial_basis_mPOD(self.D, PSI_M, N_T=self.N_T, N_PARTITIONS=self.N_PARTITIONS,
                                                       N_S=self.N_S, MEMORY_SAVING=self.MEMORY_SAVING,
                                                       FOLDER_OUT=self.FOLDER_OUT,
                                                       SAVE=SAVE)

        else:  # if not, the memory saving is on and D will not be used. We pass a dummy D
            print('Computing Phi from partitions...')
            Phi_M, Psi_M, Sigma_M = spatial_basis_mPOD(np.array([1]), PSI_M, N_T=self.N_T,
                                                       N_PARTITIONS=self.N_PARTITIONS,
                                                       N_S=self.N_S, MEMORY_SAVING=self.MEMORY_SAVING,
                                                       FOLDER_OUT=self.FOLDER_OUT,
                                                       SAVE=SAVE)

        print("Done.")

        return Phi_M, Psi_M, Sigma_M


    def compute_POD_K(self, SAVE_T_POD: bool = False):
        """
        This method computes the Proper Orthogonal Decomposition (POD) of a dataset
        using the snapshot approach, i.e. working on the temporal correlation matrix.
        The eig solver for K is defined in 'eig_solver'
        The theoretical background of the POD is briefly recalled here:

        https://youtu.be/8fhupzhAR_M

        :return Psi_P: np.array
                POD Psis

        :return Sigma_P: np.array
                POD Sigmas. If needed, Lambdas can be easily computed recalling that: Sigma_P = np.sqrt(Lambda_P)

        :return Phi_P: np.array
                POD Phis
        """

        print('Computing correlation matrix...')
        self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS,
                                   self.MEMORY_SAVING,
                                   self.FOLDER_OUT, self.SAVE_K, 
                                   D=self.Dstar, weights=self.weights)

        if self.MEMORY_SAVING:
            self.K = np.load(self.FOLDER_OUT + '/correlation_matrix/k_matrix.npz')['K']

        print("Computing Temporal Basis...")
        Psi_P, Sigma_P = Temporal_basis_POD(self.K, SAVE_T_POD,
                                            self.FOLDER_OUT, self.n_Modes, eig_solver=self.eig_solver)
        print("Done.")
        print("Computing Spatial Basis...")

        if self.MEMORY_SAVING:  # if self.D is available:
            print('Computing Phi from partitions...')
            Phi_P = Spatial_basis_POD(np.array([1]), N_T=self.N_T,
                             PSI_P=Psi_P,
                             Sigma_P=Sigma_P,
                             MEMORY_SAVING=self.MEMORY_SAVING,
                             FOLDER_OUT=self.FOLDER_OUT,
                             N_PARTITIONS=self.N_PARTITIONS)

        else:  # if not, the memory saving is on and D will not be used. We pass a dummy D
           print('Computing Phi from D...')
           Phi_P = Spatial_basis_POD(self.D, N_T=self.N_T,
                                    PSI_P=Psi_P,
                                    Sigma_P=Sigma_P,
                                    MEMORY_SAVING=self.MEMORY_SAVING,
                                    FOLDER_OUT=self.FOLDER_OUT,
                                    N_PARTITIONS=self.N_PARTITIONS)
        print("Done.")

        return Phi_P, Psi_P, Sigma_P

    def compute_POD_svd(self, SAVE_T_POD: bool = False):
        """
        This method computes the Proper Orthogonal Decomposition (POD) of a dataset
        using the SVD decomposition. The svd solver is defined by 'svd_solver'.
        Note that in this case, the memory saving option is of no help, since
        the SVD must be performed over the entire dataset.

        https://youtu.be/8fhupzhAR_M

        :return Psi_P: np.array
            POD Psis

        :return Sigma_P: np.array
            POD Sigmas. If needed, Lambdas can be easily computed recalling that: Sigma_P = np.sqrt(Lambda_P)

         :return Phi_P: np.array
            POD Phis
        """
        # If Memory saving is active, we must load back the data.
        # This process is memory demanding. Different SVD solver will handle this differently.

        if self.MEMORY_SAVING:
            if self.N_T % self.N_PARTITIONS != 0:
                tot_blocks_col = self.N_PARTITIONS + 1
            else:
                tot_blocks_col = self.N_PARTITIONS

            # Prepare the D matrix again
            D = np.zeros((self.N_S, self.N_T))
            R1 = 0

            # print(' \n Reloading D from tmp...')
            for k in tqdm(range(tot_blocks_col)):
                di = np.load(self.FOLDER_OUT + f"/data_partitions/di_{k + 1}.npz")['di']
                R2 = R1 + np.shape(di)[1]
                D[:, R1:R2] = di
                R1 = R2

            # Now that we have D back, we can proceed with the SVD approach
            Phi_P, Psi_P, Sigma_P = switch_svds(D, self.n_Modes, self.svd_solver)


        else:  # self.MEMORY_SAVING:
            Phi_P, Psi_P, Sigma_P = switch_svds(self.D, self.n_Modes, self.svd_solver)

        return Phi_P, Psi_P, Sigma_P

    def compute_DMD_PIP(self, SAVE_T_DMD: bool = True, F_S=1):
        """
        This method computes the Dynamic Mode Decomposition of the data
        using the algorithm in https://arxiv.org/abs/1312.0041, which is basically the same as
        the PIP algorithm proposed in https://www.sciencedirect.com/science/article/abs/pii/0167278996001248
        See v1 of this paper https://arxiv.org/abs/2001.01971 for more details (yes, reviewers did ask to omit this detail in v2).

        :return Phi_D: np.array
                DMD Phis. As for the DFT, these are complex.

        :return Lambda_D: np.array
                DMD Eigenvalues (of the reduced propagator). These are complex.

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
            D = np.zeros((self.N_S, self.N_T))
            R1 = 0

            # print(' \n Reloading D from tmp...')
            for k in tqdm(range(tot_blocks_col)):
                di = np.load(self.FOLDER_OUT + f"/data_partitions/di_{k + 1}.npz")['di']
                R2 = R1 + np.shape(di)[1]
                D[:, R1:R2] = di
                R1 = R2

            # Compute the DMD
            Phi_D, Lambda, freqs, a0s = dmd_s(D[:, 0:self.N_T - 1],
                                              D[:, 1:self.N_T], self.n_Modes, F_S, svd_solver=self.svd_solver)

        else:
            Phi_D, Lambda, freqs, a0s = dmd_s(self.D[:, 0:self.N_T - 1],
                                              self.D[:, 1:self.N_T], self.n_Modes, F_S, SAVE_T_DMD=SAVE_T_DMD,
                                              svd_solver=self.svd_solver, FOLDER_OUT=self.FOLDER_OUT)

        return Phi_D, Lambda, freqs, a0s

    def compute_DFT(self, F_S, SAVE_DFT=False):
        """
        This method computes the Discrete Fourier Transform of your data.

        Check out this tutorial: https://www.youtube.com/watch?v=8fhupzhAR_M&list=PLEJZLD0-4PeKW6Ze984q08bNz28GTntkR&index=2

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
            Sorted_Freqs, Phi_F, SIGMA_F = dft_fit(self.N_T, F_S, D, self.FOLDER_OUT, SAVE_DFT=SAVE_DFT)

        else:
            Sorted_Freqs, Phi_F, SIGMA_F = dft_fit(self.N_T, F_S, self.D, self.FOLDER_OUT, SAVE_DFT=SAVE_DFT)

        return Sorted_Freqs, Phi_F, SIGMA_F

    def compute_SPOD_t(self, F_S, L_B=500, O_B=250, n_Modes=10, SAVE_SPOD=True):
        """
        This method computes the Spectral POD of your data. This is the one by Towne et al
        (https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition-and-its-relationship-to-dynamic-mode-decomposition-and-resolvent-analysis/EC2A6DF76490A0B9EB208CC2CA037717)

        :param F_S: float,
                Sampling Frequency [Hz]
        :param L_B: float,
                lenght of the chunks
        :param O_B: float,
                Overlapping between blocks in the chunk
        :param n_Modes: float,
               number of modes to be computed for each frequency
        :param SAVE_SPOD: bool,
                If True, MODULO will save the output in self.FOLDER OUT/MODULO_tmp
        :return Psi_P_hat: np.array
                Spectra of the SPOD Modes
        :return Sigma_P: np.array
                Amplitudes of the SPOD Modes.
        :return Phi_P: np.array
                SPOD Phis
        :return freq: float
                frequency bins for the Spectral POD


        """
        if self.D is None:
            D = np.load(self.FOLDER_OUT + '/MODULO_tmp/data_matrix/database.npz')['D']
            Phi_SP, Sigma_SP, Freqs_Pos = compute_SPOD_t(D, F_S, L_B=L_B, O_B=O_B,
                                                         n_Modes=n_Modes, SAVE_SPOD=SAVE_SPOD,
                                                         FOLDER_OUT=self.FOLDER_OUT, possible_svds=self.svd_solver)
        else:
            Phi_SP, Sigma_SP, Freqs_Pos = compute_SPOD_t(self.D, F_S, L_B=L_B, O_B=O_B,
                                                         n_Modes=n_Modes, SAVE_SPOD=SAVE_SPOD,
                                                         FOLDER_OUT=self.FOLDER_OUT, possible_svds=self.svd_solver)

        return Phi_SP, Sigma_SP, Freqs_Pos

        # New Decomposition: SPOD f

    def compute_SPOD_s(self, F_S, N_O=100, f_c=0.3, n_Modes=10, SAVE_SPOD=True):
        """
        This method computes the Spectral POD of your data.
        This is the one by Sieber
        et al (https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition/DCD8A6EDEFD56F5A9715DBAD38BD461A)

        :param F_S: float,
                Sampling Frequency [Hz]
        :param N_o: float,
                Semi-Order of the diagonal filter.
                Note that the filter order will be 2 N_o +1 (to make sure it is odd)
        :param f_c: float,
                cut-off frequency of the diagonal filter
        :param n_Modes: float,
               number of modes to be computed
        :param SAVE_SPOD: bool,
                If True, MODULO will save the output in self.FOLDER OUT/MODULO_tmp
        :return Psi_P: np.array
                SPOD Psis
        :return Sigma_P: np.array
                SPOD Sigmas.
        :return Phi_P: np.array
                SPOD Phis
        """

        if self.D is None:
            D = np.load(self.FOLDER_OUT + '/MODULO_tmp/data_matrix/database.npz')['D']

            self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS, self.MEMORY_SAVING,
                                       self.FOLDER_OUT, self.SAVE_K, D=D)

            Phi_sP, Psi_sP, Sigma_sP = compute_SPOD_s(D, self.K, F_S, self.N_S, self.N_T, N_O, f_c,
                                                      n_Modes, SAVE_SPOD, self.FOLDER_OUT, self.MEMORY_SAVING,
                                                      self.N_PARTITIONS)

        else:
            self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS, self.MEMORY_SAVING,
                                       self.FOLDER_OUT, self.SAVE_K, D=self.D)

            Phi_sP, Psi_sP, Sigma_sP = compute_SPOD_s(self.D, self.K, F_S, self.N_S, self.N_T, N_O, f_c,
                                                      n_Modes, SAVE_SPOD, self.FOLDER_OUT, self.MEMORY_SAVING,
                                                      self.N_PARTITIONS)

        return Phi_sP, Psi_sP, Sigma_sP

    def compute_kPOD(self, M_DIST=[1, 10], k_m=0.1, cent=True,
                     n_Modes=10, alpha=1e-6, metric='rbf', K_out=False):
        """
        This function implements the kernel PCA as described in the VKI course https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552/-/online-on-site-hands-on-machine-learning-for-fluid-dynamics-2023

        The computation of the kernel function is carried out as in https://arxiv.org/pdf/2208.07746.pdf.


        :param M_DIST: array,
                position of the two snapshots that will be considered to
                estimate the minimal k. They should be the most different ones.
        :param k_m: float,
                minimum value for the kernelized correlation
        :param alpha: float
                regularization for K_zeta
        :param cent: bool,
                if True, the matrix K is centered. Else it is not
        :param n_Modes: float,
               number of modes to be computed
        :param metric: string,
               This identifies the metric for the kernel matrix. It is a wrapper to 'pairwise_kernels' from sklearn.metrics.pairwise
               Note that different metrics would need different set of parameters. For the moment, only rbf was tested; use any other option at your peril !
        :param K_out: bool,
               If true, the matrix K is also exported as a fourth output.
        :return Psi_xi: np.array
               kPOD's Psis
        :return Sigma_xi: np.array
               kPOD's Sigmas.
        :return Phi_xi: np.array
               kPOD's Phis
        :return K_zeta: np.array
               Kernel Function from which the decomposition is computed.
               (exported only if K_out=True)


        """
        if self.D is None:
            D = np.load(self.FOLDER_OUT + '/MODULO_tmp/data_matrix/database.npz')['D']
        else:
            D = self.D

        # Compute Eucledean distances
        i, j = M_DIST;
        n_s, n_t = np.shape(D)
        M_ij = np.linalg.norm(D[:, i] - D[:, j]) ** 2

        gamma = -np.log(k_m) / M_ij

        K_zeta = pairwise_kernels(D.T, metric='rbf', gamma=gamma)
        print('Kernel K ready')

        # Compute the Kernel Matrix
        n_t = np.shape(D)[1]
        # Center the Kernel Matrix (if cent is True):
        if cent:
            H = np.eye(n_t) - 1 / n_t * np.ones_like(K_zeta)
            K_zeta = H @ K_zeta @ H.T
            print('K_zeta centered')
        # Diagonalize and Sort
        lambdas, Psi_xi = linalg.eigh(K_zeta + alpha * np.eye(n_t), subset_by_index=[n_t - n_Modes, n_t - 1])
        lambdas, Psi_xi = lambdas[::-1], Psi_xi[:, ::-1];
        Sigma_xi = np.sqrt(lambdas);
        print('K_zeta diagonalized')
        # Encode
        # Z_xi=np.diag(Sigma_xi)@Psi_xi.T
        # We compute the spatial structures as projections of the data
        # onto the Psi_xi!
        R = Psi_xi.shape[1]
        PHI_xi_SIGMA_xi = np.dot(D, (Psi_xi))
        # Initialize the output
        PHI_xi = np.zeros((n_s, R))
        SIGMA_xi = np.zeros((R))

        for i in tqdm(range(0, R)):
            # Assign the norm as amplitude
            SIGMA_xi[i] = np.linalg.norm(PHI_xi_SIGMA_xi[:, i])
            # Normalize the columns of C to get spatial modes
            PHI_xi[:, i] = PHI_xi_SIGMA_xi[:, i] / SIGMA_xi[i]

        Indices = np.flipud(np.argsort(SIGMA_xi))  # find indices for sorting in decreasing order
        Sorted_Sigmas = SIGMA_xi[Indices]  # Sort all the sigmas
        Phi_xi = PHI_xi[:, Indices]  # Sorted Spatial Structures Matrix
        Psi_xi = Psi_xi[:, Indices]  # Sorted Temporal Structures Matrix
        Sigma_xi = Sorted_Sigmas  # Sorted Amplitude Matrix
        print('Phi_xi computed')

        if K_out:
            return Phi_xi, Psi_xi, Sigma_xi, K_zeta
        else:
            return Phi_xi, Psi_xi, Sigma_xi


    # --- updated functions down here --- # 
    
    def mPOD(self, Nf, Ex, F_V, Keep, SAT, boundaries, MODE, dt, SAVE=False, K_in=None):
        """
        Multi-Scale Proper Orthogonal Decomposition (mPOD) of a signal.

        Parameters
        ----------
        Nf : np.array
                Orders of the FIR filters used to isolate each scale.
        
        Ex : int
                Extension length at the boundaries to impose boundary conditions (must be at least as large as Nf).
        
        F_V : np.array
                Frequency splitting vector, containing the cutoff frequencies for each scale. Units depend on the temporal step `dt`.
        
        Keep : np.array
                Boolean array indicating scales to retain.
        
        SAT : int
                Maximum number of modes per scale.
        
        boundaries : {'nearest', 'reflect', 'wrap', 'extrap'}
                Boundary conditions for filtering to avoid edge effects. Refer to:
                https://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html
        
        MODE : {'reduced', 'complete', 'r', 'raw'}
                Mode option for QR factorization, used to enforce orthonormality of the mPOD basis to account for non-ideal filter responses.
        
        dt : float
                Temporal step size between snapshots.
        
        SAVE : bool, default=False
                Whether to save intermediate results to disk.

        load_existing_K : bool, default=True
                If True and MEMORY_SAVING is active, attempts to load an existing correlation matrix K from disk,
                skipping recomputation if possible.

        Returns
        -------
        Phi_M : np.array
                Spatial mPOD modes (spatial structures matrix).
        
        Psi_M : np.array
                Temporal mPOD modes (temporal structures matrix).
        
        Sigma_M : np.array
                Modal amplitudes.
        """
        
        if K_in is None:
                print('Computing correlation matrix D matrix...')
                self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS,
                                        self.MEMORY_SAVING,
                                        self.FOLDER_OUT, self.SAVE_K, D=self.Dstar)

                if self.MEMORY_SAVING:
                        self.K = np.load(self.FOLDER_OUT + '/correlation_matrix/k_matrix.npz')['K']
        else:
                print('Using K matrix provided by the user...')
                self.K = K_in


        print("Computing Temporal Basis...")
        PSI_M = temporal_basis_mPOD(
                K=self.K, Nf=Nf, Ex=Ex, F_V=F_V, Keep=Keep, boundaries=boundaries,
                MODE=MODE, dt=dt, FOLDER_OUT=self.FOLDER_OUT,
                n_Modes=self.n_Modes, MEMORY_SAVING=self.MEMORY_SAVING, SAT=SAT,
                eig_solver=self.eig_solver
        )
        print("Temporal Basis computed.")

        if hasattr(self, 'D'):
                print('Computing spatial modes Phi from D...')
                Phi_M, Psi_M, Sigma_M = spatial_basis_mPOD(
                self.D, PSI_M, N_T=self.N_T, N_PARTITIONS=self.N_PARTITIONS,
                N_S=self.N_S, MEMORY_SAVING=self.MEMORY_SAVING,
                FOLDER_OUT=self.FOLDER_OUT, SAVE=SAVE
                )
        else:
                print('Computing spatial modes Phi from partitions...')
                Phi_M, Psi_M, Sigma_M = spatial_basis_mPOD(
                np.array([1]), PSI_M, N_T=self.N_T,
                N_PARTITIONS=self.N_PARTITIONS, N_S=self.N_S,
                MEMORY_SAVING=self.MEMORY_SAVING,
                FOLDER_OUT=self.FOLDER_OUT, SAVE=SAVE
                )

        print("Spatial modes computed.")

        return Phi_M, Psi_M, Sigma_M

    def DFT(self, F_S, SAVE_DFT=False):
        """
        Computes the Discrete Fourier Transform (DFT) of the dataset.

        For detailed guidance, see the tutorial video:
        https://www.youtube.com/watch?v=8fhupzhAR_M&list=PLEJZLD0-4PeKW6Ze984q08bNz28GTntkR&index=2

        Parameters
        ----------
        F_S : float
                Sampling frequency in Hz.

        SAVE_DFT : bool, default=False
                If True, saves the computed DFT outputs to disk under:
                `self.FOLDER_OUT/MODULO_tmp`.

        Returns
        -------
        Phi_F : np.ndarray
                Spatial DFT modes (spatial structures matrix).
                
        Psi_F : np.ndarray
                Temporal DFT modes (temporal structures matrix).

        Sigma_F : np.ndarray
                Modal amplitudes.
        """
        if self.D is None:
            D = np.load(self.FOLDER_OUT + '/MODULO_tmp/data_matrix/database.npz')['D']
            SAVE_DFT = True
            Phi_F, Psi_F, Sigma_F = dft(self.N_T, F_S, D, self.FOLDER_OUT, SAVE_DFT=SAVE_DFT)

        else:
            Phi_F, Psi_F, Sigma_F = dft(self.N_T, F_S, self.D, self.FOLDER_OUT, SAVE_DFT=SAVE_DFT)

        return Phi_F, Psi_F, Sigma_F

    
    def POD(self, SAVE_T_POD: bool = False, mode: str = 'K'):
        """
        Compute the Proper Orthogonal Decomposition (POD) of a dataset.

        The POD is computed using the snapshot approach, working on the
        temporal correlation matrix.  The eigenvalue solver for this
        matrix is defined in the `eig_solver` attribute of the class.

        Parameters
        ----------
        SAVE_T_POD : bool, optional
                Flag to save time-dependent POD data. Default is False.
        mode : str, optional
                The mode of POD computation. Must be either 'K' or 'svd'.
                'K' (default) uses the snapshot method on the temporal
                correlation matrix.
                'svd' uses the SVD decomposition (full dataset must fit in memory).

        Returns
        -------
        Psi_P : numpy.ndarray
                POD spatial modes.
        Sigma_P : numpy.ndarray
                POD singular values (eigenvalues are Sigma_P**2).
        Phi_P : numpy.ndarray
                POD temporal modes.

        Raises
        ------
        ValueError
                If `mode` is not 'k' or 'svd'.

        Notes
        -----
        A brief recall of the theoretical background of the POD is
        available at https://youtu.be/8fhupzhAR_M
        """
        
        mode = mode.lower()
        assert mode in ('k', 'svd'), "POD mode must be either 'K', temporal correlation matrix, or 'svd'." 

        if mode == 'k':
                
                print('Computing correlation matrix...')
                self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS,
                                        self.MEMORY_SAVING,
                                        self.FOLDER_OUT, self.SAVE_K, D=self.Dstar, weights=self.weights)

                if self.MEMORY_SAVING:
                        self.K = np.load(self.FOLDER_OUT + '/correlation_matrix/k_matrix.npz')['K']

                        print("Computing Temporal Basis...")
                        Psi_P, Sigma_P = Temporal_basis_POD(self.K, SAVE_T_POD,
                                                        self.FOLDER_OUT, self.n_Modes, eig_solver=self.eig_solver)
                        print("Done.")
                        print("Computing Spatial Basis...")

                if self.MEMORY_SAVING:  # if self.D is available:
                        print('Computing Phi from partitions...')
                        Phi_P = Spatial_basis_POD(np.array([1]), N_T=self.N_T,
                                        PSI_P=Psi_P,
                                        Sigma_P=Sigma_P,
                                        MEMORY_SAVING=self.MEMORY_SAVING,
                                        FOLDER_OUT=self.FOLDER_OUT,
                                        N_PARTITIONS=self.N_PARTITIONS)

                else:  # if not, the memory saving is on and D will not be used. We pass a dummy D
                        print('Computing Phi from D...')
                        Phi_P = Spatial_basis_POD(self.D, N_T=self.N_T,
                                                PSI_P=Psi_P,
                                                Sigma_P=Sigma_P,
                                                MEMORY_SAVING=self.MEMORY_SAVING,
                                                FOLDER_OUT=self.FOLDER_OUT,
                                                N_PARTITIONS=self.N_PARTITIONS)
                        print("Done.")
        
        else:  
                if self.MEMORY_SAVING:
                        
                        if self.N_T % self.N_PARTITIONS != 0:
                                tot_blocks_col = self.N_PARTITIONS + 1
                        else:
                                tot_blocks_col = self.N_PARTITIONS

                        # Prepare the D matrix again
                        D = np.zeros((self.N_S, self.N_T))
                        R1 = 0

                        # print(' \n Reloading D from tmp...')
                        for k in tqdm(range(tot_blocks_col)):
                                di = np.load(self.FOLDER_OUT + f"/data_partitions/di_{k + 1}.npz")['di']
                                R2 = R1 + np.shape(di)[1]
                                D[:, R1:R2] = di
                                R1 = R2

                        # Now that we have D back, we can proceed with the SVD approach
                        Phi_P, Psi_P, Sigma_P = switch_svds(D, self.n_Modes, self.svd_solver)

                else:  # self.MEMORY_SAVING:
                        Phi_P, Psi_P, Sigma_P = switch_svds(self.D, self.n_Modes, self.svd_solver)    

        return Phi_P, Psi_P, Sigma_P

    
    def DMD(self, SAVE_T_DMD: bool = True, F_S: float = 1.0):
        """
        Compute the Dynamic Mode Decomposition (DMD) of the dataset.

        This implementation follows the algorithm in Tu et al. (2014) [1]_, which is
        essentially the same as Penland (1996) [2]_.  For
        additional low-level details see v1 of Mendez et al. (2020) [3]_.

        Parameters
        ----------
        SAVE_T_DMD : bool, optional
                If True, save time-dependent DMD results to disk. Default is True.
        F_S : float, optional
                Sampling frequency in Hz. Default is 1.0.

        Returns
        -------
        Phi_D : numpy.ndarray
                Complex DMD modes.
        Lambda_D : numpy.ndarray
                Complex eigenvalues of the reduced-order propagator.
        freqs : numpy.ndarray
                Frequencies (Hz) associated with each DMD mode.
        a0s : numpy.ndarray
                Initial amplitudes (coefficients) of the DMD modes.

        References
        ----------
        .. [1] https://arxiv.org/abs/1312.0041
        .. [2] https://www.sciencedirect.com/science/article/pii/0167278996001248
        .. [3] https://arxiv.org/abs/2001.01971
        """

        # If Memory saving is active, we must load back the data
        if self.MEMORY_SAVING:
            if self.N_T % self.N_PARTITIONS != 0:
                tot_blocks_col = self.N_PARTITIONS + 1
            else:
                tot_blocks_col = self.N_PARTITIONS

            # Prepare the D matrix again
            D = np.zeros((self.N_S, self.N_T))
            R1 = 0

            # print(' \n Reloading D from tmp...')
            for k in tqdm(range(tot_blocks_col)):
                di = np.load(self.FOLDER_OUT + f"/data_partitions/di_{k + 1}.npz")['di']
                R2 = R1 + np.shape(di)[1]
                D[:, R1:R2] = di
                R1 = R2

            # Compute the DMD
            Phi_D, Lambda, freqs, a0s = dmd_s(D[:, 0:self.N_T - 1],
                                              D[:, 1:self.N_T], self.n_Modes, F_S, svd_solver=self.svd_solver)

        else:
            Phi_D, Lambda, freqs, a0s = dmd_s(self.D[:, 0:self.N_T - 1],
                                              self.D[:, 1:self.N_T], self.n_Modes, F_S, SAVE_T_DMD=SAVE_T_DMD,
                                              svd_solver=self.svd_solver, FOLDER_OUT=self.FOLDER_OUT)

        return Phi_D, Lambda, freqs, a0s

    def SPOD(
        self,
        mode: str,
        F_S: float,
        n_Modes: int = 10,
        SAVE_SPOD: bool = True,
        **kwargs
    ):
        """
        Unified Spectral POD interface.

        Parameters
        ----------
        mode : {'sieber', 'towne'}
            Which SPOD algorithm to run.
        F_S : float
            Sampling frequency [Hz].
        n_Modes : int, optional
            Number of modes to compute, by default 10.
        SAVE_SPOD : bool, optional
            Whether to save outputs, by default True.
        **kwargs
            For mode='sieber', accepts:
              - N_O (int): semi-order of the diagonal filter
              - f_c (float): cutoff frequency
            For mode='towne', accepts:
              - L_B (int): block length
              - O_B (int): block overlap
              - n_processes (int): number of parallel workers

        Returns
        -------
        Phi : ndarray
            Spatial modes.
        Sigma : ndarray
            Modal amplitudes.
        Aux : tuple
            Additional outputs.
        """
        mode = mode.lower()
        if mode == 'sieber':
            N_O = kwargs.pop('N_O', 100)
            f_c = kwargs.pop('f_c', 0.3)
            
            return self.compute_SPOD_s(
                F_S=F_S,
                N_O=N_O,
                f_c=f_c,
                n_Modes=n_Modes,
                SAVE_SPOD=SAVE_SPOD
            )

        elif mode == 'towne':
            L_B = kwargs.pop('L_B', 500)
            O_B = kwargs.pop('O_B', 250)
            n_processes = kwargs.pop('n_processes', 1)

            # Load or reuse data matrix
            
            if self.D is None:
                D = np.load(f"{self.FOLDER_OUT}/MODULO_tmp/data_matrix/database.npz")['D']
            else:
                D = self.D

            # Segment and FFT
            D_hat, freqs_pos = segment_and_fft(
                D=D,
                F_S=F_S,
                L_B=L_B,
                O_B=O_B,
                n_processes=n_processes
                )

            return self.compute_SPOD_t(D_hat=D_hat, 
                                       freq_pos=freqs_pos,
                                        n_modes=n_Modes, 
                                        SAVE_SPOD=SAVE_SPOD, 
                                        svd_solver=self.svd_solver,
                                        n_processes=n_processes)
            
        else:
                raise ValueError("mode must be 'sieber' or 'towne'")


    def compute_SPOD_t(self, D_hat, freq_pos, n_Modes=10, SAVE_SPOD=True, svd_solver=None, 
                       n_processes=1):
        """
        Compute the CSD-based Spectral POD (Towne et al.) from a precomputed FFT tensor.

        Parameters
        ----------
        D_hat : ndarray, shape (n_s, n_freqs, n_blocks)
                FFT of each block, only nonnegative frequencies retained.
        freq_pos : ndarray, shape (n_freqs,)
                Positive frequency values (Hz) corresponding to D_hat’s second axis.
        n_Modes : int, optional
                Number of SPOD modes per frequency bin. Default is 10.
        SAVE_SPOD : bool, optional
                If True, save outputs under `self.FOLDER_OUT/MODULO_tmp`. Default is True.
        svd_solver : str or None, optional
                Which SVD solver to use (passed to `switch_svds`), by default None.
        n_processes : int, optional
                Number of parallel workers for the POD step. Default is 1 (serial).

        Returns
        -------
        Phi_SP : ndarray, shape (n_s, n_Modes, n_freqs)
                Spatial SPOD modes at each positive frequency.
        Sigma_SP : ndarray, shape (n_Modes, n_freqs)
                Modal energies per frequency bin.
        freq_pos : ndarray, shape (n_freqs,)
                The positive frequency vector (Hz), returned unchanged.
        """
        # Perform the POD (parallel if requested)
                # received D_hat_f, this is now just a POD on the transversal direction of the tensor,
        # e.g. the frequency domain. 
        n_freqs = len(freq_pos)
        
        # also here we can parallelize
        Phi_SP, Sigma_SP = pod_from_dhat(D_hat=D_hat, n_modes=n_Modes, n_freqs=n_freqs, 
                                         svd_solver=self.svd_solver, n_processes=n_processes)
        
        # Optionally save the results
        if SAVE_SPOD:
                np.savez(
                f"{self.FOLDER_OUT}/MODULO_tmp/spod_towne.npz",
                Phi=Phi_SP,
                Sigma=Sigma_SP,
                freqs=freq_pos
                )

        return Phi_SP, Sigma_SP, freq_pos



    def compute_SPOD_s(self, N_O=100, f_c=0.3, n_Modes=10, SAVE_SPOD=True):
        """
        Compute the filtered‐covariance Spectral POD (Sieber _et al._) of your data.

        This implementation follows Sieber et al. (2016), which applies a zero‐phase
        diagonal filter to the time‐lag covariance and then performs a single POD
        on the filtered covariance matrix.

        Parameters
        ----------
        N_O : int, optional
                Semi‐order of the diagonal FIR filter. The true filter length is
                2*N_O+1, by default 100.
        f_c : float, optional
                Normalized cutoff frequency of the diagonal filter (0 < f_c < 0.5),
                by default 0.3.
        n_Modes : int, optional
                Number of SPOD modes to compute, by default 10.
        SAVE_SPOD : bool, optional
                If True, save output under `self.FOLDER_OUT/MODULO_tmp`, by default True.

        Returns
        -------
        Phi_sP : numpy.ndarray, shape (n_S, n_Modes)
                Spatial SPOD modes.
        Psi_sP : numpy.ndarray, shape (n_t, n_Modes)
                Temporal SPOD modes (filtered).
        Sigma_sP : numpy.ndarray, shape (n_Modes,)
                Modal energies (eigenvalues of the filtered covariance).
        """
        if self.D is None:
                
                D = np.load(self.FOLDER_OUT + '/MODULO_tmp/data_matrix/database.npz')['D']

        self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS, self.MEMORY_SAVING, self.FOLDER_OUT, self.SAVE_K, D=D)
                
        # additional step: diagonal spectral filter of K 
        K_F = spectral_filter(self.K, N_o=N_O, f_c=f_c)
        
        # and then proceed with normal POD procedure
        Psi_P, Sigma_P = Temporal_basis_POD(K_F, SAVE_SPOD, self.FOLDER_OUT, n_Modes)
        
        # but with a normalization aspect to handle the non-orthogonality of the SPOD modes
        Phi_P = Spatial_basis_POD(D, N_T=self.K.shape[0], 
                                        PSI_P=Psi_P, Sigma_P=Sigma_P,
                                MEMORY_SAVING=self.MEMORY_SAVING, 
                                FOLDER_OUT=self.FOLDER_OUT,
                                N_PARTITIONS=self.N_PARTITIONS,rescale=True)
                        

        return Phi_P, Psi_P, Sigma_P

    
    def kPOD(self, M_DIST=[1, 10], 
             k_m=0.1, cent=True,
                n_Modes=10, 
                alpha=1e-6, 
                metric='rbf', 
                K_out=False, SAVE_KPOD=False):
        """
        This function implements the kernel PCA as described in the VKI course https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552/-/online-on-site-hands-on-machine-learning-for-fluid-dynamics-2023

        The computation of the kernel function is carried out as in https://arxiv.org/pdf/2208.07746.pdf.


        :param M_DIST: array,
                position of the two snapshots that will be considered to
                estimate the minimal k. They should be the most different ones.
        :param k_m: float,
                minimum value for the kernelized correlation
        :param alpha: float
                regularization for K_zeta
        :param cent: bool,
                if True, the matrix K is centered. Else it is not
        :param n_Modes: float,
               number of modes to be computed
        :param metric: string,
               This identifies the metric for the kernel matrix. It is a wrapper to 'pairwise_kernels' from sklearn.metrics.pairwise
               Note that different metrics would need different set of parameters. For the moment, only rbf was tested; use any other option at your peril !
        :param K_out: bool,
               If true, the matrix K is also exported as a fourth output.
        :return Psi_xi: np.array
               kPOD's Psis
        :return Sigma_xi: np.array
               kPOD's Sigmas.
        :return Phi_xi: np.array
               kPOD's Phis
        :return K_zeta: np.array
               Kernel Function from which the decomposition is computed.
               (exported only if K_out=True)


        """
        if self.D is None:
            D = np.load(self.FOLDER_OUT + '/MODULO_tmp/data_matrix/database.npz')['D']
        else:
            D = self.D

        # Compute Eucledean distances
        i, j = M_DIST
        
        M_ij = np.linalg.norm(D[:, i] - D[:, j]) ** 2

        K_r = kernelized_K(D=D, M_ij=M_ij, k_m=k_m, metric=metric, cent=cent, alpha=alpha)

        Psi_xi, Sigma_xi = Temporal_basis_POD(K=K_r, n_Modes=n_Modes, eig_solver='eigh')

        PHI_xi_SIGMA_xi = D @ Psi_xi
        
        Sigma_xi = np.linalg.norm(PHI_xi_SIGMA_xi, axis=0) # (R,)
        Phi_xi   = PHI_xi_SIGMA_xi / Sigma_xi[None, :] # (n_s, R)
        
        sorted_idx = np.argsort(-Sigma_xi)

        Phi_xi = Phi_xi[:, sorted_idx]  # Sorted Spatial Structures Matrix
        Psi_xi = Psi_xi[:, sorted_idx]  # Sorted Temporal Structures Matrix
        Sigma_xi = Sigma_xi[sorted_idx]

        if K_out:
            return Phi_xi, Psi_xi, Sigma_xi, K_r
        else:
            return Phi_xi, Psi_xi, Sigma_xi


        