# Functional ones:
import os
import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import pairwise_kernels
# To have fancy loading bar
from tqdm import tqdm

# All the functions from the modulo package 
from modulo_vki.core._dft import dft_fit
from modulo_vki.core._dmd_s import dmd_s
from modulo_vki.core._k_matrix import CorrelationMatrix
from modulo_vki.core._mpod_space import spatial_basis_mPOD
from modulo_vki.core._mpod_time import temporal_basis_mPOD
from modulo_vki.core._pod_space import Spatial_basis_POD
from modulo_vki.core._pod_time import Temporal_basis_POD
from modulo_vki.core._spod_s import compute_SPOD_s
from modulo_vki.core._spod_t import compute_SPOD_t
from modulo_vki.utils._utils import switch_svds

from modulo_vki.utils.read_db import ReadData

class ModuloVKI:
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
    and in time. The extension to non-uniformly sampled data will be included in future releases.


    """

    def __init__(self, data: np.array,
                 N_PARTITIONS: int = 1,
                 FOLDER_OUT='./',
                 SAVE_K: bool = False,
                 N_T: int = 100,
                 N_S: int = 200,
                 n_Modes: int = 10,
                 dtype: str = 'float32',
                 eig_solver: str = 'eigh',
                 svd_solver: str = 'svd_sklearn_truncated',
                 weights: np.array = np.array([])):
        """
        This function initializes the main parameters needed by MODULO.

        Attributes:

        :param data: This is the data matrix to factorize. It is a np.array with
               shape ((N_S, N_T)). If the data has not yet been prepared in the form of a np.array,
               the method ReadData in MODULO can be used (see ReadData). If the memory saving is active (N_PARTITIONS >1), the folder with partitions should be prepared.
               If the memory saving is active, this entry = None. The data matrix is assumed to big to be saved and the

        :param N_PARTITIONS: If memory saving feature is active, this parameter sets the number of partitions
               that will be used to store the data matrices during the computations.

        :param FOLDER_OUT: Folder in which the output will be stored.The output includes the matrices Phi, Sigma and Psi (optional) and temporary files
               used for some of the calculations (e.g.: for memory saving).

        :param  SAVE_K:  A flag deciding if the matrix will be stored in the disk (in FOLDER_OUT/correlation_matrix) or not.
            Default option is 'False'.

        :param N_T: Number of time steps, must be given when N_PARTITIONS >1

        :param N_S: Number of grid points, must be given when N_PARTITIONS >1

        :param n_Modes: Number of Modes to be computed

        :param dtype: Cast "data" with type dtype

        :param eig_solver: Numerical solver to compute the eigen values

        :param svd_solver: Numerical solver to compute the Single Value Decomposition

        :param weights: weight vector [w_i,....,w_{N_s}] where w_i = area_cell_i/area_grid
               Only needed if grid is non-uniform.


        """

        print("MODULO (MODal mULtiscale pOd) is a software developed at the von Karman Institute to perform "
              "data driven modal decomposition of numerical and experimental data. \n")

        if not isinstance(data, np.ndarray) and N_PARTITIONS == 1:
            raise TypeError(
                "Please check that your database is in an numpy array format. If D=None, then you must have memory saving (N_PARTITIONS>1)")

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

        # Load and applied the weights to the D matrix
        if weights.size != 0:
            if len(weights) == self.N_S:
                print("The weights you have input have the size of the columns of D \n"
                      "MODULO has considered that you have already duplicated the dimensions of the weights "
                      "to match the dimensions of the D columns \n")
                self.weights = weights
            elif 2 * len(weights) == self.N_S:  # 2D computation only
                self.weights = np.concatenate((weights, weights))
                print("Modulo assumes you have a 2D domain and has duplicated the weight "
                      "array to match the size of the D columns \n")
                print(weights)
            else:
                raise AttributeError("Make sure the size of the weight array is twice smaller than the size of D")
            # Dstar is used to compute the K matrix
            if isinstance(data, np.ndarray):
                # Apply the weights only if D exist.
                # If not (i.e. N_partitions >1), weights are applied in _k_matrix.py when loading partitions of D
                self.Dstar = np.transpose(np.transpose(self.D) * np.sqrt(self.weights))
            else:
                self.Dstar = None
        else:
            print("Modulo assumes you have a uniform grid. "
                  "If not, please give the weights as parameters of MODULO!")
            self.weights = weights
            self.Dstar = self.D

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

        # if self.D is None:
        #     D = np.load(self.FOLDER_OUT + '/MODULO_tmp/data_matrix/database.npz')['D']
        #     SAVE_SPOD = True
        #     # TODO : Lorenzo check this stuff
        # else:
        #     D = self.D
        #
        # n_s = self.N_S  # Repeat variable for debugging compatibility
        # n_t = self.N_T
        #
        # print('Computing Correlation Matrix \n')
        #
        # # The first step is the same as the POD: we compute the correlation matrix
        # K = CorrelationMatrix(self.N_T, self.N_PARTITIONS, self.MEMORY_SAVING,
        #                       self.FOLDER_OUT, D=self.D)
        #
        # # 1. Initialize the extended
        # K_e = np.zeros((n_t + 2 * N_o, n_t + 2 * N_o))
        # # From which we clearly know that:
        # K_e[N_o:n_t + N_o, N_o:n_t + N_o] = K
        #
        # # 2. We fill the edges ( a bit of repetition but ok.. )
        #
        # # Row-wise, Upper part
        # for i in range(0, N_o):
        #     K_e[i, i:i + n_t] = K[0, :]
        #
        # # Row-wise, bottom part
        # for i in range(N_o + n_t, n_t + 2 * N_o):
        #     K_e[i, i - n_t + 1:i + 1] = K[-1, :]
        #
        #     # Column-wise, left part
        # for j in range(0, N_o):
        #     K_e[j:j + n_t, j] = K[:, 0]
        #
        #     # Column-wise, right part
        # for j in range(N_o + n_t, 2 * N_o + n_t):
        #     K_e[j - n_t + 1:j + 1, j] = K[:, -1]
        #
        # # Now you create the diagonal kernel in 2D
        # h_f = firwin(N_o, f_c)  # Kernel in 1D
        # # This is also something that must be put in a separate file:
        # # To cancel the phase lag we make this non-causal with a symmetric
        # # shift, hence with zero padding as equal as possible on both sides
        # n_padd_l = round((n_t - N_o) / 2);
        # n_padd_r = n_t - N_o - n_padd_l
        #
        # h_f_pad = np.pad(h_f, (n_padd_l, n_padd_r))  # symmetrically padded kernel in 1D
        # h_f_2 = np.diag(h_f_pad)
        #
        # # Finally the filtered K is just
        # K_F = signal.fftconvolve(K_e, h_f_2, mode='same')[N_o:n_t + N_o, N_o:n_t + N_o]
        # # plt.plot(np.diag(K),'b--'); plt.plot(np.diag(K_F_e),'r')
        #
        # # From now on it's just POD:
        # Psi_P, Sigma_P = Temporal_basis_POD(K_F, SAVE_SPOD,
        #                                     self.FOLDER_OUT, self.n_Modes)
        #
        # Phi_P = Spatial_basis_POD(self.D, N_T=self.N_T, PSI_P=Psi_P, Sigma_P=Sigma_P,
        #                           MEMORY_SAVING=self.MEMORY_SAVING, FOLDER_OUT=self.FOLDER_OUT,
        #                           N_PARTITIONS=self.N_PARTITIONS)

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
