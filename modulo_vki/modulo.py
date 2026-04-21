import os
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
from scipy.fft import fft, fftfreq
from scipy.linalg import eigh, svd
from randomized_svd import rsvd


from modulo_vki.core._k_matrix import CorrelationMatrix, spectral_filter, kernelized_K, compute_K_F
from modulo_vki.core.temporal_structures import dft, temporal_basis_mPOD, Temporal_basis_POD
from modulo_vki.core.spatial_structures import Spatial_basis_POD, spatial_basis_mPOD
from modulo_vki.core._dmd_s import dmd_s

from modulo_vki.core.utils import segment_and_fft, pod_from_dhat, apply_weights, switch_svds, sortScalesByEnergy, taperBlockBounds
from sklearn.metrics.pairwise import pairwise_kernels


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

            if data is not None:
                raise ValueError("The memory saving option is active, so MODULO cannot be initialized with the full snapshot matrix. Use 'ReadData' routines to process the data in chunks instead.")

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
            Phi_F, Psi_F, Sigma_F = dft(self.N_T, F_S, self.D,
                                        self.FOLDER_OUT, SAVE_DFT=SAVE_DFT)

        return Phi_F, Psi_F, Sigma_F

    def POD(self, SAVE_T_POD: bool = False, mode: str = 'K',verbose=True):
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
        Phi_P : numpy.ndarray
                POD temporal modes.
        Psi_P : numpy.ndarray
                POD spatial modes.
        Sigma_P : numpy.ndarray
                POD singular values (eigenvalues are Sigma_P**2).
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
                if verbose:
                    print('Computing correlation matrix...')
                self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS,
                                        self.MEMORY_SAVING,
                                        self.FOLDER_OUT, self.SAVE_K,
                                        D=self.Dstar, weights=self.weights,
                                        verbose=verbose)

                if self.MEMORY_SAVING:
                        self.K = np.load(self.FOLDER_OUT + '/correlation_matrix/k_matrix.npz')['K']
                if verbose:
                    print("Computing Temporal Basis...")
                Psi_P, Sigma_P = Temporal_basis_POD(self.K, SAVE_T_POD,
                                                self.FOLDER_OUT, self.n_Modes, eig_solver=self.eig_solver,verbose=verbose)

                if verbose:
                    print("Done.")
                    print("Computing Spatial Basis...")

                if self.MEMORY_SAVING:  # if self.D is available:
                        if verbose:
                            print('Computing Phi from partitions...')
                        Phi_P = Spatial_basis_POD(np.array([1]), N_T=self.N_T,
                                        PSI_P=Psi_P,
                                        Sigma_P=Sigma_P,
                                        MEMORY_SAVING=self.MEMORY_SAVING,
                                        FOLDER_OUT=self.FOLDER_OUT,
                                        N_PARTITIONS=self.N_PARTITIONS,
                                        verbose=verbose)

                else:  # if not, the memory saving is on and D will not be used. We pass a dummy D
                        if verbose:
                            print('Computing Phi from D...')
                        Phi_P = Spatial_basis_POD(self.D, N_T=self.N_T,
                                                PSI_P=Psi_P,
                                                Sigma_P=Sigma_P,
                                                MEMORY_SAVING=self.MEMORY_SAVING,
                                                FOLDER_OUT=self.FOLDER_OUT,
                                                N_PARTITIONS=self.N_PARTITIONS,
                                                verbose=verbose)
                        if verbose:
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


    def mPOD(self, Nf, Ex, F_V, Keep, SAT, boundaries, MODE, dt, SAVE=False, K_in=None, Sigma_type='accurate', conv_type: str = '1d', verbose=True):
        """
        Multi-Scale Proper Orthogonal Decomposition (mPOD) of a signal.

        Parameters
        ----------
        Nf : np.array
                Orders of the FIR filters used to isolate each scale. Must be of size len(F_V) + 1.

        Ex : int
                Extension length at the boundaries to impose boundary conditions (must be at least as large as Nf).

        F_V : np.array
                Frequency splitting vector, containing the cutoff frequencies for each scale. Units depend on the temporal step `dt`.

        Keep : np.array
                Boolean array indicating scales to retain. Must be of size len(F_V) + 1.

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

        K_in : np.array, default = none
                K matrix. If none, compute it with D.

        Sigma_type : {'accurate', 'fast'}
                If accurate, recompute the Sigmas after QR polishing. Slightly slower than the fast option in which the Sigmas are not recomputed.

        conv_type : {'1d', '2d'}
            If 1d, compute Kf applying 1d FIR filters to the columns and then rows of the extended K.
            More robust against windowing effects but more expensive (useful for modes that are slow compared to the observation time).
            If 2d, compute Kf applying a 2d FIR filter on the extended K.

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
                if verbose:
                    print('Computing correlation matrix D matrix...')
                self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS,
                                        self.MEMORY_SAVING,
                                        self.FOLDER_OUT, self.SAVE_K, D=self.Dstar,
                                        verbose=verbose)

                if self.MEMORY_SAVING:
                        self.K = np.load(self.FOLDER_OUT + '/correlation_matrix/k_matrix.npz')['K']
        else:
                if verbose:
                    print('Using K matrix provided by the user...')
                self.K = K_in

        if verbose:
            print("Computing Temporal Basis...")
        PSI_M,SIGMA_M = temporal_basis_mPOD(
                K=self.K, Nf=Nf, Ex=Ex, F_V=F_V, Keep=Keep, boundaries=boundaries,
                MODE=MODE, dt=dt, FOLDER_OUT=self.FOLDER_OUT,
                n_Modes=self.n_Modes, MEMORY_SAVING=self.MEMORY_SAVING, SAT=SAT,
                eig_solver=self.eig_solver, conv_type=conv_type, verbose=verbose
        )
        if verbose:
            print("Temporal Basis computed.")

        if hasattr(self, 'D'):
                if verbose:
                    print('Computing spatial modes Phi from D...')
                Phi_M, Psi_M, Sigma_M = spatial_basis_mPOD(
                self.D, PSI_M, N_T=self.N_T, N_PARTITIONS=self.N_PARTITIONS,
                N_S=self.N_S, MEMORY_SAVING=self.MEMORY_SAVING,
                FOLDER_OUT=self.FOLDER_OUT, SAVE=SAVE, SIGMA_TYPE=Sigma_type, SIGMA_M=SIGMA_M
                )
        else:
                if verbose:
                    print('Computing spatial modes Phi from partitions...')
                Phi_M, Psi_M, Sigma_M = spatial_basis_mPOD(
                np.array([1]), PSI_M, N_T=self.N_T,
                N_PARTITIONS=self.N_PARTITIONS, N_S=self.N_S,
                MEMORY_SAVING=self.MEMORY_SAVING,
                FOLDER_OUT=self.FOLDER_OUT, SAVE=SAVE,SIGMA_TYPE=Sigma_type, SIGMA_M=SIGMA_M
                )
        if verbose:
            print("Spatial modes computed.")

        return Phi_M, Psi_M, Sigma_M


    def fastmPOD(self, F_V, fs, Keep = None, winType = "hann", taper = None, mode = "fullK", GThresh = -1, ncpus = os.cpu_count(), 
                 oversampling = 10, n_iter = -1, useFortran = False):

        """
        Fast spectral variant of Multi-Scale Proper Orthogonal Decomposition (mPOD) of a signal.

        Parameters
        ----------
        F_V : list of float in ascending order, >0 and <f_Nyq
            Frequency splitting vector. Must NOT include 0 and f_Nyq.

        fs : float > 0
            Sampling frequency of the dataset.

        Keep : list of bool, optional
            Which frequency bands to keep. No of elements is len(F_V)+1.
            Default = None (initialize to keep all scales)

        winType : str, optional
            Type of window to be used for creating tapering functions. Simple smooth windows are recommended.
            Default = "hann".

        taper : list of float, optional
            Tapering width for each scale in Hz.
            Default = None (no taper).

        mode : str, optional
            Computation mode. Has 4 options: "fullK", "bandK", "fullSVD", "randSVD".
                "fullK" = compute full K = D.T @ D, transform it to frequency domain as K_F and do all computations
                        based on K_F. Good when Ns >> Nt and kept bands are reasonably broad.
                "bandK" = compute D_hat as FFT of D and from D_hat compute K_Fsc on-the-fly for each scale, without computing
                        global K_F. This is usually a good pick when Ns ~ Nt and only a few narrow bands are kept.
                "fullSVD" = compute exact SVD of the part of D_hat belonging to current scale and select appropriate amount of modes.
                            This option skips building K of any sort. Usage similar to "bandK", should be a bit less memory-hungry.
                            BandK is however still the prefered option for those cases.
                "randSVD" = compute randomized SVD (truncated to nModes) of the part of D_hat belonging to current scale. This is
                            by far the fastest option when Ns < Nt, but it is approximate and can do badly in cases when lower
                            importance modes should be captured. Should not be used when band matrices are ill-suited for randomized SVD,
                            e.g., when the singular values of the band matrices do not decay fast enough or when the singular values
                            are not well separated. Gives the worst performance of all methods for Ns >> Nt.
            Default = "fullK".

        GThresh : float >= 0, optional
            When to start applying Gershgorin circle theorem to estimate number of signifficant modes in the scale.
            Only for K-based approaches. Gershgorin estimation is applied, when total energy of the scale in question is
            less than or equal to GThresh*lamMin where lamMin is the smallest signifficant eigenvalue at a time.
            For good performance must be set to "reasonable" value.
            Too low value: Increased computing cost due to computation of scales that can be safely skipped
                        or due to computing more eigenvalues than needed in the scale.
            Too high value: increased cost due to computing Gershgorin at scales that can not benefit from it.
            Special value: -1 = set to default value.
            Set to <1 to disable.
            Default -1 (set to nModes/2 on initialization).

        ncpus : int >0, optional
            Number of CPUs available for FFT parallelization. Set to number of physical cores /
            number of physical cores - 1 for best performance.
            Default = os.cpu_count()

        oversampling : int >0, optional
            Oversampling parameter for randomized SVD. Higher values can increase accuracy of randomized SVD,
            but also increase computational cost. This is the first parameter to tweak when randomized SVD
            does not give good results. For "randSVD" mode only, for other modes has no effect.
            Default = 10, as in scikit-learn implementation of randomized SVD.

        n_iter : int >0, optional
            Power iteration parameter for randomized SVD. Higher values can increase accuracy of randomized SVD,
            especially when singular values decay slowly, but also increase computational cost. Change only when
            changing oversampling does not improve the results. Set to -1 to let the algorithm set it on-the-fly
            based on the number of modes and size of the processed band matrix. For "randSVD" mode only,
            for other modes has no effect.
            Default = -1 (Set value on-the fly, in the same way as in scikit-learn).

        useFortran : bool, optional
            Whether to use Fortran implementations for assembly of correlation matrices. Can give up to 2x speedup
            for assembly of K (mode = 'fullK') and up to 2.7x speedup for assembly of K_Fsc (mode = 'bandK') under
            the assumption of FLOP-bound computation. For memory-bound computation this option has little to no effect.
            Default = False.

        Returns
        -------
        phi : 2D np.ndarray of float
            Spatial modes in descending signifficancy order. Column-wise (i-th column = i-th mode).

        psi : 2D np.ndarray of float
            Temporal modes in descending signifficancy order. Column-wise (i-th column = i-th mode).

        sigTot : 1D np.ndarray of float >0
            Mode amplitudes in descending signifficancy order.

        Raises
        ------
        AssertionError
            When incorrect entries are provided.

        ValueError
            When invalid computation mode or invalid n_iter value is provided.

        See documentation of the called functions to see which errors you may get from them.

        See numpy and scipy documentation of the functions used in the code to see,
        which other error messages you may get.
        """

        print("Computing mPOD using fast mPOD algorithm ...")

        nModes = self.n_Modes

        # Input type checks
        assert isinstance(F_V,list), "F_V must be a list. Got %s of type %s."%(F_V, type(F_V))
        assert fs > 0, "Invalid sampling frequency, must be a positive number. Got %s."%fs
        assert (isinstance(nModes,int) and nModes > 0 and nModes < np.min([self.D.shape[0],self.D.shape[1]])), "Invalid number of modes specified. Parameter 'nModes' must be of type 'int' and 0 < nModes < min(rows,cols) of D"
        assert isinstance(Keep,list) or Keep is None, "Keep variable must be a list"
        assert isinstance(winType,str), "winType variable must be a string"
        assert isinstance(taper,list) or taper is None, "taper variable must be a list"
        assert mode in ["fullK", "bandK", "fullSVD", "randSVD"], "Illegal computation mode. See the documentation for supported entries. Recieved %s"%mode
        assert oversampling >= 0 and isinstance(oversampling,int), "Invalid oversampling parameter for randomized SVD. Must be int >= 0."
        assert n_iter >= -1 and isinstance(n_iter,int), "Invalid n_iter for randomized SVD. Must be int >= 0 for manual setting or -1 for auto-setting."

        # Prepare F_V, Keep, Nf, GThresh
        F_V = [0] + F_V + [fs/2]
        F_V = np.array(F_V)
        if Keep is None:
            # The algorithm is very efficient, so we can allow the user to be a bit dumb / ignorant
            print(f"\nWARNING: Scales to keep not specified, defaulting to keeping all scales \n")
            Keep = [1]*(F_V.size-1)
        Keep = np.array(Keep)
        if GThresh == -1:
            GThresh = nModes/2
            print(f"Gershgorin threshold not specified, defaulting to {GThresh} (nModes/2)")
        if taper is None:
            taper = [-1]*(F_V.size-1)
        taper = np.array(taper)

        # Check input validity
        assert (np.diff(F_V) > 0).all(), "Entries of F_V outside allowed range or in incorrect order."
        assert Keep.size == F_V.size-1, "Length of 'Keep' variable must be len(F_V)+1."
        assert taper.size == F_V.size-1, "Length of 'taper' variable must be len(F_V)+1."
        assert GThresh >= 0, "Invalid 'GThresh' value, must be >=0"


        # Precompute K_F if mode = "fullK", otherwise compute D_hat
        print("Transforming data to spectral domain ...")
        #startSw = time()
        if mode == "fullK":     # compute_K_F manages Fortran import and loading internally
            K_F, freq, Nt = compute_K_F(self.D,fs,ncpus,useFortran)     # O(Ns*Nt**2) for K from D, O(Nt**2*log(Nt)) for K_F from K
        else:
            Nt = self.D.shape[1]
            freq = fftfreq(Nt,1/fs)
            D_hat = fft(self.D,axis=1,norm='ortho',workers=ncpus)  # O(Ns*Nt*log(Nt))
        #durSw = time() - startSw
        #print("Done in %.3f s."%durSw)

        taperWid = np.round(taper*Nt/fs).astype(int)  # Convert tapering width from Hz to number of frequency bins

        # Sort scales by decreasing energy
        scaleOrder, E, f, indTot, scalesToKeep = sortScalesByEnergy(F_V, Keep, freq, mode, K_F if mode == "fullK" else D_hat)
        noOfScales = Keep.size
        Etot = 0.01*np.sum(E)   # Prepare energy to print scale E content in %

        # Initialize Lambda and Psi matrices as None
        lamTot = None
        eigvTot = None

        # Loop over scales in decreasing energy order
        #startLoop = time()
        print("Looping through scales in decreasing energy order ... \n")
        lamMin = 0
        for k in scaleOrder:

            print("Processing band %d/%d (%.1f Hz - %.1f Hz), %.1f %% of resolved energy in %d bins ..."%(scalesToKeep[k]+1,noOfScales,f[k,0],f[k,1],E[k]/Etot,indTot[k].size))

            """
            NOTE:
            Since K_Fsc is hermitian symmetric, eigendecomposition K_Fsc = V@Lambda@V.H is a unitary transform,
            and thus preserves the sum along the diagonal, i.e. the trace. From physical constraints,
            Lambda_i >= 0 and < infty for all i, thus we can estimate, whether the scale contains enough energy
            for its modes to be signifficant. In the case of precomputed K_F, this can be done in O(N) time
            by comparing tr(K_Fsc) to the smallest of the nModes significant eigvals at the time. This allows us
            to skip the eigenvalue computation for scales that do not contain enough energy, which has complexity
            of O(N**3). In the case of the on-the-fly computation of K_Fsc, this saves even more time,
            since K_Fsc computation is O(Ns*N**2), although estimating the trace indirectly from Frobenius norm of
            D_hat is more expensive at O(Ns*N). The computation of scale energies (traces) was moved outside of
            this loop to 'sortScalesByEnergy()' to allow for energy-wise scale sorting.
            """

            # Check energy in scale, stop computation when not enough energy is in the scale,
            # since all following scales have even less energy
            if E[k] <= lamMin:
                print(" -> Band %d does not contain enough energy \n"%(k+1))
                print("Energy limit reached, exiting the loop")
                print(" -> All unprocessed bands have even less energy because of energy sorting.\n")
                break


            # Get indices of frequencies in current scale
            indices = indTot[k]

            # Optional tapering
            if taperWid[scalesToKeep[k]] > 0:
                print(" -> Computing smoothing mask of width %d bins"%(taperWid[scalesToKeep[k]]))
                mask1D = taperBlockBounds(indices, winType=winType, taperSize=taperWid[scalesToKeep[k]], Nt=Nt, fBounds=f[k,:], fs=fs)

            # number of frequencies in the current scale
            N = indices.size

            if mode in ["fullK", "bandK"]:      # K-based approaches
                if mode == "fullK":
                    print(" -> Using precomputed K_F")
                    # Extract submatrix of K_F corresponding to current scale
                    K_Fsc = K_F[indices,:]      # O(1) -- just creating a view of K_F, no copying
                    K_Fsc = K_Fsc[:,indices]    # O(1)
                else:
                    print(" -> Computing K_Fsc on the fly")
                    # Assemble K_Fsc in place
                    if useFortran:
                        from modulo_vki.fortran import load_sym_matmul
                        sm = load_sym_matmul()
                        if f[k,0] == 0:
                            K_Fsc = sm.sym_routines.compute_persym_aha_dc(np.asfortranarray(D_hat[:,indices].copy()), D_hat.shape[0], N)
                        elif f[k,1] == fs/2 and Nt%2 == 0:
                            K_Fsc = sm.sym_routines.compute_persym_aha_nyq(np.asfortranarray(D_hat[:,indices].copy()), D_hat.shape[0], N)
                        else:
                            K_Fsc = sm.sym_routines.compute_persym_aha_band(np.asfortranarray(D_hat[:,indices].copy()), D_hat.shape[0], N)
                    else:
                        K_Fsc = np.linalg.matmul(D_hat[:,indices].conj().T, D_hat[:,indices])   # O(Ns*N**2), but with complex-valued matrix


                if taperWid[scalesToKeep[k]] > 0:
                    print(" -> Applying smoothing to K_Fsc")
                    K_Fsc = K_Fsc * np.outer(mask1D,mask1D)    # Apply smoothing mask to K_Fsc -- O(N**2)


                """
                NOTE:
                Gershgorin circle theorem is usually more restrictive than the general energy (trace) rule,
                especially in the case of higly diagonally dominant matrices and matrices, where the entries
                are sort of "uniformly distributed" across rows/columns (there is no row/column, that would
                stand out with respect to its sum). However, it is also a lot more computationally expensive
                at O(N**2), and cannot be computed without prior computation of K_Fsc, hence it is used only
                when the energy content of the scale is reasonably low, i.e., the chance of skipping the scale
                or lowernig the number of modes in it based on this theorem is high.
                """

                if E[k] <= GThresh*lamMin:
                    print(" -> Low energy band, estimating no of modes from Gershgorin theorem")
                    # Compute L1 norms of rows, i.e., upper bounds on eivals from Gershgorin theorem
                    potLam = np.sum(np.abs(K_Fsc), axis=1)  # This is O(N**2)
                    maxNoOfLam = np.count_nonzero(potLam > lamMin)
                    if maxNoOfLam == 0:
                        print(" -> Skippimg band %d based on Gershgorin theorem \n"%(k+1))
                        continue
                    else:
                        print(" -> Estimate: Up to %d signifficant modes in the band"%maxNoOfLam)
                else:
                    maxNoOfLam = nModes

                noOfLams = np.min([nModes,N,maxNoOfLam])

                """
                NOTE:
                Whole K_F as computed from either K or D_hat should be hermitian symmetric by definition,
                but we dont give a duck about the whole thing, we are only interested in the part of it
                that is nonzero in current scale, so lets assure only the hermitian symmetry of K_Fsc for
                computational efficiency reasons. This symmetrization should filter out some numerical jitter.
                """

                # Enforcing hermitian symmetry of K_Fsc
                K_Fsc = (K_Fsc + K_Fsc.conj().T)/2

                # Compute noOfLams largest eigenvalues
                print(" -> Computing eigenvalues")
                lam, eigv = eigh(K_Fsc,subset_by_index=[N-noOfLams,N-1],check_finite=False)     # This is O(N**3)



            elif mode in ["fullSVD", "randSVD"]:    # SVD-based approaches
                noOfLams = np.min([nModes,N])

                if mode == "fullSVD":
                    if taperWid[scalesToKeep[k]] > 0:
                        print(" -> Computing eigenvectors of K_Fsc using full SVD with smoothing")
                        _, lam, eigv = svd(D_hat[:,indices]*mask1D, full_matrices=False, check_finite=False) # O(Ns*N**2)
                    else:
                        print(" -> Computing eigenvectors of K_Fsc using full SVD")
                        _, lam, eigv = svd(D_hat[:,indices], full_matrices=False, check_finite=False) # O(Ns*N**2)

                    # Take only the noOfLams largest singular values and corresponding right singular vectors
                    lam = lam[:noOfLams]
                    eigv = eigv[:noOfLams,:]

                else: # mode == "randSVD"
                    if n_iter >= 0:
                        powIter = n_iter
                    elif n_iter == -1:
                        powIter = 7 if noOfLams < 0.1*N else 4
                    else:
                        raise ValueError("Invalid n_iter value, must be >=0 for manual setting or -1 for auto-setting.")

                    if taperWid[scalesToKeep[k]] > 0:
                        print(" -> Computing eigenvectors of K_Fsc using randomized SVD with smoothing")
                        _, lam, eigv = rsvd(D_hat[:,indices]*mask1D, t=int(noOfLams), p=powIter, oversampling=oversampling) # O(Ns*N*noOfLams)
                    else:
                        print(" -> Computing eigenvectors of K_Fsc using randomized SVD")
                        _, lam, eigv = rsvd(D_hat[:,indices], t=int(noOfLams), p=powIter, oversampling=oversampling) # O(Ns*N*noOfLams)
                    lam = np.diag(lam)

                eigv = eigv.conj().T  # To be consistent with eigendecomposition output, where eigenvectors are column-wise, in SVD they are row-wise

            else:
                raise ValueError("Invalid computation mode specified.")

            """
            NOTE:
            Sorting of eigenvalues/eigenvectors per scale and keeping only 2*nModes eigvals and eigvecs globally
            in each iteration is computationally comparable to sorting all scales at the end and keeping only
            nModes globally (noOfScales*O(2*nModes log 2*nModes) vs O(noOfScales*nModes log noOfScales*nModes)).
            This approach was made even cheaper by only sorting the nonzero entries in each scale, not the full 2*nModes.
            It also saves memory, and allows the energy check per scale described above to operate much more efficiently.
            """

            print(" -> Selecting significant modes")
            if not lamTot is None:
                ind2 = np.nonzero(lam > lamMin)[0]
                if ind2.size == 0:
                    print(" -> No significant modes found \n")
                    continue

                maxInd = nModes+ind2.size
                lamTot[nModes:maxInd] = lam[ind2]
                # Expand eigenvectors to full frequency space (decompress them by 0 padding)
                eigvTot[indices,nModes:maxInd] = eigv[:,ind2]

                # Sort all modes by descending eigenvalue magnitude
                idx = np.flip(np.argsort(lamTot[:maxInd]))
                lamTot[:maxInd] = lamTot[idx]
                eigvTot[:,:maxInd] = eigvTot[:,idx]
                # Zero-out nonsignifficant modes
                lamTot[nModes:] = 0
                eigvTot[:,nModes:] = 0
                print(" -> Kept %d modes \n"%ind2.size)

            else:
                # lamTot is None
                print(" -> Selection unapplicable, taking %d modes \n"%noOfLams)
                # Allocate space for eigenvalues and eigenvectors for faster operation
                lamTot = np.zeros((2*nModes))
                eigvTot = np.zeros((Nt,2*nModes), dtype = 'complex')

                # Sorting to have descending eigenvalues
                idx = np.flip(np.argsort(lam))
                lamTot[:lam.size] = lam[idx]
                # Expand eigenvectors to full frequency space (decompress them by 0 padding)
                eigvTot[indices,:lam.size] = eigv[:,idx]

            lamMin = lamTot[nModes-1]   # Update the smallest significant eigenvalue for checks in the next scale

        # Take only the desired amount of modes
        eigvTot = eigvTot[:,:nModes]

        # Only the final modes are transformed back to time domain, saving time (least possible ffts)
        psi = fft(eigvTot,Nt,axis=0,norm='ortho',workers=np.min([ncpus,nModes])).real
        #print("Done in %.3f s"%(time()-startLoop))

        #startNorm = time()
        print("Normalizing temporal modes ...")
        psi = psi/np.linalg.norm(psi,axis=0)
        #print("Done in %.3f s"%(time()-startNorm))

        # Reconstruct spatial mPOD modes
        #startSpat = time()
        print("Reconstructing spatial modes ...")
        phi = np.linalg.matmul(self.D, psi)  # Get spatial mode shapes
        # Get singular values as norms of spatial modes (more accurate than computation from K_Fsc, should fit the data better by enforcing ||phi_i||=1)
        sigTot = np.linalg.norm(phi,axis=0)
        phi = phi/sigTot    # Normalize spatial modes
        # Make sure the singular values and modes are correctly sorted and that the correction of singvals did not mess with the ordering
        idx = np.flip(np.argsort(sigTot))
        sigTot = sigTot[idx]
        phi = phi[:,idx]
        psi = psi[:,idx]
        #print("Done in %.3f s"%(time()-startSpat))

        return phi, psi, sigTot
    

    def DMD(self, SAVE_T_DMD: bool = True, F_S: float = 1.0, verbose:  bool = True):
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
                                              D[:, 1:self.N_T], self.n_Modes, F_S, svd_solver=self.svd_solver,verbose=verbose)

        else:
            Phi_D, Lambda, freqs, a0s = dmd_s(self.D[:, 0:self.N_T - 1],
                                              self.D[:, 1:self.N_T], self.n_Modes, F_S, SAVE_T_DMD=SAVE_T_DMD,
                                              svd_solver=self.svd_solver, FOLDER_OUT=self.FOLDER_OUT,verbose=verbose)

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

            # Segment and FFT - fallback n_processes in case of misassignment
            D_hat, freqs_pos, n_processes = segment_and_fft(
                D=D,
                F_S=F_S,
                L_B=L_B,
                O_B=O_B,
                n_processes=n_processes
                )

            return self.compute_SPOD_t(D_hat=D_hat,
                                       freq_pos=freqs_pos,
                                        n_Modes=n_Modes,
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
                folder_out = self.FOLDER_OUT + "MODULO_tmp/"
                os.makedirs(folder_out, exist_ok=True)
                np.savez(
                folder_out + "spod_towne.npz",
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
        else:
                D = self.D

        self.K = CorrelationMatrix(self.N_T, self.N_PARTITIONS, self.MEMORY_SAVING,
                                   self.FOLDER_OUT, self.SAVE_K, D=D)

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
        r"""
        Perform kernel PCA (kPOD) for snapshot data, following the VKI Machine Learning for Fluid Dynamics course.

        Parameters
        ----------
        M_DIST : array-like of shape (2,), optional
            Indices of two snapshots used to estimate the minimal kernel value.
            These should correspond to the most distant snapshots in the dataset.
            Default is [1, 10].

        k_m : float, optional
            Minimum value for the kernelized correlation. Default is 0.1.

        cent : bool, optional
            If True, center the kernel matrix before decomposition.
            Default is True.

        n_Modes : int, optional
            Number of principal modes to compute. Default is 10.

        alpha : float, optional
            Regularization parameter for the modified kernel matrix \(K_{\zeta}\).
            Default is 1e-6.

        metric : str, optional
            Kernel function identifier passed to
            `sklearn.metrics.pairwise.pairwise_kernels`.
            Only 'rbf' has been tested; other metrics may require additional
            parameters. Default is 'rbf'.

        K_out : bool, optional
            If True, also return the full kernel matrix \(K\).
            Default is False.

        SAVE_KPOD : bool, optional
            If True, save the computed kPOD results to disk.
            Default is False.

        Returns
        -------
        Psi_xi : ndarray of shape (n_samples, n_Modes)
            The kPOD principal component coefficients.

        Sigma_xi : ndarray of shape (n_Modes,)
            The kPOD singular values, obtained from the eigendecomposition of the
            centered kernel matrix.

        Phi_xi : ndarray of shape (n_samples, n_Modes)
            The mapped eigenvectors, that is, the principal modes in feature space.

        K_zeta : ndarray of shape (n_samples, n_samples)
            The regularized and centered kernel matrix used for the decomposition.
            Returned only if `K_out` is True.

        Notes
        -----
        - This implementation follows the hands-on tutorial from the VKI Machine
        Learning for Fluid Dynamics course.
        - The kernel is computed as described in Horenko et al.,
        *Machine learning for dynamics and model reduction*, arXiv:2208.07746.
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
            return Phi_xi, Psi_xi, Sigma_xi, None


#     def kDMD(self,
#              F_S=1.0,
#              M_DIST=[1, 10],
#              k_m=0.1, cent=True,
#              n_Modes=10,
#              n_modes_latent=None,
#              alpha=1e-6,
#              metric='rbf', K_out=False):
#         """
#         Perform kernel DMD (kDMD) for snapshot data as in VKI’s ML for Fluid Dynamics course.

#         Parameters
#         ----------
#         M_DIST : array-like of shape (2,), optional
#                 Indices of two snapshots used to estimate the minimal kernel value.
#                 These should be the most “distant” snapshots in your dataset. Default is [1, 10].
#         F_S: float, sampling frequency.
#         k_m : float, optional
#                 Minimum value for the kernelized correlation. Default is 0.1.
#         cent : bool, optional
#                 If True, center the kernel matrix before decomposition. Default is True.
#         n_Modes : int, optional
#                 Number of principal modes to compute. Default is 10.
#         alpha : float, optional
#                 Regularization parameter for the modified kernel matrix \(K_{\zeta}\). Default is 1e-6.
#         metric : str, optional
#                 Kernel function identifier (passed to `sklearn.metrics.pairwise.pairwise_kernels`).
#                 Only 'rbf' has been tested; other metrics may require different parameters. Default is 'rbf'.
#         K_out : bool, optional
#                 If True, also return the full kernel matrix \(K\). Default is False.
#         SAVE_KPOD : bool, optional
#                 If True, save the computed kPOD results to disk. Default is False.

#         Returns
#         -------
#         Psi_xi : ndarray of shape (n_samples, n_Modes)
#                 The kPOD principal component time coefficients.
#         Sigma_xi : ndarray of shape (n_Modes,)
#                 The kPOD singular values (eigenvalues of the centered kernel).
#         Phi_xi : ndarray of shape (n_samples, n_Modes)
#                 The mapped eigenvectors (principal modes) in feature space.
#         K_zeta : ndarray of shape (n_samples, n_samples)
#                 The (regularized and centered) kernel matrix used for decomposition.
#                 Only returned if `K_out` is True.

#         Notes
#         -----
#         - Follows the hands-on ML for Fluid Dynamics tutorial by VKI
#         (https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552).
#         - Kernel computed as described in
#         Horenko et al., *Machine learning for dynamics and model reduction*, arXiv:2208.07746.
#         """
#         # we need the snapshot matrix in memory for this decomposition

#         if self.MEMORY_SAVING:
#                 if self.N_T % self.N_PARTITIONS != 0:
#                         tot_blocks_col = self.N_PARTITIONS + 1
#                 else:
#                         tot_blocks_col = self.N_PARTITIONS

#                 # Prepare the D matrix again
#                 D = np.zeros((self.N_S, self.N_T))
#                 R1 = 0

#                 # print(' \n Reloading D from tmp...')
#                 for k in tqdm(range(tot_blocks_col)):
#                         di = np.load(self.FOLDER_OUT + f"/data_partitions/di_{k + 1}.npz")['di']
#                         R2 = R1 + np.shape(di)[1]
#                         D[:, R1:R2] = di
#                         R1 = R2
#         else:
#                 D = self.D

#         n_s, n_t = D.shape
#         # as done with the classic dmd, we assume X = D_1 = D(0:n_t - 1) and
#         # Y = D_2 = D(1:n_t)

#         X = D[:, :-1]
#         Y = D[:, 1:]

#         # we seek A = argmin_A ||Y - AX|| = YX^+ = Y(Psi_r Sigma_r^+ Phi^*)
#         n_modes_latent = n_Modes if n_modes_latent is None else n_modes_latent

#         # leverage MODULO kPOD routine to compress the system instead of standard POD
#         # we are now in the kernel (feature) space, thus:
#         i, j = M_DIST

#         # gamma needs to be the same for the feature spaces otherwise
#         # leads to inconsistent galerkin proj.!

#         M_ij = np.linalg.norm(X[:, i] - X[:, j]) ** 2

#         gamma = - np.log(k_m) / M_ij

#         K_XX = pairwise_kernels(X.T, X.T, metric=metric, gamma=gamma)
#         K_YX = pairwise_kernels(Y.T, X.T, metric=metric, gamma=gamma)

#         # (optional) center feature‐space mean by centering K_XX only
#         if cent:
#                 n = K_XX.shape[0]
#                 H = np.eye(n) - np.ones((n, n)) / n
#                 K_XX = H @ K_XX @ H

#         # add ridge to K_XX
#         K_XX += alpha * np.eye(K_XX.shape[0])

#         # kernel‐POD on the regularized, centered K_XX
#         Psi_xi, sigma_xi = Temporal_basis_POD(K=K_XX, n_Modes=n_modes_latent, eig_solver='eigh')
#         Sigma_inv = np.diag(1.0 / sigma_xi)

#         # Galerkin projection using the **unmodified** K_YX
#         A_r = Sigma_inv @ Psi_xi.T @ K_YX @ Psi_xi @ Sigma_inv

#         # eigendecomposition of A gives DMD modes
#         dt = 1/F_S
#         Lambda, Phi_Ar = LA.eig(A_r)
#         freqs = np.imag(np.log(Lambda)) / (2 * np.pi * dt)

#         # we can trace back the eigenvalues of the not-truncated A (Tu et al.)
#         Phi_D = Y @ Psi_xi @ Sigma_inv @ Phi_Ar
#         a0s = LA.pinv(Phi_D).dot(X[:, 0])

#         return Phi_D, Lambda, freqs, a0s, None







