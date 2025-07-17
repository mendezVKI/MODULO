import os
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm

from modulo_vki.core._k_matrix import CorrelationMatrix, spectral_filter, kernelized_K
from modulo_vki.core.temporal_structures import dft, temporal_basis_mPOD, Temporal_basis_POD
from modulo_vki.core.spatial_structures import Spatial_basis_POD, spatial_basis_mPOD
from modulo_vki.core._dmd_s import dmd_s

from modulo_vki.core.utils import segment_and_fft, pod_from_dhat, apply_weights, switch_svds
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
        """
        Perform kernel PCA (kPOD) for snapshot data as in VKI Machine Learning for Fluid Dynamics course.

        Parameters
        ----------
        M_DIST : array-like of shape (2,), optional
                Indices of two snapshots used to estimate the minimal kernel value.
                These should be the most “distant” snapshots in your dataset. Default is [1, 10].
        k_m : float, optional
                Minimum value for the kernelized correlation. Default is 0.1.
        cent : bool, optional
                If True, center the kernel matrix before decomposition. Default is True.
        n_Modes : int, optional
                Number of principal modes to compute. Default is 10.
        alpha : float, optional
                Regularization parameter for the modified kernel matrix \(K_{\zeta}\). Default is 1e-6.
        metric : str, optional
                Kernel function identifier (passed to `sklearn.metrics.pairwise.pairwise_kernels`).
                Only 'rbf' has been tested; other metrics may require different parameters. Default is 'rbf'.
        K_out : bool, optional
                If True, also return the full kernel matrix \(K\). Default is False.
        SAVE_KPOD : bool, optional
                If True, save the computed kPOD results to disk. Default is False.

        Returns
        -------
        Psi_xi : ndarray of shape (n_samples, n_Modes)
                The kPOD principal component time coefficients.
        Sigma_xi : ndarray of shape (n_Modes,)
                The kPOD singular values (eigenvalues of the centered kernel).
        Phi_xi : ndarray of shape (n_samples, n_Modes)
                The mapped eigenvectors (principal modes) in feature space.
        K_zeta : ndarray of shape (n_samples, n_samples)
                The (regularized and centered) kernel matrix used for decomposition.
                Only returned if `K_out` is True.

        Notes
        -----
        - Follows the hands-on ML for Fluid Dynamics tutorial by VKI  
        (https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552).  
        - Kernel computed as described in  
        Horenko et al., *Machine learning for dynamics and model reduction*, arXiv:2208.07746.

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
        
                

                
    


        