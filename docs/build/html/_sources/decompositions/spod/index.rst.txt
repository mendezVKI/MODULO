Spectral Proper Orthogonal Decomposition (SPOD)
=================================================

This document offers a brief reference guide to the use of the Spectral Proper Orthogonal Decomposition (SPOD).

Theory 
----------
The SPOD aims at identifying the coherent modes of a stochastic process at specific frequencies. In the literature, 
there are two spectral paths to achieve this result: (1) low-pass filtering the covariance matrix :math:`\mathbf{K}`
along the diagonals (:cite:`sieber_paschereit_oberleithner_2016`); or (2) by solving an Fredholm eigenvalue problem for 
the cross-spectral density matrix at each frequency bin (:cite:`Towne_2018`).

Both options are available in MODULO, and are briefly reviewed below. 

Filtered-Covariance SPOD (Sieber et al.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The key idea of this approach is to enforce a diagonal similarity of the correlation matrix to get a smoother dynamics. 
To this end, the (temporal) correlation matrix :math:`\mathbf{K}` is low-pass filtered along the diagonals. This reads:

.. math::
    
    K_{i, j}^f = \sum_{k=-{N_f}}^{N_f} g_k \; K_{i+k, j+k} 

where :math:`g_k` is a symmetric FIR filter with coefficients of length :math:`2N_f + 1`. Once the filtered correlation
matrix :math:`\mathbf{K}^f` is botained, the procedure of the SPOD is the same as for the classical POD.

.. note::

    Note that by default the spatial modes obtained with this procedure :math:`\mathbf{\Phi}(x)` are no longer orthogonal. 
    MODULO rescales them ensuring orthonormality using the standard snapshot approach, via the argument ``rescale=True``
    in the ``Spatial_basis_POD`` call.


Cross-Spectral Density SPOD (Towne et al.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This formulation operates directly in the frequency domain. Thus, to find modes that depends on both space and time, 
one needs to define the appropriate inner product. Following the original derivation, this reads:

.. math::
    \langle \mathbf{u}, \mathbf{v} \rangle_{x, t} = \int_{-\infty}^\infty \int_\Omega \mathbf{v}^*(\mathbf{x}, t) \mathbf{u}(\mathbf{x}, t) \; d\mathbf{x}dt.

For each frequency :math:`f`, we compute the cross‐spectral density by sliding a DFT window over the lags:

.. math::
      S_{i,j}(f)= \sum_{m=-M}^{M}
        K_{\,i+m,\;j+m}\,\exp\bigl(-\,\mathrm{i}2\pi f\,m\Delta t\bigr)\quad\Delta t = 1/F_S.

The deterministic functions :math:`\mathbf{\phi}(\mathbf{x},t)` are then obtained by solving, at each frequency :math:`f`, 
the Hermitian Fredholm eigenproblem:

.. math::
   \sum_{j=1}^{n_t}
     S_{i,j}(f)\,\psi_r(t_j;f)
   = \lambda_r(f)\,\psi_r(t_i;f),
   \quad
   \sum_{i=1}^{n_t}\psi_r^*(t_i;f)\psi_s(t_i;f)=\delta_{rs},

where :math:`\psi_r(t_j, f)` are the temporal SPOD modes and :math:`\lambda_r(f)` the corresponding spectral energies.  
The spatial SPOD modes :math:`\phi_r(\mathbf{x};f)` follow from the snapshot projection of the data onto :math:`\psi_r`, 
and the full spatio‐temporal structures become

.. math::
   \mathbf{\Phi}_r(\mathbf{x},t;f)
   = \phi_r(\mathbf{x};f)\;\exp\!\bigl(\mathrm{i}2\pi f\,t\bigr).

Example in MODULO 
-------------------

The function ``SPOD`` of MODULO wraps both approaches, yet the arguments required are different.

.. code-block:: python 

    # --- Initialize MODULO object
    m = ModuloVKI(data=np.nan_to_num(D))
    # Compute the SPOD of Sieber et al.
    Phi_S, Psi_S, Sigma_S = m.SPOD(mode='sieber',
                                    N_O=100,
                                    f_c=0.01,
                                    n_Modes=25,
                                    SAVE_SPOD=True)
    # Compute the SPOD of Towne et al.
    Phi_ST, Sigma_ST, Freqs_PosT = m.SPOD(mode='towne',
                                            F_S=2000, # sampling frequency
                                            L_B=200, # Length of the chunks for time average
                                            O_B=150, # Overlap between chunks
                                            n_Modes=3) # number of modes PER FREQUENCY

.. note::
    `MODULO v. 2.1.0` includes some parallelization capabilities for the CSD SPOD. These take effect in two 
    steps: (1) when segmenting and computing the FFT on each block of :math:`\hat{D}(f)`; and (2) when 
    computing the POD `n_modes` for each frequency. This enables the user for a direct control over the threads
    leveraged in the computation, that otherwise are left up to the numpy vectorization routines.
    
