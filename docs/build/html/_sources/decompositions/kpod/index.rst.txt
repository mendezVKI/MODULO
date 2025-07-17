Kernel Proper Orthogonal Decomposition (KPOD)
=================================================

This document offers a brief reference guide to the use of the Kernel Proper Orthogonal Decomposition (KPOD).

Theory 
----------
The Kernel POD is a nonlinear variant of the POD. It seeks to capture nonlinear correlations between snapshots 
via their embedding into a higher dimensional feature space (that we shall denote as :math:`xi`), defined by a 
kernel function, and performs the POD there :cite:`mendez_2023`.

Thus, the main step is the computation of this kernel function. To this end, we compute the euclidean distance 
between snapshots :math:`\mathbf{d}_i \in \mathbb{R}^{n_S}` as 

.. math::
    M_{ij} = \| \mathbf{d}_i - \mathbf{d}_j \|^2 \;.

In the following, we use Radial Basis Function (RBFs) as kernel function. We compute the inverse length-scale
of the RBF as: 

.. math::
    \gamma = - \frac{\log{k_m}}{M_{ij}}

where :math:`k_m` is a minimum value for the kernelized correlation. This leads to the kernelized expression of
the correlation matrix as:

.. math::
    \mathbf{K}_\xi = \exp{(-\gamma \| \mathbf{d}_i - \mathbf{d}_j \|^2)} \;.

From this point onward, the procedure is the one of a standard POD: the temporal modes :math:`\psi^\xi_r` are 
eigenvectors of :math:`\mathbf{K}^\xi` (that now is a nonlinear manifold), the amplitudes are 
:math:`\sigma_r^\xi = \sqrt{\lambda^xi}`, and the spatial modes :math:`\phi^\xi_r` are obtained via projection.

Example with MODULO 
--------------------
The ``kPOD`` is called in MODULO as:

.. code-block::

    from modulo_vki import ModuloVKI

    # --- Initialize MODULO object
    m = ModuloVKI(data=D,svd_solver='svd_scipy_sparse')
    M_DIST=[1,19]
    Phi_kPOD, Psi_kPOD, Sigma_kPOD,K_zeta = m.kPOD(M_DIST=M_DIST,k_m=k_m,
    cent=True, K_out=True)

where ``M_DIST`` defines the snapshot indexes of which to compute the distance, ``k_m`` is the minimum kernelized
correlation threshold described above, ``cent=True`` performs centering of :math:`\mathbf{K}^\xi` and ``K_out=True``
returns the kernelized correlation matrix. 

.. note::
    The hyper parameter :math:`k_m` heavily influences the outcomes of the kPOD. For large :math:`k_m`
    then :math:`\mathbf{K}^\xi \rightarrow \mathbf{K}`, whilst smaller :math:`k_m` will identify 
    correlation happening at a specific frequency. The matrix is Toeplitz, and its eigenvectors
    tends naturally towards the Fourier basis. Outcome of this step would remind the filtering as 
    in the diagonally filtered SPOD (Sieber et al.), but with more pronounced focus on a specific frequency range.








