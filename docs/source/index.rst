MODULO (MODal mULtiscale pOd): Data-Driven Modal Analysis
==========================================================

MODULO (MODal mULtiscale pOd) is a software developed at the von Karman Institute for Fluid Dynamics to perform data-driven modal decompositions.
Initially focused on the Multiscale Proper Orthogonal Decomposition (mPOD), it has recently been extended to perform also other decompositions that include POD, SPODs, DFT, DMD, mPOD. 


.. toctree::
   :maxdepth: 2
   :caption: Getting Started  

   intro
   installation
   importing_data


Data-Driven Modal Decompositions
---------------------------------

All modal decompositions are implemented in MODULO as matrix factorization of the snapshot matrix :math:`D(\mathbf{x}, t) \in \mathbb{R}^{n_S \times n_t}`.
Each entry of the snapshot matrix is a flattened realization of the data, regardless of the data dimensionality (2D, 3D). Thus, this decomposition
reads 

.. math:: 

   D(\mathbf{x}, t) = \mathbf{\Phi}\mathbf{\Sigma}\mathbf{\Psi}^T

where :math:`\mathbf{\Phi} \in \mathbb{R}^{n_S \times r}` and :math:`\mathbf{\Psi} \in \mathbb{R}^{n_t \times r}` are the spatial and temporal modes, 
respectively, and :math:`\mathbf{\Sigma} \in \mathbb{R}^{r \times r}` is a diagonal matrix containing the modal energy associated with each mode.

All decompositions available in MODULO have an orthonormal temporal basis (:math:`\Psi^{-1}=\Psi^\dagger`,  where :math:`\dagger` denotes Hermitian transpose).
Then, the spatial structures are readily computed from:

.. math::
   \mathbf{\Phi} = \mathbf{D} \bar{\mathbf{\Psi}} \mathbf{\Sigma}^{-1} \;,

where the modal amplitude matrix :math:`\mathbf{\Sigma}` is retreived by normalization of the spatial structures, that is:

.. math::
   \sigma_r = \| \mathbf{D} \Psi_r \| \;.

The reminder of this section overviews the salient theoretical aspects of the implement modal decompositions, and the corresponding 
implementation in MODULO. The following decompositions are available:

.. toctree::
   :maxdepth: 1
   :glob:
   
   decompositions/dft/index
   decompositions/pod/index
   decompositions/dmd/index
   decompositions/mpod/index
   decompositions/kpod/index
   decompositions/spod/index

   .. decompositions/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/dft 
   api/pod 
   api/mpod


.. toctree::
   :maxdepth: 1
   :caption: Changelog

   change_log 
