=========================================================
Towards customization: accessing MODULO internal modules
=========================================================

The methods for each decomposition (POD, DMD, mPOD, etc) are made of different sub-methods. 
It is possible to call each of these separately, hence allowing the user to customize the various steps if needed.

+++++++++++++++++++++
Functions for the POD
+++++++++++++++++++++

For example, the method compute_POD_K acts in three steps: 1. compute K, 2. diagonalize it to compute the Psi's and 3 project to compute the Phi's.

These are handled by three methods, with use the functions in modulo.core: 

.. autofunction:: modulo.core._k_matrix.CorrelationMatrix  

.. autofunction:: modulo.core._pod_time.Temporal_basis_POD

.. autofunction:: modulo.core._pod_space.Spatial_basis_POD


++++++++++++++++++++++
Functions for the mPOD
++++++++++++++++++++++

 Similarly, the equivalent version of these functions for the mPOD are:

.. autofunction:: modulo.core._mpod_time.temporal_basis_mPOD

.. autofunction:: modulo.core._mpod_space.spatial_basis_mPOD

++++++++++++++++++++++
Functions for the DFT
++++++++++++++++++++++

.. autofunction:: modulo.core._dft.dft_fit

+++++++++++++++++++++++++++
Functions for the DMD (PIP)
+++++++++++++++++++++++++++

.. autofunction:: modulo.core._dmd_s.dmd_s





