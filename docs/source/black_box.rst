======================================
Computing decompositions
======================================

``MODULO`` allows for computing POD, SPODs, DFT, DMD and mPOD. 


MODULO Initialization
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.__init__

Then, the key functions for the available decompositions are: 

POD via SVD
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_POD_svd


POD via matrix K
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_POD_K

DFT
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_DFT


SPOD_s
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_SPOD_s


SPOD_t
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_SPOD_t


DMD ( or PIP)
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_DMD_PIP


mPOD
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_mPOD


kPOD
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_kPOD




