======================================
Computing decompositions
======================================

``MODULO`` allows for computing POD, SPODs, DFT, DMD and mPOD. 


MODULO Initialization
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo_vki.__init__

Then, the key functions for the available decompositions are: 

POD via SVD
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo_vki.modulo.ModuloVKI.compute_POD_svd


POD via matrix K
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo_vki.modulo.ModuloVKI.compute_POD_K

DFT
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo_vki.modulo.ModuloVKI.compute_DFT


SPOD_s
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo_vki.modulo.ModuloVKI.compute_SPOD_s


SPOD_t
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo_vki.modulo.ModuloVKI.compute_SPOD_t


DMD (or PIP)
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo_vki.modulo.ModuloVKI.compute_DMD_PIP


mPOD
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo_vki.modulo.ModuloVKI.compute_mPOD


kPOD
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo_vki.modulo.ModuloVKI.compute_kPOD




