======================================
Computing decompositions
======================================

``MODULO`` allows for computing POD, SPODs, DFT, DMD and mPOD. 

The modulo object is initialized as follows:

.. autofunction:: modulo.MODULO.__init__

Then, the key functions for the available decompositions are: 

+++++++++++++++++++++++++++++++++++++
1. POD via SVD 
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_POD_svd


+++++++++++++++++++++++++++++++++++++
2. POD via matrix K 
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_POD_K

+++++++++++++++++++++++++++++++++++++
3. DFT 
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_DFT

+++++++++++++++++++++++++++++++++++++
4. SPOD_s  
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_SPOD_s

+++++++++++++++++++++++++++++++++++++
5. SPOD_t 
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_SPOD_t

+++++++++++++++++++++++++++++++++++++
6. DMD ( or PIP)  
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_DMD_PIP

+++++++++++++++++++++++++++++++++++++
7. mPOD
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_mPOD

+++++++++++++++++++++++++++++++++++++
8. kPOD
+++++++++++++++++++++++++++++++++++++

.. autofunction:: modulo.MODULO.compute_kPOD




