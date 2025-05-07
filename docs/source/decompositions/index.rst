Data-Driven Modal Decompositions
=================================================

All modal decompositions are implemented in MODULO as matrix factorization of the snapshot matrix :math:`D(\mathbf{x}, t) \in \mathbb{R}^{n_S \times n_t}`.
Each entry of the snapshot matrix is a flattened realization of the data, regardless of the data dimensionality (2D, 3D). Thus, this decomposition
reads 

.. math:: 

    D(\mathbf{x}, t) = \mathbf{\Phi}\mathbf{\Sigma}\mathbf{\Psi}^T

where :math:`\mathbf{\Phi} \in \mathbb{R}^{n_S \times r}` and :math:`\mathbf{\Psi} \in \mathbb{R}^{n_t \times r}` are the spatial and temporal modes, 
respectively, and :math:`\mathbf{\Sigma} \in \mathbb{R}^{r \times r}` is a diagonal matrix containing the modal energy associated with each mode. 

The reminder of this section overviews the salient theoretical aspects of the implement modal decompositions, and the corresponding 
implementation in MODULO. The following decompositions are available:

.. toctree::
   :maxdepth: 2
   :glob:
   
   dft/*
   pod/*
   dmd/*
   mpod/*
   kpod/*
   spod/*