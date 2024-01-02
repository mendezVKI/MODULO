

MODULO - latest update 2.0
===================

This repository contains the version 2.0 of MODULO (MODal mULtiscale pOd), a software developed at the von Karman Institute to perform Multiscale Modal Analysis of numerical and experimental data using the Multiscale Proper Orthogonal Decomposition (mPOD).


The old version based on MATLAB implementation and related GUI is no longer mantained but will remain available at on the branch "Old_Matlab_Implementation". We keep the first Python implementation in the branch "Old_Python_Implementation". 

On the other hand, we do not keep track of the version 1.1 since this is fully replaced by the current version in this repository.


### What is MODULO ?





### What is new in this V 2.0


The new version of MODULO has the following major updates

1. Implementation of faster POD algorithms, using powerful randomized svd solvers from scikit_learn.
2. Implementation of Dynamic Mode Decomposition (DMD)
3. Implementation of the two Spectral POD formulations, namely the one from [Sieber et al 2016](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition/DCD8A6EDEFD56F5A9715DBAD38BD461A), and the one from [Towne et al 2018](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition-and-its-relationship-to-dynamic-mode-decomposition-and-resolvent-analysis/EC2A6DF76490A0B9EB208CC2CA037717)
4. Implementation of a kernel version of the POD, in which the correlation matrix is replaced by a kernel matrix.
5. Implementation of a formulation for non-uniform meshes, using a weighted matrix for the snapshot matrix.  
6. Implementation of a different solvers for diagonalizing K (particularly relevant for the mPOD).


While most of the aforementioned decompositions are implemented in existing open-source packages, MODULO is currently the only package implementing the mPOD.


The theoretical foundation of the decomposition is described in
- M.A. Mendez, M. Balabane, J.-M. Buchlin, Multiscale Proper Orthogonal Decomposition of Complex Fluid Flows, Journal of Fluid Mechanics, Vol 870, July 2019, pp. 988-1036. The pre-print is available at https://arxiv.org/abs/1804.09646


To familiarize with the usage of MODULO, this repository contains four tutorials. Namely:


1. DMD and POD analysis of a 2D flow past a cylinder, simulated using 2D LES in openFOAM.
2. POD, SPODs and mPOD analysis of an impinging jet flow, obtained from a TR-PIV campaign at the von Karman Institute. This is one of test cases in [Mendez et al 2018](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/multiscale-proper-orthogonal-decomposition-of-complex-fluid-flows/D078BD2873B1C30B6DD9016E30B62DA8), available on arxiv at https://arxiv.org/abs/1804.09646. 
3. POD, mPOD and kPOD of the flow past a cylinder in transient conditions. This is the dataset analyzed in [Mendez et al 2020](https://iopscience.iop.org/article/10.1088/1361-6501/ab82be/meta)




