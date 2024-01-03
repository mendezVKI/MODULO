

MODULO - latest update 2.0
===================

This repository contains the version 2.0 of MODULO (MODal mULtiscale pOd), a software developed at the von Karman Institute to perform data driven modal decompositions and in particular the Multiscale Proper Orthogonal Decomposition (mPOD).

The old version based on MATLAB implementation and related GUI is no longer mantained but will remain available at on the branch "Old_Matlab_Implementation". We also keep the first Python implementation in the branch "Old_Python_Implementation". See the Readme file in these branches for more information.

### What is MODULO and what are data-driven decompositions ?

MODULO allows to perform data-driven decompositions of experimental and numerical data.
To have a concise overview of the context, we refer to: 

- Ninni, D., Mendez, M. A. (2020), "MODULO: A Software for Multiscale Proper Orthogonal Decomposition of data", Software X, Vol 12, 100622, https://doi.org/10.1016/j.softx.2020.100622.

This article also presents the first version of MODULO (available in the OLD_Matlab_Implementation branch) and its GUI developed by D. Ninni. 
For a more comprehensive overview of data driven decompositions, we refer to the chapter:

- Mendez, M. A. (2023) "Generalized and Multiscale Modal Analysis". In : Mendez M.A., Ianiro, A., Noack, B.R., Brunton, S. L. (Eds), "Data-Driven Fluid Mechanics: Combining First Principles and Machine Learning". Cambridge University Press, 2023:153-181. https://doi.org/10.1017/9781108896214.013. The pre-print is available at https://arxiv.org/abs/2208.12630. 


### What is new in this V 2.0 ? 

This version expands considerably the version v1 in "Old_Python_Implementation", for which a first tutorial was provided by L. Schena in https://www.youtube.com/watch?v=y2uSvdxAwHk. 
The major updates are the following :

1. Faster EIG/SVD algorithms, using powerful randomized svd solvers from scikit_learn (see [this](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) and [this](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html) ). It is now possible to select various options as "eig_solver" and "svd_solver", offering different trade offs in terms of accuracy vs computational time.

2. In addition to the traditional POD computation using the K matrix (Sirovinch's method), it is now possible to compute the POD directly via SVD using any of the four "svd_solver" option.
This is generally faster, but generally requires more memory.

3. Faster subscale estimators for the mPOD: the previous version used the rank of the correlation matrix in each scale to define the number of modes to be computed in each portion of the splitting vector before assemblying the full basis. This is computationally very demanding. This estimation has been replaced by a frequency-based threshold (i.e based on the frequency bins within each portion), since one can show that the frequency-based estimator is always more "conservative" than the rank based estimator.

4. Major improvement on the memory saving option: the previous version of modulo always required in input the matrix D. Then, if the memory saving option was active, the matrix was partitioned and stored locally to minimize the RAM usage (see [this tutorial by D. Ninni](https://www.youtube.com/watch?v=LclxO1WTuao))


4. Implementation of Dynamic Mode Decomposition (DMD) from [Schmid, P.J 2010](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4)
4. Implementation of the two Spectral POD formulations, namely the one from [Sieber et al 2016](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition/DCD8A6EDEFD56F5A9715DBAD38BD461A), and the one from [Towne et al 2018](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition-and-its-relationship-to-dynamic-mode-decomposition-and-resolvent-analysis/EC2A6DF76490A0B9EB208CC2CA037717)
5. Implementation of a kernel version of the POD, in which the correlation matrix is replaced by a kernel matrix. This is described in Lecture 15 of the course [Hands on Machine Learning for Fluid dynamics 2023](https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552/-/online-on-site-hands-on-machine-learning-for-fluid-dynamics-2023)
6. Implementation of a formulation for non-uniform meshes, using a weighted matrix for the snapshot matrix.  





While most of the aforementioned decompositions are implemented in existing open-source packages, MODULO is currently the only package implementing the mPOD.


The theoretical foundation of the decomposition is described in
- M.A. Mendez, M. Balabane, J.-M. Buchlin, Multiscale Proper Orthogonal Decomposition of Complex Fluid Flows, Journal of Fluid Mechanics, Vol 870, July 2019, pp. 988-1036. The pre-print is available at https://arxiv.org/abs/1804.09646


To familiarize with the usage of MODULO, this repository contains four tutorials. Namely:


1. DMD and POD analysis of a 2D flow past a cylinder, simulated using 2D LES in openFOAM.
2. POD, SPODs and mPOD analysis of an impinging jet flow, obtained from a TR-PIV campaign at the von Karman Institute. This is one of test cases in [Mendez et al 2018](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/multiscale-proper-orthogonal-decomposition-of-complex-fluid-flows/D078BD2873B1C30B6DD9016E30B62DA8), available on arxiv at https://arxiv.org/abs/1804.09646. 
3. POD, mPOD and kPOD of the flow past a cylinder in transient conditions. This is the dataset analyzed in [Mendez et al 2020](https://iopscience.iop.org/article/10.1088/1361-6501/ab82be/meta)




