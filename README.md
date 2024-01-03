

MODULO - latest update 2.0
===================

This repository contains version 2.0 of MODULO (MODal mULtiscale pOd), a software developed at the von Karman Institute to perform data-driven modal decompositions and, in particular, the Multiscale Proper Orthogonal Decomposition (MPD).

The old version based on MATLAB implementation and related GUI is no longer maintained but will remain available on the branch "Old_Matlab_Implementation". We also keep the first Python implementation in the branch "Old_Python_Implementation". See the Readme file in these branches for more information.

### What is MODULO, and what are data-driven decompositions?

MODULO allows to compute data-driven decompositions of experimental and numerical data. To have a concise overview of the context, we refer to: 

- Ninni, D., Mendez, M. A. (2020), "MODULO: A Software for Multiscale Proper Orthogonal Decomposition of data", Software X, Vol 12, 100622, https://doi.org/10.1016/j.softx.2020.100622.

This article also presents the first version of MODULO (available in the OLD_Matlab_Implementation branch) and its GUI developed by D. Ninni. 
For a more comprehensive overview of data-driven decompositions, we refer to the chapter:

- Mendez, M. A. (2023) "Generalized and Multiscale Modal Analysis". In : Mendez M.A., Ianiro, A., Noack, B.R., Brunton, S. L. (Eds), "Data-Driven Fluid Mechanics: Combining First Principles and Machine Learning". Cambridge University Press, 2023:153-181. https://doi.org/10.1017/9781108896214.013. The pre-print is available at https://arxiv.org/abs/2208.12630. 


### What is new in this V 2.0? 

This version expands considerably the version v1 in "Old_Python_Implementation", for which a first tutorial was provided by L. Schena in https://www.youtube.com/watch?v=y2uSvdxAwHk. 
The major updates are the following :

1. Faster EIG/SVD algorithms, using powerful randomized svd solvers from scikit_learn (see [this](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) and [this](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html) ). It is now possible to select various options as "eig_solver" and "svd_solver", offering different trade-offs in terms of accuracy vs computational time.

2. In addition to the traditional POD computation using the K matrix (Sirovinch's method), it is now possible to compute the POD directly via SVD using any of the four "svd_solver" options.
This is generally faster but requires more memory.

3. Faster subscale estimators for the mPOD: the previous version used the rank of the correlation matrix in each scale to define the number of modes to be computed in each portion of the splitting vector before assembling the full basis. This is computationally very demanding. This estimation has been replaced by a frequency-based threshold (i.e. based on the frequency bins within each portion) since one can show that the frequency-based estimator is always more "conservative" than the rank-based estimator.

4. Major improvement on the memory saving option: the previous version of modulo always required in input the matrix D. Then, if the memory saving option was active, the matrix was partitioned and stored locally to free the RAM before computing the correlation matrix (see [this tutorial by D. Ninni](https://www.youtube.com/watch?v=LclxO1WTuao)). In the new version, it is possible to initialize a modulo object *without* the matrix D (see exercise 5 in the examples). Instead, one can create the partitions without loading the matrix D.

4. Implementation of Dynamic Mode Decomposition (DMD) from [Schmid, P.J 2010](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4).

4. Implementation of the two Spectral POD formulations, namely the one from [Sieber et al 2016](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition/DCD8A6EDEFD56F5A9715DBAD38BD461A), and the one from [Towne et al 2018](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition-and-its-relationship-to-dynamic-mode-decomposition-and-resolvent-analysis/EC2A6DF76490A0B9EB208CC2CA037717)

5. Implementation of a kernel version of the POD, in which the correlation matrix is replaced by a kernel matrix. This is described in Lecture 15 of the course [Hands on Machine Learning for Fluid dynamics 2023](https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552/-/online-on-site-hands-on-machine-learning-for-fluid-dynamics-2023). See also [this](https://arxiv.org/abs/2208.07746).

6. Implementation of a formulation for non-uniform meshes, using a weighted matrix for all the relevant inner products. This is currently available only for POD and mPOD but allows for handling data produced from CFD simulation without resampling on a uniform grid (see exercise 4). It can be used both with and without the memory-saving option.


### New Tutorials 

The installation provides with 5 exercises to explore all the features of MODULO while familizarizing with data driven decompositions. These are available in the /exercise/ folder, both in plain python format and in jupyter notebooks. 

- Exercise 1. In this exercise we consider the flow past a cylinder. This is by far the most popular test case because it's well known to have a simple low order representation, with modes that have nearly harmonic temporal structure. We compute the POD and the DMD and compare the results... the difference between DMD and POD modes is hardly distinguishable !.
The data for this test case was produced by Denis Dumoulin during his STP project at VKI in 2016 (Report available on request).

- Exercise 2. We consider the flow of an impinging gas jet, taken from [this](https://arxiv.org/abs/1804.09646) paper. This  

- Exercise 3.

- Exercise 4.

- Exercise 5.





