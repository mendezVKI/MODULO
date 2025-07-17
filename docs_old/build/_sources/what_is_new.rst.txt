================================
What is new in this V 2.0? 
================================


The major updates are the following :

1. Faster EIG/SVD algorithms, using powerful randomized svd solvers from scikit_learn (see `TruncatedSVD
<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_
and 
`randomized_svd
<https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html>`_) . It is now possible to select various options as "eig_solver" and "svd_solver", offering different trade-offs in terms of accuracy vs computational time.

2. In addition to the traditional POD computation using the K matrix (Sirovinch's method), it is now possible to compute the POD directly via SVD using any of the four "svd_solver" options. This is generally faster but requires more memory.

3. Much faster subscale estimators for the mPOD: the previous version used the rank of the correlation matrix in each scale to define the number of modes to be computed in each portion of the splitting vector. This is computationally very demanding. The estimation has been replaced by a frequency-based threshold (i.e. based on the frequency bins within each portion) since one can show that the frequency-based estimator is always more "conservative" than the rank-based estimator. Combining this estimator with a stochastic solver for eigendecomposition at each scale makes the mPOD much, much faster than before!

4. Major improvement on the memory saving option: the previous version of modulo always required in input the matrix D. Then, if the memory saving option was active, the matrix was partitioned and stored locally to free the RAM before computing the correlation matrix (see this tutorial by D. Ninni https://www.youtube.com/watch?v=LclxO1WTuao). In the new version, it is possible to initialize a modulo object and create the data partitions  *without* loading the matrix D (see exercise 4 in the examples). This means that the only bottle neck is provided by the number of snapshots...!!
   
5. Implementation of Dynamic Mode Decomposition (DMD) from Schmid, P.J 2010: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4).

6. Implementation of the two Spectral POD formulations, namely the one from Sieber et al 2016: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition/DCD8A6EDEFD56F5A9715DBAD38BD461A, and the one from Towne et al 2018: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition-and-its-relationship-to-dynamic-mode-decomposition-and-resolvent-analysis/EC2A6DF76490A0B9EB208CC2CA037717

7. Implementation of a kernel version of the POD, in which the correlation matrix is replaced by a kernel matrix. This is described in Lecture 15 of the course *Hands on Machine Learning for Fluid dynamics 2023*: https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552/-/online-on-site-hands-on-machine-learning-for-fluid-dynamics-2023. See also https://arxiv.org/abs/2208.07746).

8. Implementation of a formulation for non-uniform meshes, using a weighted matrix for all the relevant inner products. This is currently available only for POD and mPOD which allows for handling data produced from CFD simulation without resampling on a uniform grid (see exercise 5). It can be used both with and without the memory-saving option.




