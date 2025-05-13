Versions
================================


2.1.0
------
- `mPOD` bug fix: the previous version of the mPOD was skipping the last scale of the frequency splitting vector. Fixed in this version.
- CSD - SPOD can now be parallelized, leveraging `joblib`. The user needs just to pass the argument `n_processes` for the computation to be
  split between different workers.

2.0.7
------
Major updates:

1. Faster EIG/SVD algorithms, using powerful randomized svd solvers from scikit_learn (see `TruncatedSVD
<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_
and 
`randomized_svd
<https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html>`_) . It is now possible to select various options as "eig_solver" and "svd_solver", offering different trade-offs in terms of accuracy vs computational time.

1. In addition to the traditional POD computation using the K matrix (Sirovinch's method), it is now possible to compute the POD directly via SVD using any of the four "svd_solver" options. This is generally faster but requires more memory.

2. Much faster subscale estimators for the mPOD: the previous version used the rank of the correlation matrix in each scale to define the number of modes to be computed in each portion of the splitting vector. This is computationally very demanding. The estimation has been replaced by a frequency-based threshold (i.e. based on the frequency bins within each portion) since one can show that the frequency-based estimator is always more "conservative" than the rank-based estimator. Combining this estimator with a stochastic solver for eigendecomposition at each scale makes the mPOD much, much faster than before!

3. Major improvement on the memory saving option: the previous version of modulo always required in input the matrix D. Then, if the memory saving option was active, the matrix was partitioned and stored locally to free the RAM before computing the correlation matrix (see this tutorial by D. Ninni https://www.youtube.com/watch?v=LclxO1WTuao). In the new version, it is possible to initialize a modulo object and create the data partitions  *without* loading the matrix D (see exercise 4 in the examples). This means that the only bottle neck is provided by the number of snapshots...!!
   
4. Implementation of Dynamic Mode Decomposition (DMD) from Schmid, P.J 2010: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4).

5. Implementation of the two Spectral POD formulations, namely the one from Sieber et al 2016: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition/DCD8A6EDEFD56F5A9715DBAD38BD461A, and the one from Towne et al 2018: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/spectral-proper-orthogonal-decomposition-and-its-relationship-to-dynamic-mode-decomposition-and-resolvent-analysis/EC2A6DF76490A0B9EB208CC2CA037717

6. Implementation of a kernel version of the POD, in which the correlation matrix is replaced by a kernel matrix. This is described in Lecture 15 of the course *Hands on Machine Learning for Fluid dynamics 2023*: https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552/-/online-on-site-hands-on-machine-learning-for-fluid-dynamics-2023. See also https://arxiv.org/abs/2208.07746).

7. Implementation of a formulation for non-uniform meshes, using a weighted matrix for all the relevant inner products. This is currently available only for POD and mPOD which allows for handling data produced from CFD simulation without resampling on a uniform grid (see exercise 5). It can be used both with and without the memory-saving option.


MODULO < 2.0
-------------------------------------

The first version was developed in Matlab and was equipped with a GUI by D. Ninni.
A minicourse on data-driven decompositions and the usage of MODULO was provided here:

https://www.youtube.com/watch?v=ED3x00H4yN4&list=PLEJZLD0-4PeKW6Ze984q08bNz28GTntkR&index=1

Although much of the material covered in these videos is still relevant, a new version of this minicourse is being prepared.
This is a compressed version of the course "Data-Driven Modal Analysis" that I give as part of the Research Master Program at the von Karman Institute (https://www.vki.ac.be/index.php/research-master-in-fluid-dynamics)

This first MODULO version implemented POD, DFT and mPOD and was already equipped with the first version of the memory-saving feature.
The GUI was also available as an executable that the user could install without needing a Matlab license.

This first version is still accessible from the branch "Old_Matlab_Implementation", but it is no longer maintained.

The main functions were then imported to Python, and L. Schena developed the first package PyPi.
A tutorial on how to use that version is given here: 

https://www.youtube.com/watch?v=y2uSvdxAwHk&list=PLEJZLD0-4PeKW6Ze984q08bNz28GTntkR&index=9

The current architecture of MODULO v2 is built from that version but has been considerably expanded. This first Python version is still accessible from the branch "Old_Python_Implementation" together with the first (unpackaged) versions of the codes. These are still used in some courses for didactic purposes.

