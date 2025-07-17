================================
New Tutorial Exercises
================================


The installation provides five exercises to explore MODULO's features while familiarizing with data-driven decompositions. These are available in the /exercise/ folder in plain Python format and jupyter notebooks. 
Some of these are adapted from the previous exercises  `in our YouTube channel <https://www.youtube.com/channel/UC-RoU7LisZSLy6o-EO4BUDA/featured>`_, while others are new.

Tutorials
--------------

- **Exercise 1**. In this exercise, we consider the flow past a cylinder. The dataset was created via Large Eddy Simulations (LES) by Denis Dumoulin during his STP at VKI in 2016 (Report available on request). For convenience, the data was first mapped to a Cartesian grid. This test case is by far the most popular because it's well-known to have a simple low-order representation with modes that have nearly harmonic temporal structures. We compute the POD and the DMD and compare the results... the difference between DMD and POD modes is hardly distinguishable!

- **Exercise 2**. We consider the flow of an impinging gas jet, taken from https://arxiv.org/abs/1804.09646. This dataset was collected via Time-Resolved Particle Image Velocimetry (TR-PIV). Only the first 200 POD modes were stored. This dataset has much richer dynamics than the previous one and cannot be easily approximated using a few modes. We use it to explore the differences between the DFT, the SPODs and the mPOD. These have different purposes and look for different features.

- **Exercise 3**. We take back the cylinder test case to explore the differences between the POD and the generalized Karhunen–Loève (KL) expansion in which a kernel matrix replaces the correlation matrix. The POD is a particular case of KL where the kernel function generating the kernel matrix is the plain inner product. Here, we also consider a Gaussian kernel. Different kernel functions define similarity in different ways and thus produce widely different modes. Different modal structures tell different stories about the dataset, but... what can you say about efficiency in data compression? 

- **Exercise 4**. We consider the flow past a cylinder again, but this time in transient conditions and on an experimental test case taken from https://arxiv.org/abs/2001.01971. In this exercise, you can reproduce the same results from the article to see how the mPOD allows to achieve both time and frequency localization without compromising much of the convergence of the POD. The dataset is quite large, so you might have difficulties handling it if you have less than 32 GB of RAM. But fear not: the memory saving feature allows to compute POD and mPOD without loading the data into memory!

- **Exercise 5**. We consider the flow of an impinging gas jet again, but this time on a numerical test case. This dataset was produced by Yannic Lowenstein during his STP at VKI at the end of 2023, with the help of Dr. Maria Faruoli. The Reynolds number is two orders of magnitude higher than in exercise 2, yet the flow features you will observe are pretty similar, at least qualitatively. From a learning perspective, the key feature of this test case is that the data is not available on a uniform grid. But fear not: with the new features, it is possible to compute the decompositions using appropriate weights!
 


