---
title: 'MODULO: a python toolbox for data-driven modal decomposition'
tags:
  - Python
  - fluid dynamics 
  - modal decomposition

authors:
  - name: R. Poletti
    orcid: 0000-0003-3566-6956
    #equal-contrib: true
    corresponding: False # (This is how to denote the corresponding author)
    affiliation: "1, 2" 
  - name: L. Schena
    orcid: 0000-0002-7183-0242
    #equal-contrib: true
    corresponding: False # (This is how to denote the corresponding author)
    affiliation: "1, 3"
  - name: D. Ninni
    orcid: 0000-0002-7179-3322
    #equal-contrib: true
    corresponding: False # (This is how to denote the corresponding author)
    affiliation: 4
  - name: M. A. Mendez
    orcid: 0000-0002-1115-2187
    #equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1


affiliations:
 - name: von Karman Insitute for Fluid Dynamics
   index: 1
 - name: University of Ghent, Belgium
   index: 2
 - name: Vrije Universiteit Brussel (VUB), Belgium
   index: 3
 - name: Politecnico di Bari, Italy
   index: 4
date: 05 December 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
Dimensionality reduction is an essential tool in processing large datasets, enabling data compression, pattern recognition and reduced order modelling. Many linear tools for dimensionality reduction have been developed in fluid mechanics, where they have been formulated to identify coherent structures and build reduced-order models of turbulent flows [@berkooz_proper_1993]. 
This work proposes a major upgrade of the software package MODULO (MODal mULtiscale pOd,[@ninni_modulo_2020]), which was designed to perform Multiscale Proper Orthogonal Decomposition (mPOD)[@mendez_balabane_buchlin_2019]. In addition to implementing the classic Fourier Transform (DFT) and Proper Orthogonal Decomposition (POD), MODULO now also allows for computing Dynamic Mode Decomposition (DMD) [@schmid_2010] as well as the Spectral POD by [@sieber_paschereit_oberleithner_2016], the Spectral POD by [@Towne_2018] and a generalized kernel-based decomposition akin to kernel PCA [@mendez_2023]
All algorithms are wrapped in a ‘SciKit’-like Python API which allows computin all decompositions in one line of code. Documentation, exercises, and video tutorials are also provided to offer a primer on data drive modal analysis.

# Statement of Need
As extensively illustrated in recent reviews [@mendez_2023], [@Taira2020], all modal decompositions can be seen as special kinds of matrix factorizations. The matrix being factorized collects (many) snapshots (samples) of a large dimensional variable.
The factorization provides a basis for the column and the row spaces of the matrix, with the goal of identifying the most essential patterns (modes) according to a certain criterion. In the common arrangement encountered in fluid dynamics, the basis for the column space is a set of ‘spatial structures’ while the basis for the row space is a set of `temporal structures'. These are paired by a scalar which defines their relative importance. 
In the POD, closely related to Principal Component Analysis, the modes the ones that have the largest amplitudes while remaining orthonormal both in space and time.  In the DFT, as implemented in MODULO, modes are defined to evolve as orthonormal complex exponential in time. This implies that the associated frequencies are integer multiples of a fundamental tone. The DMD generalizes the DFT by releasing the constraint of orthogonality and considering complex frequencies, i.e., modes that can potentially vanish or decay. 
Both the constraint of energy optimality and harmonic modes can lead to poor performances in terms of convergence and feature detection. This motivated the development of hybrid methods such as the the Spectral POD by [@Towne_2018], Spectral POD by [@sieber_paschereit_oberleithner_2016], Multiscale Proper Orthogonal Decomposition (mPOD)[@mendez_balabane_buchlin_2019]. The first can be seen as an optimally averaged DMD while the second consists in bringing POD and DFT with the use of a filtering operation. Both the SPODs assume statistically stationary data and are designed to identify harmonic (or quasi-harmonic) modes. The mPOD combines POD with Multi-resolution Analysis (MRA), to provide modes that are optimal within a pre-scribed frequency band. The mPOD modes are thus spectrally less narrow than those obtained by the SPODs, but this allows for localizing them in time (i.e. potentially having compact support in time).
Finally, recent developments on nonlinear methods such as kernel PCA and their applications to fluid dynamics (see [@mendez_2023]) have motivated the interest on the connection between nonlinear methods and the most general Karhunen–Loeve expansion (KL). This generalizes the POD as the decomposition of data onto the eigenfunction of a kernel functions (the POD being a KL for the case of linear kernel). 
While many excellent Python APIs are available to POD, DMD and both SPODs (see for example [@py_DMD], [@Mengaldo2021], [@SpyOD]), MODULO is currently the only one for computing multiscale POD along with all the others, including the generalized KL with kernel functions interfacing with SciKit-learn.


# New Features 
The new release implements four new decompositions: the two SPODs, the DMD and the general KL. It also allows for different approaches to compute the POD (interfacing with various SVD/EIG solvers from Scipy and ScikiLearn) as well as a first implementation for nonuniform grids. The memory saving feature has been improved and the software can now handle 3D decompositions.


