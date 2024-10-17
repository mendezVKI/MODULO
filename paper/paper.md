---
title: 'MODULO: A Python toolbox for data-driven modal decomposition'
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
Dimensionality reduction is an essential tool in processing large datasets, enabling data compression, pattern recognition, and reduced-order modeling. Many linear tools for dimensionality reduction have been developed in fluid mechanics, where they have been formulated to identify coherent structures and build reduced-order models of turbulent flows [@berkooz_proper_1993]. 
This work proposes a major upgrade of the software package MODULO (MODal mULtiscale pOd) [@ninni_modulo_2020], which was designed to perform Multiscale Proper Orthogonal Decomposition (mPOD) [@mendez_balabane_buchlin_2019]. In addition to implementing the classic Fourier Transform (DFT) and Proper Orthogonal Decomposition (POD), MODULO now also allows for computing Dynamic Mode Decomposition (DMD) [@schmid_2010] as well as the Spectral POD by @sieber_paschereit_oberleithner_2016, the Spectral POD by @Towne_2018 and a generalized kernel-based decomposition akin to kernel PCA [@mendez_2023]. All algorithms are wrapped in a ‘SciKit’-like Python API, which allows computing all decompositions in one line of code. Documentation, exercises, and video tutorials are also provided to offer a primer on data drive modal analysis.

# Statement of Need
As extensively illustrated in recent reviews [@mendez_2023; @Taira2020], all modal decompositions can be considered as special matrix factorizations. The matrix being factorized collects (many) snapshots (samples) of a high-dimensional variable. The factorization provides a basis for the matrix's column and row spaces to identify the most essential patterns (modes) according to a certain criterion. In what follows, we will refer to common terminologies in fluid dynamics. Nevertheless, it is worth stressing that these tools can be applied to any high-dimensional dataset to identify patterns and build reduced-order models [@mendez_balabane_buchlin_2019]. In the common arrangement encountered in fluid dynamics, the basis for the column space is a set of ‘spatial structures’ while the basis for the row space is a set of `temporal structures'. These are paired by a scalar, which defines their relative importance. The POD, closely related to Principal Component Analysis, yields modes with the highest energy (variance) content and, in addition, guarantees their orthonormality by construction.
In the DFT, as implemented in MODULO, modes are defined to evolve as orthonormal complex exponential in time. This implies that the associated frequencies are integer multiples of a fundamental tone. The DMD generalizes the DFT by releasing the orthogonality constraint and considering complex frequencies, i.e., modes that can vanish or explode. 
Both the constraint of energy optimality and harmonic modes can lead to poor convergence and feature detection performances. This motivated the development of hybrid methods such as the Spectral POD by @Towne_2018, Spectral POD by @sieber_paschereit_oberleithner_2016, and Multiscale Proper Orthogonal Decomposition (mPOD) [@mendez_balabane_buchlin_2019]. The first can be seen as an optimally averaged DMD, while the second combines POD and DFT with a filtering operation. Both SPODs assume statistically stationary data and are designed to identify harmonic (or quasi-harmonic) modes. The mPOD combines POD with Multi-resolution Analysis (MRA) to provide optimal modes within a prescribed frequency band. The mPOD modes are thus spectrally less narrow than those obtained by the SPODs, but this allows for localizing them in time (i.e., potentially having compact support in time). 
Finally, recent developments in nonlinear methods such as kernel PCA and their applications to fluid dynamics (see @mendez_2023) have motivated the interest in the connection between nonlinear methods and the most general Karhunen-Loeve expansion (KL). This generalizes the POD as the decomposition of data onto the eigenfunction of a kernel function (the POD being a KL for the case of linear kernel). 


MODULO provides a unified tool to carry out different decompositions with a shared API. This simplifies comparing different techniques and streamlines their application to a given dataset (problem). In addition, it is the only package that includes the mPOD and the generalized KL with kernel functions interfacing with SciKit-learn. For decomposition-specific packages, we refer the reader to many excellent Python APIs that are available to compute the POD, DMD, and both SPODs, for example [@py_DMD; @Mengaldo2021; @SpyOD; @rogowski2024unlocking].


# New Features 
This manuscript accompanies MODULO version 2.0. This version features four new decompositions: the two SPODs, the DMD, and the general KL. It also allows for different approaches to computing the POD (interfacing with various SVD/EIG solvers from Scipy and SciKit-learn) and a first implementation for nonuniform grids. The memory-saving feature has been improved, and the software can now handle 3D decompositions.

# Conclusions
MODULO is a versatile and user-friendly toolbox for data-driven modal decomposition. It provides a unified interface to various decomposition methods, allowing for a straightforward comparison and benchmarking. The package allows for modal decompositions in one line of code. It is also designed to handle large datasets via the so-called "memory saving" option and can handle nonuniformly sampled data. 

# Acknowledgements
R. Poletti and L. Schena are supported by Fonds Wetenschappelijk Onderzoek (FWO), grant numbers 1SD7823N and 1S75825N, respectively.

# References
