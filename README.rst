

MODULO: a python toolbox for data-driven modal decomposition
-----------------------------------------------------------

.. image:: https://readthedocs.org/projects/modulo/badge/?version=latest
    :target: https://modulo.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. raw:: html

   <div style="text-align: center;">
       <img src="https://modulo.readthedocs.io/en/latest/_images/modulo_logo.png" alt="Modulo Logo" width="500"/>
   </div>

**MODULO** is a modal decomposition package developed at the von Karman Institute for Fluid Dynamics (VKI). 
It offers a wide range of decomposition techniques, enabling users to select the most suitable method based 
on the specific physics of their problem and their desired outcomes. MODULO can natively handle large
datasets efficiently, featuring a memory-saving option that partitions the data and computes the decomposition in 
chunks (ninni2020modulo). Additionally, it supports non-uniform meshes through its weighted inner product formulation.

While the discontinued MATLAB version of MODULO (ninni2020modulo) is accessible in the “Old_Matlab_Implementation” branch, 
it is no longer maintained. The latest decomposition techniques are exclusively available in the current Python version.

As a part of the MODULO project, we provide a series of lectures on data-driven modal decomposition, and its applications.
These are available at the `MODULO YouTube channel <https://www.youtube.com/@modulompod5682>`_.


.. contents:: Table of contents

Modal decompositions
--------------------
Modal decompositions aim to describe the data as a linear combination of *modes*, obtained by projecting the data 
onto a suitable set of basis. For instance, consider a matrix $D(x, t)$, where $x$ and $t$ are the spatial and temporal
coordinates, respectively, the modal decomposition can be written as:

$D(x_i, t_k) = \\phi(x_i) \\Sigma \\psi(t_k)^T$

where $\\phi(x_i)$ is the spatial basis, $\\psi(t_k)$ is the temporal basis, and $\\Sigma$ is the modal coefficients. 
Different decompositions employ different basis, such as prescribed Fourier basis ($\\psi_\\mathcal{F}$) for 
the Discrete Fourier Transform (DFT), or data-driven basis, i.e. tailored on the dataset at hand, 
for the Proper Orthogonal Decomposition (POD). 

We refer to (mendez2022statistical, mendez2022generalizedmultiscalemodalanalysis, Mendez_2023) for an introduction to the topic.

MODULO currently features the following decompositions: 
- Discrete Fourier Transform (DFT) (briggs1995dft)
- Proper Orthogonal Decomposition (POD) (sirovich1987turbulence, berkooz1993proper)
- Multi-Scale Proper Orthogonal Decomposition (mPOD) (mendez2019multi)
- Dynamic Mode Decomposition (DMD) (schmid2010dynamic)
- Spectral Proper Orthogonal Decomposition (SPOD) (csieber2016spectral, towne2018spectral), note that the two are 
different formulations, and both are available in MODULO.
- Kernel Proper Orthogonal Decomposition (KPOD) (mika1998kernel)

We remind the curious reader to the respective references for a detailed description of each decomposition, and to the
documentation for a practical guide on how to use them in MODULO.


Release Notes
-------------
The latest version of MODULO (v2.0) includes the following updates:

1. **Faster EIG/SVD algorithms**, using powerful randomized svd solvers from scikit_learn 
    (see `here<https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_ 
    and `here <https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html>`_.) 
    It is now possible to select various options as "eig_solver" and "svd_solver", 
    offering different trade-offs in terms of accuracy vs computational time.

2. **Computation the POD directly via SVD**, using any of the four "svd_solver" options.
This is generally faster but requires more memory.

1. **Faster subscale estimators for the mPOD:** the previous version used the rank of the correlation matrix in each scale to define 
the number of modes to be computed in each portion of the splitting vector before assembling the full basis. 
This is computationally very demanding. This estimation has been replaced by a 
frequency-based threshold (i.e. based on the frequency bins within each portion) since one can show that the 
frequency-based estimator is always more "conservative" than the rank-based estimator.

1. **Major improvement on the memory saving option**: the previous version of modulo always required in input the matrix D. 
Then, if the memory saving option was active, the matrix was partitioned and stored locally to free the RAM before computing the 
correlation matrix (see [this tutorial by D. Ninni](https://www.youtube.com/watch?v=LclxO1WTuao)). 
In the new version, it is possible to initialize a modulo object *without* the matrix D (see exercise 5 in the examples). 
Instead, one can create the partitions without loading the matrix D.

1. **Implementation of Dynamic Mode Decomposition (DMD)** from (Schmid, P.J 2010)

2. **Implementation of the two Spectral POD formulations**, namely the one from (Sieber et al 2016), 
   and the one from (Towne et al 2018).

3. **Implementation of a kernel version of the POD**, in which the correlation matrix is replaced by a kernel matrix. 
This is described in Lecture 15 of the course `Hands on Machine Learning for Fluid dynamics 2023 <https://www.vki.ac.be/index.php/events-ls/events/eventdetail/552/-/online-on-site-hands-on-machine-learning-for-fluid-dynamics-2023>`_. 
We refer also to: `https://arxiv.org/abs/2208.07746`. 

1. **Implementation of a formulation for non-uniform meshes**, using a weighted matrix for all the relevant inner products. 
This is currently available only for POD and mPOD but allows for handling data produced from CFD simulation without resampling on a uniform grid (see exercise 4). 
It can be used both with and without the memory-saving option.


Installation
-------------

Installation via pip
^^^^^^^^^^^^^^^^^^^^

You can access the latest update of the modulo python package on PyPI using the command line:

.. code-block:: bash

    $ pip install modulo_vki

Installation from source 
^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can clone the repository and install the package locally:

.. code-block:: bash

    $ git clone https://github.com/mendezVKI/MODULO.git

    $ cd MODULO

    $ python setup.py install

or, if you have pip installed in your environment, 

.. code-block:: bash

    $ pip install .


Documentation
-------------

The documentation of MODULO is available `here <https://modulo.readthedocs.io/en/latest/intro.html>`_. It 
contains a comprehensive guide on how to install and use the package, as well as a detailed description of the
decompositions required inputs and outputs. A `list of YouTube videos<https://www.youtube.com/@modulompod5682>`_ 
is also available to guide the introduce the user to modal decomposition and MODULO.

Example 
-------------

Example 1: POD decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example illustrates how to decompose a data set (D) using the POD decomposition.

.. code-block:: python 

    from modulo_vki import ModuloVKI 
    import numpy as np

    # Create a random dataset
    D = np.random.rand(100, 1000)

    # Initialize the ModuloVKI object
    m = ModuloVKI(D) 

    # Compute the POD decomposition
    phi_POD, Sigma_POD, psi_POD = m.Compute_POD_K()

which returns the spatial basis ($\phi$), the temporal basis ($\psi$), and the modal 
amplitudes ($\Sigma$) of the POD decomposition. 

Example 2: Memory Saving option 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the Memory Saving option, MODULO decomposes $D$ in `N_partitions`, defined 
by the user (refer to `examples/ex_04_Memory_Saving.py`).

.. code-block:: python

    from modulo_vki import ModuloVKI 
    import numpy as np

    # Create a random dataset
    D = np.random.rand(100, 1000)
 
    # Initialize the ModuloVKI object
    m = ModuloVKI(D, N_PARTITIONS=10) 

    # Compute the POD decomposition
    phi_POD, Sigma_POD, psi_POD = m.Compute_POD_K()

Example 3: non-uniform grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are dealing with non-uniform grid (e.g. output of a Computational Fluid Dynamic (CFD) simulation),
you can use the weighted inner product formulation (refer to `examples/ex_05_nonUniform_POD.py`).

.. code-block:: python 

    from modulo_vki import ModuloVKI 
    import numpy as np

    # Create a random dataset
    D = np.random.rand(100, 1000)

    # Get the area of the grid
    a_dataSet = gridData.compute_cell_sizes()
    area = a_dataSet['Area']

    # Compute weights
    areaTot = np.sum(area)
    weights = area/areaTot # sum should be equal to 1

    # Initialize the ModuloVKI object
    m = ModuloVKI(D, weights=weights) 

    # Compute the POD decomposition
    phi_POD, Sigma_POD, psi_POD = m.Compute_POD_K()



Community guidelines
---------------------

Contributing to MODULO
^^^^^^^^^^^^^^^^^^^^^^^
We welcome contributions to MODULO. To create a new feature, please submit a pull request, specifying the proposed changes and 
providing an example of how to use the new feature (that will be included in the `examples/` folder).

The pull request will be reviewed by the MODULO team before being merged into the main branch, and your contribution duly acknowledged.

Report bugs 
^^^^^^^^^^^^
If you find a bug, or you encounter unexpected behaviour, please open an issue on the MODULO GitHub repository.

Ask for help
^^^^^^^^^^^^

Citation
---------
If you use MODULO in your research, please cite it as follows:

``Ninni, D., & Mendez, M. A. (2020). MODULO: A software for Multiscale Proper Orthogonal Decomposition of data. SoftwareX, 12, 100622.``

.. code-block:: text 

    @article{ninni2020modulo,
        title={MODULO: A software for Multiscale Proper Orthogonal Decomposition of data},
        author={Ninni, Davide and Mendez, Miguel A},
        journal={SoftwareX},
        volume={12},
        pages={100622},
        year={2020},
        publisher={Elsevier}
    }

We are currently working on a Journal of Open Source article that will be available soon.

References
----------

- Mendez, Miguel Alfonso. "Statistical Treatment, Fourier and Modal Decomposition." arXiv preprint arXiv:2201.03847 (2022).
- Mendez, M. A. (2023) "Generalized and Multiscale Modal Analysis". In : Mendez M.A., Ianiro, A., Noack, B.R., Brunton, S. L. (Eds), 
  "Data-Driven Fluid Mechanics: Combining First Principles and Machine Learning". Cambridge University Press, 2023:153-181. 
  https://doi.org/10.1017/9781108896214.013. The pre-print is available at https://arxiv.org/abs/2208.12630.
- Ninni, Davide, and Miguel A. Mendez. "MODULO: A software for Multiscale Proper Orthogonal Decomposition of data." SoftwareX 12 (2020): 100622.
- Mendez, Miguel A. "Linear and nonlinear dimensionality reduction from fluid mechanics to machine learning." Measurement Science and Technology 34.4 (2023): 042001. 
- Briggs, William L., and Van Emden Henson. The DFT: an owner's manual for the discrete Fourier transform. Society for Industrial and Applied Mathematics, 1995.
- Berkooz, Gal, Philip Holmes, and John L. Lumley. "The proper orthogonal decomposition in the analysis of turbulent flows." Annual review of fluid mechanics 25.1 (1993): 539-575.
- Sirovich, Lawrence. "Turbulence and the dynamics of coherent structures. III. Dynamics and scaling." Quarterly of Applied mathematics 45.3 (1987): 583-590.
- Mendez, M. A., M. Balabane, and J-M. Buchlin. "Multi-scale proper orthogonal decomposition of complex fluid flows." Journal of Fluid Mechanics 870 (2019): 988-1036.
- Schmid, Peter J. "Dynamic mode decomposition of numerical and experimental data." Journal of fluid mechanics 656 (2010): 5-28.
- Sieber, Moritz, C. Oliver Paschereit, and Kilian Oberleithner. "Spectral proper orthogonal decomposition." Journal of Fluid Mechanics 792 (2016): 798-828.
- Towne, Aaron, Oliver T. Schmidt, and Tim Colonius. "Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis." Journal of Fluid Mechanics 847 (2018): 821-867.
- Mika, Sebastian, et al. "Kernel PCA and de-noising in feature spaces." Advances in neural information processing systems 11 (1998).

Related projects
----------------
MODULO encapsulates a wide range of decomposition techniques, but not all of them. We refer to the project below for an additional set of decomposition techniques:

- ModRed, https://github.com/belson17/modred

There are also decomposition-specific projects, some of which are listed below:

- Rogowski, Marcin, Brandon CY Yeung, Oliver T. Schmidt, Romit Maulik, Lisandro Dalcin, Matteo Parsani, and Gianmarco Mengaldo. "Unlocking massively parallel spectral proper orthogonal decompositions in the PySPOD package." Computer Physics Communications 302 (2024): 109246.
- Lario, A., Maulik, R., Schmidt, O.T., Rozza, G. and Mengaldo, G., 2022. Neural-network learning of SPOD latent dynamics. Journal of Computational Physics, 468, p.111475.
- Ichinaga, Andreuzzi, Demo, Tezzele, Lapo, Rozza, Brunton, Kutz. PyDMD: A Python package for robust dynamic mode decomposition. arXiv preprint, 2024.
- Rogowski, Marcin, et al. "Unlocking massively parallel spectral proper orthogonal decompositions in the PySPOD package." Computer Physics Communications 302 (2024): 109246.

