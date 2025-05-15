MODULO (MODal mULtiscale pOd): Data-Driven Modal Analysis
==========================================================

MODULO (MODal mULtiscale pOd) is a software developed at the von Karman Institute for Fluid Dynamics to perform data-driven modal decompositions.
Initially focused on the Multiscale Proper Orthogonal Decomposition (mPOD), it has recently been extended to perform also other decompositions that include POD, SPODs, DFT, DMD, mPOD. 


.. toctree::
   :maxdepth: 2
   :caption: Getting Started  

   intro
   installation


.. toctree::
   :maxdepth: 1
   :caption: Data pre-processing 

   importing_data
   weights_inner


Data-Driven Modal Decompositions
---------------------------------

All modal decompositions are implemented in MODULO as matrix factorization of the snapshot matrix :math:`D(\mathbf{x}, t) \in \mathbb{R}^{n_S \times n_t}`.
Each entry of the snapshot matrix is a flattened realization of the data, regardless of the data dimensionality (2D, 3D). Thus, this decomposition
reads 

.. math:: 

   D(\mathbf{x}, t) = \mathbf{\Phi}\mathbf{\Sigma}\mathbf{\Psi}^T

where :math:`\mathbf{\Phi} \in \mathbb{R}^{n_S \times r}` and :math:`\mathbf{\Psi} \in \mathbb{R}^{n_t \times r}` are the spatial and temporal modes, 
respectively, and :math:`\mathbf{\Sigma} \in \mathbb{R}^{r \times r}` is a diagonal matrix containing the modal energy associated with each mode.

Each mode :math:`r`` is characterized by:

- **Spatial structure** :math:`\mathbf{\phi}_r`, which encodes the the pattern in space,  
- **Temporal structure** :math:`\mathbf{\psi}_r`, which describes how that pattern evolves in time,  
- **Amplitude** :math:`\sigma_r`, which quantifies the modal amplitude, e.g. energy or variance captured by the mode.  

The reminder of this section overviews the salient theoretical aspects of the implement modal decompositions, and the corresponding 
implementation in MODULO. The following decompositions are available:

The field of applications ranges from **reduced-order modeling** (ROM) and **flow control** to **filtering**, and **feature extraction**. 
For a comprehensive review of modal methods in fluid dynamics, see Mendez et al. (2023) :cite:`mendez_2023`.

.. toctree::
   :maxdepth: 1
   :caption: Data-Driven Modal Decompositions
   :glob:
   
   decompositions/dft/index
   decompositions/pod/index
   decompositions/dmd/index
   decompositions/mpod/index
   decompositions/kpod/index
   decompositions/spod/index

   .. decompositions/index


Key features of MODULO
----------------------
- **Multiple decomposition algorithms**:

  - **POD** (Proper Orthogonal Decomposition) via the snapshot method or SVD  
  - **mPOD** (Multiscale POD) for frequency‐band separation  
  - **SPOD** (Spectral POD)  
  - **DFT** (Discrete Fourier Transform)  
  - **DMD** (Dynamic Mode Decomposition)  
  - **KPOD** (Kernel POD)  

- **Memory‐saving implementations** :cite:`ninni_modulo_2020`:

  - Partition‐based loading for very large datasets  
  - On‐disk correlation matrices (Ninni et al., 2020) :cite:`ninni_modulo_2020`  
  

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/readdata
   api/dft 
   api/pod 
   api/mpod
   api/dmd
   api/spod 
   api/kpod  


Quick Start 
------------
MODULO is published on PyPI and can be installed with ``pip install modulo_vki`` (see :doc:`installation`). The user is first 
requested to assemble the matrix field to be decomposed, or the folder of the raw files to be imported if she wants to use the 
data loading routines present in the auxialliary code of MODULO. We refer the reader to :doc:`importing_data` for an example 
on the usage of these routines, and now we illustrate the case in which the matrix :math:`\mathbf{D}` is already available in memory.
In this case, the decomposition can be carried out as: 

.. code-block:: python 

    from modulo_vki import ModuloVKI # this is to create modulo objects
    
    D = np.array([]) # snapshot matrix containing the data 

    m = ModuloVKI(data=D)

    # Compute the DFT
    Phi_F, Psi_F, Sorted_Sigmas = m.DFT(Fs)

This output is consistent regardless of the decomposition, e.g. MODULO will always yield the spatial structures first,
then the temporal ones and lastly the amplitudes of the modes.


Collaborate on GitHub
----------------------

We welcome contributions to MODULO. 

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing/get_code
   contributing/testing 

The version changes are available here: 

.. toctree::
   :maxdepth: 1
   :caption: Changelog

   change_log 


References 
-----------

.. toctree::
   :maxdepth: 1
   :caption: References & Acknowledgements

   refs_and_ack/refs
   refs_and_ack/ack 


.. ++++++++++++++++++++++++++++++++++++++++++++++++++++
.. What are data-driven decompositions? Some literature
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++

.. Data-driven decompositions seek to partition a dataset as a linear combination of elementary contributions called modes.
.. For a brief overview of the context, we refer to: 

.. - Ninni, D., Mendez, M. A. (2020), "MODULO: A Software for Multiscale Proper Orthogonal Decomposition of data", Software X, Vol 12, 100622, https://doi.org/10.1016/j.softx.2020.100622.

.. - Poletti, R., Schena, L., Ninni, D., Mendez, M.A (2024) "MODULO: a python toolbox for data-driven modal decomposition", Submitted to Journal of Open Source Software. Preprint available at https://www.researchgate.net/publication/376885484_MODULO_a_python_toolbox_for_data-driven_modal_decomposition

.. The first article also presents the first version of MODULO and its GUI developed by D. Ninni (available in the Old_Matlab_Implementation branch). The second article introduces MODULO v2 from this branch and alternative open source projects.
.. While many projects allows for computing common decompositions such as POD, DMD and the SPODs, MODULO is currently the only opensource code allowing to compute the mPOD.

.. For a more comprehensive overview on the theory of data-driven decompositions, we refer to the chapter:

.. - Mendez, M. A. (2023) "Generalized and Multiscale Modal Analysis". In : Mendez M.A., Ianiro, A., Noack, B.R., Brunton, S. L. (Eds), "Data-Driven Fluid Mechanics: Combining First Principles and Machine Learning". Cambridge University Press, 2023:153-181. https://doi.org/10.1017/9781108896214.013. The pre-print is available at https://arxiv.org/abs/2208.12630. 

.. and the article that first presented the complete theory of the mPOD :

.. - Mendez, M. A., Balabane, M., Buchlin, J.-M. (2019) "Multi-Scale Proper Orthogonal Decomposition of Complex Fluid Flows" Journal of Fluid Mechanics 870:988-1036, https://doi.org/10.1017/9781108896214.013. The pre-print is available at https://arxiv.org/abs/2208.12630. 

.. The first version of the mPOD was based on wavelet decomposition and was presented here :

.. - Mendez, M.A.,  Scelzo, M.T., Buchlin, J.-M. , Multiscale Modal Analysis of an Oscillating Impinging Gas Jet, Experimental Thermal and Fluid Science, Vol 91, February 2018, pp. 256-276 (https://doi.org/10.1016/j.expthermflusci.2017.10.032).

.. Ongoing works on nonlinear methods are discussed here:

.. - Mendez, M. A. (2023) "Linear and Nonlinear Dimensionality Reduction from Fluid Mechanics to Machine Learning", Meas. Sci. Technol. 34(042001), https://doi.org/10.1088/1361-6501/acaffe. The pre-print is available at https://arxiv.org/abs/2208.07746.   

.. Some examples of applications of mPOD for processing experimental and numerical are listed below:

..  - Zhang ,Y., Ma, H., Guo, G. (2023), Multi-scale proper orthogonal decomposition (mPOD) analysis of vortex evolution and viscous dissipation in a circular-cylinder wake controlled by parallel symmetric jets, Ocean Engineering, 289(2), 116280 (https://doi.org/10.1016/j.oceaneng.2023.116280)
..  - Chi, C., Thevenin, D., Smits, A.J., Wolligandt, S., Theisel, H (2022),Identification and analysis of very-large-scale turbulent motions using multiscale proper orthogonal decomposition, Phys Rev. Fluids 7, 084603, https://doi.org/10.1103/PhysRevFluids.7.084603  
..  - Procaci, A., Kamal, M.M., Mendez, M.A., Hochgreg, S., Coussement, A., Parente, A. (2022) Multi-scale proper orthogonal decomposition analysis of instabilities in swirled and stratified flames, Physics of Fluids 34, 124103, https://doi.org/10.1063/5.0127956
..  - Barreiro-Villaverde, Gosset, A., Mendez, M.A. (2021), On the dynamics of jet wiping: Numerical simulations and modal analysis, Physics of Fluids 33, 062114, https://doi.org/10.1063/5.0051451
..  - Esposito, C., Mendez, M.A., Steelant, J., Vetrano, M.R. (2021), Spectral and modal analysis of a cavitating flow through an orifice, Experimental Thermal and Fluid Science, Vol 121, February 2021, 110251, https://doi.org/10.1016/j.expthermflusci.2020.110251
..  - Mendez, M.A., Hess, D.  Watz, B., Buchlin, J.-M. (2020)  Multiscale proper orthogonal decomposition (mPOD) of TR-PIV data—a case study on stationary and transient cylinder wake flows, Meas. Sci. Technol. 31 https://iopscience.iop.org/article/10.1088/1361-6501/ab82be/meta. Preprint at https://arxiv.org/abs/2001.01971.   
..  - Mendez, M.A., Gosset, A., Buchlin, J.-M. (2019) Experimental Analysis of the Stability of the Jet Wiping Process, Part II: Multiscale Modal Analysis of the Gas Jet-Liquid Film Interaction, Experimental Thermal and Fluid Science, Vol 106, September 2019, pp. 48-67 (https://doi.org/10.1016/j.expthermflusci.2019.03.004).
 
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++
.. Previous versions of MODULO
.. ++++++++++++++++++++++++++++++++++++++++++++++++++++

.. The first version was developed in Matlab and was equipped with a GUI by D. Ninni.
.. A minicourse on data-driven decompositions and the usage of MODULO was provided here:

.. https://www.youtube.com/watch?v=ED3x00H4yN4&list=PLEJZLD0-4PeKW6Ze984q08bNz28GTntkR&index=1

.. Although much of the material covered in these videos is still relevant, a new version of this minicourse is being prepared.
.. This is a compressed version of the course "Data-Driven Modal Analysis" that I give as part of the Research Master Program at the von Karman Institute (https://www.vki.ac.be/index.php/research-master-in-fluid-dynamics)

.. This first MODULO version implemented POD, DFT and mPOD and was already equipped with the first version of the memory-saving feature.
.. The GUI was also available as an executable that the user could install without needing a Matlab license.

.. This first version is still accessible from the branch "Old_Matlab_Implementation", but it is no longer maintained.

.. The main functions were then imported to Python, and L. Schena developed the first package PyPi.
.. A tutorial on how to use that version is given here: 

.. https://www.youtube.com/watch?v=y2uSvdxAwHk&list=PLEJZLD0-4PeKW6Ze984q08bNz28GTntkR&index=9

.. The current architecture of MODULO v2 is built from that version but has been considerably expanded. This first Python version is still accessible from the branch "Old_Python_Implementation" together with the first (unpackaged) versions of the codes. These are still used in some courses for didactic purposes.

