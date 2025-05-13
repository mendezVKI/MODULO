Discrete Fourier Transform (DFT)
=================================================

This document offers a brief reference guide to the use of the Discrete Fourier Transform (DFT).

.. .. toctree::
..    :maxdepth: 1 
..    :numbered:

..    theory.rst
..    basic_usage.rst

Theory
---------------------------------

The DFT is the only not strictly data-driven decomposition present in MODULO. In fact, the temporal basis :math:`\mathbf{\psi}_F` is the Fourier basis, computed *a priori*. 
This reads:

.. math:: 
    \mathbf{\psi}_F = \frac{1}{\sqrt{n_t}} \exp{(2 \pi f_r t)} 

in which :math:`f_r=r\Delta f` is the frequency of the :math:`r`-th Fourier mode, :math:`\Delta f=f_s/n_t` is the frequency resolution, and :math:`n_t` is the number of time samples. 
The term :math:`\frac{1}{\sqrt{n_t}}` is a normalization factor, which ensures that the Fourier basis is orthonormal, e.g. :math:`\|\mathbf{\psi}_F\|^2=1`.

Considering the DFT of the signal at one specific location :math:`\mathbf{x}_i`, the Fourier coefficients are computed as:

.. math:: 
    \mathbf{C}_F(\mathbf{x}_i) = \frac{1}{\sqrt{n_t}} \sum_{t=0}^{n_t-1} \mathbf{u}(\mathbf{x}_i,t) \exp{(-2 \pi f_r t)}

that effectively projects the signal onto the Fourier basis. For each of these projections, one can compute the norm as:

.. math::
    \sigma_F = \|\mathbf{C}_F\| 

that leads to the normalized projected fields:

.. math:: 
    \mathbf{\Phi}_F = \frac{\mathbf{C}_F}{\sigma_F}\;,

that are the spatial structure of the DFT. This completes the DFT decomposition, presented here as a matrix factorization:

.. math:: 
    \mathbf{D} = \mathbf{\Phi}_F \mathbf{\Sigma}_F \mathbf{\Psi}_F^T\;,

where :math:`\mathbf{\Sigma}_F` is a diagonal matrix containing the modal amplitudes :math:`\sigma_F`.


Example in MODULO
------------------
An example of the usage of the mPOD routine, extracted from the `examples` folder is reported below.

.. code-block:: python
    
    import numpy as np
    from modulo_vki.modulo import ModuloVKI
    from modulo_vki.utils import plot_mPOD

    FOLDER_DFT_RESULTS=FOLDER+os.sep+'DFT_Results_Jet'

    if not os.path.exists(FOLDER_DFT_RESULTS):
        os.mkdir(FOLDER_DFT_RESULTS)

    # We perform the DFT first
    # --- Remove the mean from this dataset (stationary flow )!
    D,D_MEAN=ReadData._data_processing(D,MR=True)
    # We create a matrix of mean flow (used to sum again for the videos):
    D_MEAN_mat=np.array([D_MEAN, ] * n_t).transpose()    

    # --- Initialize MODULO object
    m = ModuloVKI(data=D)
    # Compute the DFT
    Phi_F, Psi_F, Sigma_F = m.DFT(Fs)

