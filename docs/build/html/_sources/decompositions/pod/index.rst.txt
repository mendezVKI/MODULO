Proper Orthogonal Decomposition (POD)
=================================================

This document offers a brief reference guide to the use of the Proper Orthogonal Decomposition (POD).


.. .. toctree::
..    :maxdepth: 1

..    theory
..    basic_usage

Theory
--------------------------------------

The POD is an energy-based decomposition, which provides the modes capturing the largest amount of energy (variance) in the dataset,
such that the convergence of an approximation with increasing number of modes is optimal.

The temporal structures satisfying these properties are the eigenvectors of the temporal correlation matrix :math:`\mathbf{K}=\langle \mathbf{u}(\mathbf{x}, t), \mathbf{u}(\mathbf{x}, t)\rangle`,
where :math:`\langle \cdot \rangle` defines a suitable inner product. Thus,

.. math::
    \mathbf{K} \psi_{r}(t_k) = \lambda_r \psi_{r}(t_k) \quad r \in [1, R]

that yields to the factorization of :math:`\mathbf{K}` as:

.. math::
    \mathbf{K} = \sum_{r=1}^R \lambda_r \psi_r \psi_r^\top \;,

recalling that the correlation matrix is symmetric and positive definite. Under this condition, the POD is 'properly' orthogonal, that is,
also the spatial structures :math:`\psi_r` are orthonormal by construction.

Moreover, because of this proper orthogonality, the amplitudes of the modes are :math:`\sigma_r = \sqrt{\lambda_r}` and lead to the 
direct computation of the spatial structures via correlation:

.. math:: 
    \phi_r(\mathbf{x}) = \frac{1}{\sigma_r}\langle \mathbf{u}(\mathbf{x}, t), \psi_r(t) \rangle_T = \frac{1}{\sigma_r} \sum_{k=1}^{n_t} \mathbf{u}(\mathbf{x}, t) \psi_r(\mathbf{x})\;.
 

We note also that because of this joint orthonormality, one might decide to work with the temporal correlation matrix :math:`\mathbf{K}` or the spatial
analogue :math:`\mathbf{C}` depending on the shape of the dataset, and arrive to the same result.

An alternative path is offered by computing the SVD of the dataset :math:`\mathbf{D}`, yielding the same modes if the POD inner-product coincides with the 
standard Euclidean one used in the SVD procedure. 

.. note::

   By default, ``POD`` uses the temporal‐correlation approach; to switch to SVD, pass ``method='svd'``.


Example in MODULO
----------------

.. code-block:: python

    import numpy as np
    from modulo_vki.modulo import ModuloVKI
    
    # --- Initialize MODULO object
    m = ModuloVKI(data=np.nan_to_num(D))
    # Compute the POD using Sirovinch's method
    Phi_POD, Psi_POD, Sigma_POD = m.POD(mode='K')
    # compute the POD using SVD
    Phi_SVD, Psi_SVD, Sigma_SVD = m.POD(mode='SVD')


