Theory
--------------------------------------

The POD is an energy-based decomposition, which provides the modes capturing the largest amount of energy (variance) in the dataset,
such that the convergence of an approximation with increasing number of modes is optimal.

The temporal structures satisfying these properties are the eigenvectors of the temporal correlation matrix :math:`\mathbf{K}=\langle \mathbf{u}(x, t), \mathbf{u}(x, t)\rangle`,
where :math:`\langle \cdot \rangle` defines a suitable inner product. Thus,

.. math::
    \mathbf{K} \psi_{r}(t_k) = \lambda_r \psi_{r}(t_k) \quad r \in [1, R]

that yields to the factorization of :math:`\mathbf{K}` as:

.. math::
    \mathbf{K} = \sum_{r=1}^R \lambda_r \psi_r \psi_r^\top \;,

recalling that the correlation matrix is symmetric and positive definite. Under this condition, the POD is'properly' orthogonal, that is,
also the spatial structures :math:`\psi_r` are orthonormal by construction. 

Moreover, because of this proper orthogonality, the amplitudes of the modes are :math:`\sigma_r = \sqrt{\lambda_r}` and lead to the 
direct computation of the spatial structures via correlation:

.. math:: 
    \psi_r(x) = \frac{1}{\sigma_r}\langle \mathbf{u}(x, t), \psi_r(t) \rangle_T = \frac{1}{\sigma_r} \sum_{k=1}^{n_t} \mathbf{u}(x, t) \phi_r(x)\;.
 

We note also that because of this joint orthonormality, one might decide to work with the temporal correlation matrix :math:`\mathbf{K}` or the spatial
analogue :math:`\mathbf{C}` depending on the shape of the dataset, and arrive to the same result.

An alternative path is offered by computing the SVD of the dataset :math:`\mathbf{D}`, yielding the same modes if the POD inner-product coincides with the 
standard Euclidean one used in the SVD procedure. 

.. note::

   By default, ``POD`` uses the temporal‐correlation approach; to switch to SVD, pass ``method='svd'``.
