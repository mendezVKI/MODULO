Theory
-------------------------------------------------

The main idea of the mPOD is to perform a POD on non-overlapping portions of the frequency domain of the signal :cite:`ninni_modulo_2020`.
Considering a signal :math:`\mathbf{u}(t)` defined in the time domain, sampled with a sampling frequency :math:`f_s`, one can define to 
isolate phenomena of interest belonging in different frequency bands identified by :math:`M` frequency bands. This divides the frequency
in blocks, :math:`[0, f_1]`, :math:`[f_1, f_2]`, ..., :math:`[f_{M-1}, f_M]` where :math:`f_1, f_2, \ldots, f_M` are the frequencies of interest.

The mPOD performs a multi-resolution analysis (MRA) on the temporal correlation matrix :math:`\mathbf{K} = \mathbf{D}^T \mathbf{D} \in \mathbb{R}^{n_t \times n_t}`. Given
a list of suitable transfer functions :math:`\left\{H_m\right\}_{m=1}^M`, the mPOD splits the matrix :math:`\mathbf{K}` as:

.. math:: 
    \mathbf{K} = \sum_{m=1}^M \mathbf{K}_m = \sum_{m=1}^M \mathbf{\Psi}_F \big[ \hat{\mathbf{K}} \odot H_m \big] \mathbf{\Psi}_F

where :math:`\mathbf{\Psi}_F` is the Fourier transform matrix, :math:`\hat{\mathbf{K}}=\bar{\mathbf{\Psi}}_F \mathbf{K} \bar{\mathbf{\Psi}}_F` is the
2D Fourier transform of the correlation matrix, and :math:`\odot` is the entry by entry product.

The filtering ensures that the key properties of :math:`\mathbf{K}` are preserved, and thus characterised with a set of orthonormal eigenvectors
and eigenvalues: 

.. math:: 
    \mathbf{K}_m = \sum_{j=1}^{n_m} \lambda_{m,j} \Psi_{P, mj} \Psi_{P, mj}^T 

and :math:`n_m` is the number of nonzero eigenvalues at each scale. This procedure also ensures that the eigenvectors of all scales
are orthogonal complements that span :math:`\mathbb{R}^{n_t}`. 

Finally, the mPOD temporal basis is constructed by collecting the POD basis of all scales, such that:

.. math:: 
    \mathbf{\Psi}_M = [\mathbf{\Psi}_{1}, \mathbf{\Psi}_{2}, \ldots, \mathbf{\Psi}_{M}] \mathbf{P}_\mathbf{\Sigma}

where :math:`\mathbf{P}_\mathbf{\Sigma}` is a permutation matrix that ranks the structures in decreasing order of energy contribution. 

In conclusion, we note that the mPOD generalizes POD and DFT: for :math:`M=1`, the mPOD is equivalent to POD, while for :math:`M=n_t`, 
the mPOD is equivalent to DFT.

fast mPOD
^^^^^^^^^

The fast mPOD follows the spectral formulation introduced in Belda *et al.* by replacing the classical FIR filter bank with compact spectral masks :math:`M_m` acting on strictly disjoint frequency supports. Starting from the spectral representation of the temporal correlation matrix,

.. math::
   \mathbf{K}_F = \mathbf{\Psi}_F \mathbf{K} \mathbf{\Psi}_F,

the contribution of the :math:`m`-th band is written as

.. math::
   \mathbf{K}_{F,m} = M_m \, \mathbf{K}_F \, M_m.

Because the masks have compact, non-overlapping support, :math:`\mathbf{K}_{F,m}` contains only :math:`n^{(m)}` active frequencies and reduces to a compact operator :math:`\mathbf{K}_{F,c}`. The eigenvalue problem is then solved on this reduced block,

.. math::
   \mathbf{K}_{F,c}\,\widehat{\mathbf{\Psi}}_m
   =
   \widehat{\mathbf{\Psi}}_m \mathbf{\Sigma}_m^2,

and the temporal modes are finally recovered as

.. math::
   \mathbf{\Psi}_m = \mathbf{\Psi}_F \widehat{\mathbf{\Psi}}_m.

Compared with the classical mPOD, the essential difference is therefore that each scale is no longer treated through a full :math:`n_t \times n_t` eigenvalue problem, but through an independent reduced problem whose size is set only by the number of active frequencies in that band. This is the key idea behind the fast spectral formulation proposed in Belda *et al (2026)*.