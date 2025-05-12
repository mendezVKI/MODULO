Dynamic Mode Decomposition (DMD)
=================================================

This document offers a brief reference guide to the use of the Dynamic Mode Decomposition (DMD).

Theory
----------

The DMD seeks to overcome the `spectral-mixing` induced by the POD compression, while simultaneously avoid _windowing_ problems
caused by the dependency of the DFT on the fundamental tone of the data to be decomposed. The main idea introduced by Schmid (2010) :cite:`schmid_2010`,
is to fit a linear system onto a reduced POD projection of the dataset. 

The algorithm we implement in MODULO is the one proposed by Tu et al. (2013) :cite:`tu2013dynamic`, and a more robust theoretical formulation can also be found 
in Mendez (2021) :cite:`mendez2022statistical`.

This method creates two portions of the snapshot matrix :math:`\mathbf{D}` that differ of one step in time, that is:

.. math::
    \mathbf{D}_1 = [\mathbf{d}_1, \dots, \mathbf{d}_{n_t - 1}] \in \mathbb{R}^{n_S \times n_t - 1} \;,

and 

.. math::
    \mathbf{D}_2 = [\mathbf{d}_2, \dots, \mathbf{d}_{n_t}] \in \mathbb{R}^{n_S \times n_t - 1} .

Then, the linear system propagator :math:`\mathbf{P}: \mathbf{d}_{t} \rightarrow \mathbf{d}_{t+\Delta t} \in \mathbb{R}^{n_S \times n_S}` can 
be solved by least square. This can be computed via POD: 

.. math::
    \mathbf{P} \approx \mathbf{D}_2 \mathbf{\Psi}_P \mathbf{\Sigma}_P^{-1} \mathbf{\Phi}_P^\top \;.

Yet, this can be ill-conditioned and computationally prohibitive to compute and it is further projected onto a reduced POD space spanned 
by the basis :math:`\tilde{\mathbf{\Phi}} = [\phi^\bullet, \dots, \phi^\bullet_{R}]`, where :math:`n_r \ll n_t`. This leads to a system 
in :math:`\mathbb{R}^{n_r \times n_r}` wich reads: 

.. math::
    \mathbf{D}_2 \approx \mathbf{P}{\mathbf{D}}_1 \rightarrow \tilde{\mathbf{\Phi}}_P^\top \mathbf{D}_2 \approx \tilde{\mathbf{\Phi}}_P^\top \mathbf{P} \tilde{\mathbf{\Phi}}_P \tilde{\mathbf{\Phi}}_P^\top \mathbf{D}_1 \;,

that we can compact writing:

.. math::
    \tilde{\mathbf{V}}_2 \approx \tilde{\mathbf{S}} \tilde{\mathbf{V}}_1 \;. 

The reduced-order propagator is then:

.. math::
    \tilde{\mathbf{S}} = \tilde{\mathbf{\Phi}}_P^\top \mathbf{P} \tilde{\mathbf{\Phi}}_P = \tilde{\mathbf{\Phi}}_P^\top \mathbf{D}_2 \tilde{\mathbf{\Psi}} \tilde{\mathbf{\Sigma}}^{-1} \;.

The eigenvalue decomposition of this propagator governs the evolution of the reduced system, and the eigenvalues dictate the stability of each mode: 
:math:`\|\lambda_r\|<1` implies vanishing modes, :math:`\|\lambda_r\|>1` exploding ones and :math:`\|\lambda_r\|=1` are purely harmonic (like DFT). Note that,
in this latter case, the frequency is not constrained to be a multiple of the fundamental tone :math:`f_0 = 1/T = 1/n_t \Delta t`.

At last, the complex spatial structures of the DMD can be computed by mapping to the original space using the POD basis: 

.. math:: 
    \mathbf{\Phi}_D = \tilde{\Phi}_P \mathbf{Q} 

where :math:`\mathbf{Q}` are the eigenvectors of the propagator :math:`\tilde{\mathbf{S}}`.


.. bibliography:: references.bib
   :style: unsrt
   :cited:


Example in MODULO 
------------------

.. code-block:: python 

    # --- Initialize MODULO object
    m = ModuloVKI(data=np.nan_to_num(D))
    # Compute the POD using PIP's method
    Phi_D, Lambda, freqs, a0s = m.DMD(save_dmd, F_S=F_S)

The ``DMD`` module returns the spatial structures :math:`\mathbf{\Phi}_D`, the complex 
eigenvalues :math:`\lambda_r` of the reduced-order propagator :math:`\tilde{\mathbf{S}}`,
the frequency associated with each DMD mode, and their amplitudes :math:`\mathbf{a}_0`.
