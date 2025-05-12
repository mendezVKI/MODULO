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
