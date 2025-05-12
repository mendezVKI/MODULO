POD Example
=============

.. code-block:: python

    import numpy as np
    from modulo_vki.modulo import ModuloVKI
    
    # --- Initialize MODULO object
    m = ModuloVKI(data=np.nan_to_num(D))
    # Compute the POD using Sirovinch's method
    Phi_POD, Psi_POD, Sigma_POD = m.POD(mode='K')
    # compute the POD using SVD
    Phi_SVD, Psi_SVD, Sigma_SVD = m.POD(mode='SVD')


