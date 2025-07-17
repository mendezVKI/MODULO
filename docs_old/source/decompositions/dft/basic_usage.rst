mPOD example
========================
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

