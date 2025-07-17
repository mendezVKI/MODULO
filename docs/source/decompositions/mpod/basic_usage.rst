Example in MODULO
------------------
An example of the usage of the mPOD routine, extracted from the `examples` folder is reported below.

.. code-block:: python

    import numpy as np
    from modulo_vki.modulo import ModuloVKI
    from modulo_vki.utils import plot_mPOD

    FOLDER_MPOD_RESULTS=FOLDER+os.sep+'mPOD_Results_Jet'
    if not os.path.exists(FOLDER_MPOD_RESULTS):
        os.mkdir(FOLDER_MPOD_RESULTS)

    # We here perform the mPOD as done in the previous tutorials.
    # This is mostly a copy paste from those, but we include it for completenetss

    # Updated in MODULO 2.1.0: mPOD requires now Keep and Nf of size len(F_V) + 1, to ensure all scales are computed. First entry specifies the low pass filter.

    Keep = np.array([1, 1, 1, 1, 1])
    Nf = np.array([201, 201, 201, 201, 201])
    # --- Test Case Data:
    # + Stand off distance nozzle to plate
    H = 4 / 1000  
    # + Mean velocity of the jet at the outlet
    U0 = 6.5  
    # + Input frequency splitting vector in dimensionless form (Strohual Number)
    ST_V = np.array([0.1, 0.2, 0.25, 0.4])  
    # + Frequency Splitting Vector in Hz
    F_V = ST_V * U0 / H
    # + Size of the extension for the BC (Check Docs)
    Ex = 203  # This must be at least as Nf.
    dt = 1/2000; boundaries = 'reflective'; MODE = 'reduced'
    # Here 's the mPOD
    Phi_M, Psi_M, Sigmas_M = m.mPOD(Nf, Ex, F_V, Keep, 20 ,boundaries, MODE, dt, False)

The variable `Keep` is a vector of size `len(F_V) + 1`, defines whether the scale is processed or not, 
while `Nf` is a vector of size `len(F_V) + 1` is the number of points in the frequency domain, and 
`Ex` is the size of the extension for the BC. The boundary conditions can be set to `reflect`, `nearest`,
`wrap` or `extract`.