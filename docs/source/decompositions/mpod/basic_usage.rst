Example in MODULO
------------------

Classical mPOD
^^^^^^^^^^^^^^

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


Fast mPOD
^^^^^^^^^

Fast mPOD is fully functional without Fortran. The default
``useFortran=False`` path is portable and is the supported configuration on
Windows. See :doc:`../../installation` before opting into the Linux/macOS
Fortran accelerator.

An example of the usage of the fast mPOD routine, extracted from the `examples` folder is reported below.

.. code-block:: python

    import numpy as np
    from modulo_vki.modulo import ModuloVKI
    import os
    from math import pi

    if not os.path.exists("./ex6_results"):
        os.makedirs("./ex6_results")

    # Create artificial spatial structure
    toposes = np.zeros((3000,3))
    toposes[:1000,0] = 1
    toposes[1000:2000,1] = 1
    toposes[2000:,2] = 1
    toposes = toposes/np.linalg.norm(toposes,axis=0)

    # Associate importance with the test modes
    sigmas = np.diag([6000,4000,2000])

    # Create artificial temporal structure
    nt = 6000   # number of time steps
    chronoses = np.zeros((nt,3))
    for i in range(2000):
        chronoses[i,0] = np.sin(2*pi*i/60)+(i-500)/1200     # f1/fs = 1/60 + slope
    for i in range(4000,6000):
        chronoses[i,1] = np.sin(2*pi*(i-4000)/30)-(i-4000)/800   # f2/fs = 1/30 + slope

    # single wavelet
    for i in range(2200,3000):
        chronoses[i,2] = 1
    for i in range(3000,3800):
        chronoses[i,2] = -1    

    for i in range(3):
        chronoses[:,i] = chronoses[:,i]/np.linalg.norm(chronoses[:,i])

    # Assemble the data matrix
    D = toposes @ sigmas @ chronoses.T

    # Save temporal evolutions
    chronTot = np.zeros((chronoses.shape[0],chronoses.shape[1]+1))
    chronTot[:,0] = np.arange(nt)
    chronTot[:,1:] = chronoses
    np.save(f"{target_dir}/chronoses.npy", chronTot)

    # Frequency split and selection of scales
    nModes = 10     # Number of modes to compute
    F_V = [1/120,3/120,5/120,10/120]   # Frequency splitting vector in Hz, assuming fs = 1 (normalized frequencies) for this tutorial
    Keep = [1, 1, 1, 1, 0]  # Which scales to keep (0 means not processed, 1 means processed)
    winType = "hann" # Type of window for the tapering (hann, hamming, blackman, etc)
    taper = [1/300, 1/300, 1/300, 1/300, 1/300] # Taper width in Hz for each scale, assuming fs = 1 (normalized frequencies) for this tutorial
    mode = "fullK" # Type of computation (supports "fullK", "bandK", "fullSVD", "randSVD")


    # MODULO object initialization when NOT using Fortran.
    m = ModuloVKI(data=np.nan_to_num(D), n_Modes = nModes)

    # Fast mPOD call
    phi, psi, sig = m.fastmPOD(F_V, 1, Keep, winType = winType, taper = taper, mode = mode, GThresh=5, ncpus=9) 


An example of the usage of the fast mPOD routine with Fortran support for correlation matrix assembly. Only the parts that differ from the previous example are shown below.

.. code-block:: python

    # MODULO object initialization when USING Fortran.
    m = ModuloVKI(data=np.nan_to_num(D), n_Modes = nModes, dtype=np.float64) # dtype = np.float64 is MANDATORY for Fortran support.

    # Fast mPOD call
    phi, psi, sig = m.fastmPOD(F_V, fs, Keep, winType = winType, taper = taper, mode = mode, GThresh=5, ncpus=9, useFortran=True)
