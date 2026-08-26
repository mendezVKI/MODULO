Development and Testing
========================

Quick local checks
------------------

The normal test suite is deterministic and does not download external data:

.. code-block:: bash

   python -m pytest -q
   python -m pytest -q modulo_vki/test/test_fast_mpod.py
   PYTHONWARNINGS=error::SyntaxWarning \
     python -m compileall -f -q modulo_vki dev_tests

The fast-mPOD tests compare the ``fullK``, ``bandK``, ``fullSVD``, and
``randSVD`` backends, check reconstruction and temporal orthogonality, and
exercise the default installation without the optional Fortran extension.

Cross-version release regression
--------------------------------

``dev_tests/test_all_modal.py`` runs the decompositions on the external
2,000-snapshot JET-PIV tutorial dataset and stores one ``.npz`` result per
method. Run it once with the published 2.1.5 package and once with the 3.0.0
candidate, using separate environments and working directories:

.. code-block:: bash

   python dev_tests/test_all_modal.py \
       --version 2.1.5 \
       --dataset-dir /path/to/tutorial-data \
       --output-dir /tmp/modulo-regression

   python dev_tests/test_all_modal.py \
       --version 3.0.0 \
       --dataset-dir /path/to/tutorial-data \
       --output-dir /tmp/modulo-regression

The dataset directory must contain
``Tutorial_2_JET_PIV/Ex_4_TR_PIV_Jet``. The dataset is available from
``https://osf.io/c28de/download``.

Compare the shared outputs while allowing fast mPOD as the expected new
candidate-only result:

.. code-block:: bash

   python dev_tests/compare_all_modal.py \
       /tmp/modulo-regression/2.1.5 \
       /tmp/modulo-regression/3.0.0 \
       --rtol 1e-5 --atol 1e-8 --allow-extra-candidate

The comparison exits with a non-zero status for missing files or keys, shape
mismatches, non-finite values, or numerical differences outside tolerance.
Real eigenvector signs and complex phases are aligned before comparison.

MODULO 3.0 release assessment
-----------------------------

The final 3.0.0 release assessment produced zero maximum absolute difference
between 2.1.5 and 3.0.0 for every saved JET-PIV result from DFT, POD, classical
mPOD, filtered-covariance SPOD, and serial and parallel cross-spectral-density
SPOD. The fast-mPOD result was the sole expected candidate-only artifact.

``Tutorial_6_fast_mPOD.ipynb`` also executed without errors, produced 61 PDF
figures, and reproduced all committed numerical reference arrays. These
external-data checks complement the normal pytest suite and should be rerun
before a final release tag is created.

