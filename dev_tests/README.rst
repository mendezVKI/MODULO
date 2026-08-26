Release regression tests
========================

``test_all_modal.py`` runs the same external tutorial dataset with an installed
MODULO version and stores one ``.npz`` file per decomposition. MODULO 3.0 also
stores a fast-mPOD result. Run each version in its own clean environment:

.. code-block:: bash

   python dev_tests/test_all_modal.py \
       --version 2.1.5 \
       --dataset-dir /path/to/tutorial-data \
       --output-dir /tmp/modulo-regression

   python dev_tests/test_all_modal.py \
       --version 3.0.0 \
       --dataset-dir /path/to/tutorial-data \
       --output-dir /tmp/modulo-regression

Compare shared outputs and permit the expected new fast-mPOD artifact:

.. code-block:: bash

   python dev_tests/compare_all_modal.py \
       /tmp/modulo-regression/2.1.5 \
       /tmp/modulo-regression/3.0.0 \
       --rtol 1e-5 --atol 1e-8 --allow-extra-candidate

The comparison command returns a non-zero exit status for missing shared
outputs, shape/key mismatches, non-finite values, or numerical differences
outside tolerance. Eigenvector signs and complex phases are aligned before
comparison.

The normal pytest suite does not require the external tutorial data. It uses a
deterministic synthetic dataset to compare the four fast-mPOD backends and to
exercise the default no-Fortran installation path:

.. code-block:: bash

   python -m pytest -q

MODULO 3.0 release result
-------------------------

For the final 3.0.0 assessment, all saved arrays shared with published version
2.1.5 had a maximum absolute difference of zero on the 2,000-snapshot JET-PIV
dataset. This covered DFT, POD, classical mPOD, filtered-covariance SPOD, and
serial and parallel cross-spectral-density SPOD. Fast mPOD was the only new
candidate-only output. Tutorial 6 also completed without errors, generated 61
PDF figures, and reproduced its committed numerical reference arrays.
