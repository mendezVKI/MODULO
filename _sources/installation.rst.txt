Installation
============

MODULO requires Python 3.10 or newer. The standard installation is pure Python
and works without a Fortran compiler, including on Windows:

.. code-block:: bash

   python -m pip install modulo_vki

To install the current source tree instead:

.. code-block:: bash

   git clone https://github.com/mendezVKI/MODULO.git
   cd MODULO
   python -m pip install -e ".[tutorials]"

This editable tutorial installation includes an IPython kernel and
``nbconvert`` so the notebooks can be opened in an IDE or executed from the
command line.

Optional dependencies
---------------------

Tutorial and VTK dependencies are deliberately kept out of the base package:

.. code-block:: bash

   python -m pip install "modulo_vki[tutorials]"
   python -m pip install "modulo_vki[vtk_utils]"

Optional Fortran acceleration
-----------------------------

Fast mPOD uses the NumPy implementation by default. The optional Fortran
extension is never compiled or imported during a normal installation. This is
the recommended configuration on Windows and is selected with the default
``useFortran=False`` argument.

Linux and macOS users who want to build the accelerator need a GNU-compatible
Fortran compiler and OpenBLAS in addition to the Python build tools:

.. code-block:: bash

   python -m pip install "modulo_vki[fortran]"
   python -m modulo_vki.fortran.compile_fortran

The build helper accepts explicit BLAS linker flags when OpenBLAS cannot be
detected automatically. For example:

.. code-block:: bash

   python -m modulo_vki.fortran.compile_fortran --blas "-L/path/to/lib -lopenblas"

After a successful build, opt in for a supported K-based backend:

.. code-block:: python

   model = ModuloVKI(data, n_Modes=10, dtype="float64")
   phi, psi, sigma = model.fastmPOD(
       frequency_splits,
       sampling_frequency,
       mode="bandK",
       useFortran=True,
   )

The MODULO build helper intentionally does not compile the extension on
Windows. Windows users should install the base package and retain
``useFortran=False``.

Development installation
------------------------

For tests and documentation builds:

.. code-block:: bash

   python -m pip install -r requirements-dev.txt
   python -m pip install -e .
   python -m pytest -q
   python -m sphinx -W -b html docs/source docs/build/html
