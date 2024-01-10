=======================================
Importing data and D matrix preparation
=======================================

MODULO can operate with two options.

**Option 1**: the snapshot matrix is provided by the users. MODULO accepts ``numpy.array`` data matrices only.
We use this feature in Example 1.

**Option 2**: a folder containing the data is provided and MOUDLO must assemble the snapshot mtrix.
We use this feature in Example 2.

When operating with MEMORY_SAVING=True, MODULO will refrain from working on them full matrix D.
If this matrix was provided (Option 1), but MEMORY_SAVING=True, MODULO will break the data into chunks, store it in a local folder and delete it from memory.
If only the data folder was provided (Option 2), then MODULO will assemble the various portions and store them in a local folder.

The format expected for the data is ``.dat`` or ``.txt``.  Variants for ``.csv`` files are in preparation.

The codes to generate the matrix D are availble in  ``ReadData``. The main function is 
*_data_processing*, documented below:

.. autofunction:: modulo.utils.read_db.ReadData._data_processing
