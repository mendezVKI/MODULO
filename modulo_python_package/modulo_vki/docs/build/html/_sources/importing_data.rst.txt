===================================
Importing data
===================================

MODULO accepts ``numpy.array`` data matrices only. If you do not have it already, ``ReadData`` is here to help: specifying where your files
are stored and other details it takes care of converting your single files into a processable matrix.

.. autoclass:: read_db.ReadData

Currently, it supports ``.dat``, ``.csv`` and ``.txt`` files. 

.. autofunction:: read_db.ReadData._from_dat

.. autofunction:: read_db.ReadData._from_csv

.. autofunction:: read_db.ReadData._from_txt


