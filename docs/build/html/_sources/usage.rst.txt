=====
Usage
=====

Start by importing MODULO. Then, you create a modulo object from which the decompositions can be carried out using any of
the methods (see "Computing decompositions"). For example, here is how to initialize the modulo object passing the dataset matrix D and use the method compute_POD_K() 
to perform the POD.

.. code-block:: python

    from modulo.modulo import MODULO    
    m = MODULO(data=D); Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_K()



 