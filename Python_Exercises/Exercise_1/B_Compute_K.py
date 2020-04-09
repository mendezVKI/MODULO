# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 01:08:07 2019

@author: mendez
"""

# For a Cartesian Domain, we use no Weight Matrix. 
# Therefore the correlation is computed simply via matrix multiplication.
import numpy as np
data = np.load('Data.npz')
D=data['D']
K = np.dot(D.transpose(), D)

# Save as numpy array all the data
np.savez('Correlation_K',K=K)
 