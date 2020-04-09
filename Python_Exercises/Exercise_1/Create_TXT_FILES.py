# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:28:27 2019

@author: mendez
"""

# This file generates the output in the format used by the MODULO GUI
import numpy as np

data = np.load('Data.npz')
D=data['D']
t=data['t']
dt=data['dt']
n_t=data['n_t']
y=data['y']
dy=data['dy']
n_y=data['n_y']

Fol_Out='Exercise_1_txts'
import os
Fol_Out = 'Ex_1_1D_Analytical'
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)

# Save the Mesh file
Name=Fol_Out+os.sep + 'Mesh.dat'
np.savetxt(Name, np.around(y,6), fmt="%s",header="mesh", comments='')

for k in range(0,n_t):
   u=np.around(D[:,k],4)
   Name=Fol_Out+os.sep + 'EX_1_Step%03d' % (k)+'.dat'
   np.savetxt(Name, u, fmt="%s",header="data", comments='')
   print('Export '+str(k)+' of '+ str(n_t))