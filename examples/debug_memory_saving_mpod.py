# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:03:48 2023

@author: mendez
"""


import numpy as np
import matplotlib.pyplot as plt 
import os  # To create folders an delete them


from tqdm import tqdm

from modulo.utils.read_db import ReadData # to read the data
from modulo.modulo import MODULO
from modulo.utils.others import plot_grid_cylinder_flow,Plot_Field_TEXT_Cylinder


FOLDER='Tutorial_5_2D_Cylinder_Memory_Saving'


#%% Partition the data into blocks
n_t=13200; Fs=3000; dt=1/Fs 
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 

#%% Get info about mesh and data, and construct D or partitions

# --- Component fields (N=2 for 2D velocity fields, N=1 for pressure fields)
N = 2 
# --- Number of mesh points and snapshots
nxny = 2130; N_T=n_t
# --- Header (H),  footer (F) and columns (C) to be skipped during acquisition
H = 1; F = 0; C=0
# --- Read one sample snapshot (to get N_S)
Name = FOLDER+os.sep+"Res00001.dat"
Dat = np.genfromtxt(Name, skip_header=H, skip_footer=F)


Name_Mesh=FOLDER+os.sep+'MESH.dat'
Name_FIG='Cylinder_Flow_snapshot_'+str(2)+'.png'

# This function reads all the info about the grid
n_s,Xg,Yg,Vxg,Vyg,X_S,Y_S=Plot_Field_TEXT_Cylinder(Name,Name_Mesh,Name_FIG) 
# number of points in x and y
n_x,n_y=np.shape(Xg)



#% Check 1 : there is no problem in the splitting: D is ok.

# Prepare matrix D (1 partition)
D = ReadData._from_dat(folder=FOLDER, filename='Res%05d', 
                       N=2, N_S=2*nxny,N_T=N_T,
                       h = H, f = F, c=C,
                       N_PARTITIONS=1, MR= True)

# Now do the same with the partitions, then read them back:
D_p = ReadData._from_dat(folder=FOLDER, filename='Res%05d', 
                       N=2, N_S=2*nxny,N_T=N_T,
                       h = H, f = F, c=C,
                       N_PARTITIONS=10, MR= True)    


# Read the various blocks
N_PARTITIONS=10
N_S,N_T=np.shape(D)
if N_T % N_PARTITIONS != 0:
    tot_blocks_col = N_PARTITIONS + 1
else:
    tot_blocks_col = N_PARTITIONS


D_blocks=np.zeros((2*nxny,N_T))
R1=0
for k in tqdm(range(tot_blocks_col)):
    di = np.load(f"MODULO_tmp/data_partitions/di_{k + 1}.npz")['di']
    R2 = R1 + np.shape(di)[1]
    D_blocks[:, R1:R2] = di
    R1 = R2

# Check that they are the same
Check=np.allclose(D_blocks,D)

# This shows that test is passed. No prob on splitting K


#%% Check 2: there is no problem in inner prod: K is ok.

# we now compute the matrix K in two ways.
# Full computation:
K=D.T.dot(D)
# Modulo computation
from modulo.core._k_matrix import CorrelationMatrix

K2=CorrelationMatrix(N_T, 1, False,
                           './', D)
Check2=np.allclose(K,K2)
    

K3=CorrelationMatrix(N_T, 10, True,
                           './MODULO_tmp', True, D)
# In this case we have to read it 

K3=np.load("./MODULO_tmp/correlation_matrix/k_matrix.npz")['K']

Check3=np.allclose(K,K3)


# mPOD Settings
# Frequency splitting Vector
F_V=np.array([10,290,320,430,470])
# This vector collects the length of the filter kernels.
Nf=np.array([1201,801,801,801,801]); 
Keep=np.array([0,0,1,0,0])
Ex=1203; boundaries = 'nearest'; MODE = 'reduced'


# Study the temporal bases
from modulo.core._mpod_time import temporal_basis_mPOD

PSI_M1=temporal_basis_mPOD(K, Nf, Ex, F_V, Keep, boundaries,
                       MODE, dt, './MODULO_tmp', False, False,
                       100,100)

PSI_M2=temporal_basis_mPOD(K3, Nf, Ex, F_V, Keep, boundaries,
                       MODE, dt, './MODULO_tmp', False, False,
                       100,5,'eigh')

check6=np.allclose(PSI_M1,PSI_M2)
print(check6)
# Case 1: no memory saving




# This shows that the test is passed. No prob on K

#%% Check 3: there is no problem in projection: Phi is ok

# We now compute the Phi's in the two cases. 


# Compute the mPOD with full matrix
m = MODULO(data=D,N_T=N_T,
           N_S=2*nxny,
           n_Modes=10,
           N_PARTITIONS=10,
           MEMORY_SAVING=False)

Phi_M,Psi_M,Sigmas_M = m.compute_mPOD(Nf,
                                      Ex,
                                      F_V,
                                      Keep,
                                      boundaries,
                                      MODE,1/Fs)

# Now repeat the same with the memory saving
m2 = MODULO(data=[],N_T=N_T,
           N_S=2*nxny,
           n_Modes=10,
           N_PARTITIONS=10,
           MEMORY_SAVING=True)

Phi_M2,Psi_M2,Sigmas_M2 = m2.compute_mPOD(Nf,
                                      Ex,
                                      F_V,
                                      Keep,
                                      boundaries,
                                      MODE,1/Fs)

Check4=np.allclose(np.abs(Phi_M),np.abs(Phi_M2))
check5=np.allclose(np.abs(Psi_M2),np.abs(Psi_M))

# this test was not passed. We go inside the functions.
# in fact, the psi's are not the same even if K's are ! 
Ind=2
#plt.plot(Psi_M2[:,Ind],Psi_M[:,Ind],'ko')
plt.figure()
plt.plot(Psi_M2[:,Ind])
plt.plot(Psi_M[:,Ind])

plt.figure()
plt.plot(Sigmas_M2,'ko')
plt.plot(Sigmas_M,'rs')




#%% We finally close with the phi's

r=3
#%% Take the mode's structure
Phi_M_Mode_U=np.real(Phi_M[0:nxny,r]).reshape((n_y,n_x))
Phi_M_Mode_V=np.real(Phi_M[nxny::,r]).reshape((n_y,n_x))
# Assign to Vxg, Vyg which will be plot
Vxg=Phi_M_Mode_U; Vyg=Phi_M_Mode_V
plot_grid_cylinder_flow(Xg.T, Yg.T, Vxg, Vyg)
ax = plt.gca()
Tit_string='$\phi_{\mathcal{M}'+str(r)+'}(\mathbf{x}_i)$'
plt.title(Tit_string,fontsize=17)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
plt.show()

#%% Take the mode's structure
Phi_M_Mode_U=np.real(Phi_M2[0:nxny,r]).reshape((n_y,n_x))
Phi_M_Mode_V=np.real(Phi_M2[nxny::,r]).reshape((n_y,n_x))
# Assign to Vxg, Vyg which will be plot
Vxg=Phi_M_Mode_U; Vyg=Phi_M_Mode_V
plot_grid_cylinder_flow(Xg.T, Yg.T, Vxg, Vyg)
ax = plt.gca()
Tit_string='$\phi_{\mathcal{M}'+str(r)+'}(\mathbf{x}_i)$'
plt.title(Tit_string,fontsize=17)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
plt.show()







