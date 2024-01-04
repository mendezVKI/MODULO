# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:46:01 2023

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt 
import os  # To create folders an delete them

from modulo.utils.read_db import ReadData # to read the data

# This exercise uses the memory saving feature.
# The idea is to compute all the decomposition without never assembling the
# matrix D. This makes the computation much slower, because of the time
# lost in reading/saving things on the disk, but it allows to process 
# datasets that are too large to fit in you RAM.

# In MODULO, the memory saving feature is currently implemented for
# the POD (if done via matrix K) and mPOD.

from modulo.modulo import MODULO
from modulo.utils.others import plot_grid_cylinder_flow,Plot_Field_TEXT_Cylinder
# Download the dataset locally


### Plot Customization (Optional )
fontsize = 16
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['font.size'] = fontsize


#%% Get the data
FOLDER='Tutorial_5_2D_Cylinder_Memory_Saving'

# Script 1: Get the Data
import urllib.request
print('Downloading PIV data TR PIV Cyl...')
url = 'https://osf.io/47ftd/download'
urllib.request.urlretrieve(url, 'Ex_5_TR_PIV_Cylinder.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_5_TR_PIV_Cylinder.zip'
zf = ZipFile(String,'r'); 
zf.extractall('./DATA_CYLINDER'); zf.close()

os.rename('DATA_CYLINDER', FOLDER) # rename the data flolder to FOLDER
os.remove(String) # Delete the zip file with the data 
print('Data set unzipped and ready ! ')

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


# Prepare 10 partitions
D = ReadData._data_processing(database=None,FOLDER_IN=FOLDER, filename='Res%05d',
                       N=2, N_S=2*nxny,N_T=N_T,
                       h = H, f = F, c=C,
                       N_PARTITIONS=10, MR= False,FOLDER_OUT='./MODULO_tmp')

# --- Initialize MODULO object
m = MODULO(data=D,N_T=N_T,
           N_S=2*nxny,
           n_Modes=100,
           N_PARTITIONS=10,eig_solver='svd_sklearn_randomized')

# compute the POD
Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_K()

#%% Post Process the POD

# Prepare the frequencies 
Freqs = np.fft.fftfreq(N_T) * Fs  # Compute the frequency bins

FOLDER_POD_RESULTS=FOLDER+os.sep+'POD_Results_Cylinder_PIV'
if not os.path.exists(FOLDER_POD_RESULTS):
    os.makedirs(FOLDER_POD_RESULTS)


# Plot the spatial structures and the spectra of the first 3 modes
for r in range(5):
 #%% Take the mode's structure
 Phi_P_Mode_U=np.real(Phi_POD[0:nxny,r]).reshape((n_y,n_x))
 Phi_P_Mode_V=np.real(Phi_POD[nxny::,r]).reshape((n_y,n_x))
 # Assign to Vxg, Vyg which will be plot
 Vxg=Phi_P_Mode_U; Vyg=Phi_P_Mode_V
 plot_grid_cylinder_flow(Xg.T , Yg.T, Vxg, Vyg)
 ax = plt.gca()
 Tit_string='$\phi_{\mathcal{P}'+str(r)+'}(\mathbf{x}_i)$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name=FOLDER_POD_RESULTS+os.sep+'POD_Phi_'+str(r)+'R.png'
 plt.savefig(Name, dpi=200) 
 plt.show()

 #%% Plot the temporal evolution of the structure 
 fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
 plt.plot(t,-Psi_POD[:,r])
 plt.xlim([0,t.max()])
 #ax.set_xscale('log')
 plt.xlabel('$ t [s]$',fontsize=18)
 plt.ylabel('$\psi_{\mathcal{P}r}$',fontsize=18)
 Tit_string='$\psi_{\mathcal{P}'+str(r)+'}(t_k)$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name=FOLDER_POD_RESULTS+os.sep+'POD_Psi_'+str(r)+'R.png'
 plt.savefig(Name, dpi=200) 
 plt.show()

 #%% Plot the spectral content of the structure 
 fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
 psi_hat=np.fft.fft(Psi_POD[:,r]*Sigma_POD[r]/n_t)
 plt.plot(Freqs,20*np.log10(psi_hat),'ko')
 plt.xlim([0,1500])
 plt.ylim([-125,50])
 #ax.set_xscale('log')
 plt.xlabel('$ f_n [Hz]$',fontsize=18)
 plt.ylabel('$\mathcal{F}\{\psi_{\mathcal{P}r}\} [db]$',fontsize=18)
 Tit_string='$\mathcal{F}\{\psi\}_{\mathcal{P}'+str(r)+'}(f_n)$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name=FOLDER_POD_RESULTS+os.sep+'POD_Psi_hat_'+str(r)+'R.png'
 plt.savefig(Name, dpi=200) 
 plt.show()

#%% compute the mPOD

# mPOD Settings
# Frequency splitting Vector
F_V=np.array([10,290,320,430,470])
# This vector collects the length of the filter kernels.
Nf=np.array([1201,801,801,801,801]); 
Keep=np.array([1,0,1,1,0])
Ex=1203; boundaries = 'nearest'; MODE = 'reduced'
# Compute the mPOD 
m = MODULO(None,
           N_T=13200,
           N_S=2*nxny,
           N_PARTITIONS=10,
           n_Modes = 500,           
           eig_solver='svd_sklearn_randomized')

Phi_M,Psi_M,Sigmas_M = m.compute_mPOD(Nf=Nf,
                                      Ex=Ex,
                                      F_V=F_V,
                                      Keep=Keep,
                                      SAT=5,
                                      boundaries=boundaries,
                                      MODE=MODE,dt=1/Fs,SAVE=True)


#%% Post Process the mPOD 

FOLDER_mPOD_RESULTS=FOLDER+os.sep+'mPOD_Results_Cylinder_PIV'
if not os.path.exists(FOLDER_mPOD_RESULTS):
    os.makedirs(FOLDER_mPOD_RESULTS)

# Plot the spatial structures and the spectra of the first 3 modes
for r in range(5):
 #%% Take the mode's structure
 Phi_M_Mode_U=np.real(Phi_M[0:nxny,r]).reshape((n_y,n_x))
 Phi_M_Mode_V=np.real(Phi_M[nxny::,r]).reshape((n_y,n_x))
 # Assign to Vxg, Vyg which will be plot
 Vxg=Phi_M_Mode_U; Vyg=Phi_M_Mode_V
 plot_grid_cylinder_flow(Xg.T , Yg.T, Vxg, Vyg)
 ax = plt.gca()
 Tit_string='$\phi_{\mathcal{M}'+str(r)+'}(\mathbf{x}_i)$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name=FOLDER_mPOD_RESULTS+os.sep+'mPOD_Phi_'+str(r)+'R.png'
 plt.savefig(Name, dpi=200) 
 plt.show()

 #%% Plot the temporal evolution of the structure 
 fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
 plt.plot(t,-Psi_M[:,r])
 plt.xlim([0,t.max()])
 #ax.set_xscale('log')
 plt.xlabel('$ t [s]$',fontsize=18)
 plt.ylabel('$\psi_{\mathcal{M}r}$',fontsize=18)
 Tit_string='$\psi_{\mathcal{M}'+str(r)+'}(t_k)$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name=FOLDER_mPOD_RESULTS+os.sep+'mPOD_Psi_'+str(r)+'R.png'
 plt.savefig(Name, dpi=200) 
 plt.show()

 #%% Plot the spectral content of the structure 
 fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
 psi_hat=np.fft.fft(Psi_M[:,r]*Sigmas_M[r]/n_t)
 plt.plot(Freqs,20*np.log10(psi_hat),'ko')
 plt.xlim([0,1500])
 plt.ylim([-125,50])
 #ax.set_xscale('log')
 plt.xlabel('$ f_n [Hz]$',fontsize=18)
 plt.ylabel('$\mathcal{F}\{\psi_{\mathcal{M}r}\} [db]$',fontsize=18)
 Tit_string='$\mathcal{F}\{\psi\}_{\mathcal{M}'+str(r)+'}(f_n)$'
 plt.title(Tit_string,fontsize=17)
 plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
 Name=FOLDER_mPOD_RESULTS+os.sep+'mPOD_Psi_hat_'+str(r)+'R.png'
 plt.savefig(Name, dpi=200) 
 plt.show()




















