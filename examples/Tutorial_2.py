# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:06:08 2023

@author: Mendez
"""

#%% Tutorial 2

# We study the decomposition of a TR-PIV dataset of an impinging jet.

import numpy as np
import matplotlib.pyplot as plt 

# This is for plot customization
fontsize = 16
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['font.size'] = fontsize


#%% Download the data
 
# Extract all the Zip Files
FOLDER='Ex_4_TR_PIV_Jet'

# First we unzip the file 
import urllib.request
print('Downloading Data for Ex 4...')
url = 'https://osf.io/c28de/download'
urllib.request.urlretrieve(url, 'Ex_4_TR_PIV.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_4_TR_PIV.zip'
zf = ZipFile(String,'r')
zf.extractall('./')
zf.close()   
    
#%% Next we prepare the snapshot matrix

# We can use the modulo function ReadData to create the snapshot matrix
from modulo.utils.read_db import ReadData


FOLDER = "./Ex_4_TR_PIV_Jet"
# --- Component fields (N=2 for 2D velocity fields, N=1 for pressure fields)
N = 2 
# --- Number of mesh points
N_S = 6840
# --- Header (H) and footer (F) to be skipped during acquisition
H = 1; F = 0
# --- Read one sample snapshot (to get N_S)
Name = "./Ex_4_TR_PIV_jet/Res00001.dat"
Dat = np.genfromtxt(Name, skip_header=H, skip_footer=F)

D = ReadData._from_dat(folder='./Ex_4_TR_PIV_jet/', filename='Res%05d', 
                       N=2, N_S=2*Dat.shape[0],h = H, f = F, c=2)


#%% Initialize the MODULO object.
# We might now decide to remove the mean or not , or use the memory saving

from modulo.modulo import MODULO

# --- Initialize MODULO object
m = MODULO(data=D,
           MEMORY_SAVING=False,
           N_PARTITIONS=11,
           eig_solver= 'eigsh',   # eig solver for h matrices
           svd_solver= 'svd_sklearn_randomized', # this is the fastest approach
           )
# --- Check for D
# D = m._data_processing()

import time
import tracemalloc # To compute memory usage

# Compute the POD using snapshot's method
start_time = time.time() # Start the timing
tracemalloc.start() # Start the measure of memory usage
Phi_P1, Psi_P1, Sigma_P1 = m.compute_POD_K()
print("POD_K runs in %1.2f s ---" % (time.time() - start_time) )
current, _ = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"Memory usage in POD_K was {current / 10**6} MB ---")


# compute the POD using SVD methods    
start_time = time.time() # Start the timing
tracemalloc.start() # Start the measure of memory usage
Phi_P2, Psi_P2, Sigma_P2 = m.compute_POD_svd()
print("POD_SVDs runs in %1.2f s ---" % (time.time() - start_time) )
current, _ = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"Memory usage in POD_SVDs was {current / 10**6} MB ---")


#%% Compare the different results 
    
# Compare 2 with ref (their difference is below atol and rtol)
RTOL=1e-05 ; ATOL=1e-05

print('Comparison check for POD K vs POD_SVD')
print(np.allclose(np.abs(Phi_P1), np.abs(Phi_P2),rtol=1e-03,atol=1e-03))
print(np.allclose(np.abs(Psi_P1), np.abs(Psi_P2),rtol=1e-03,atol=1e-03))
print(np.allclose(np.abs(Sigma_P1), np.abs(Sigma_P2),rtol=1e-03,atol=1e-03))
    





