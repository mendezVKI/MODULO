# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:11:03 2020
Expanded and cleaned on Tue Jan  2 17:09:08 2024

@author: mendez
"""

# This is a script to download the data for the all exercises in this repo.
#  Make sure you have installed request
#  
# The data is available at https://osf.io/5na4h/
# This collects the portion of script you should copy paste onto your file for downloading the data
# (thus forgive me for the multiple import of the same packages)

# NOTE: for each exercise, the data is stored in a folder (EX1,EX2, EX3,....)

# However, the numbering differes within the brunches. For example, the original 
# matlab version was build from exercises 1,2,3,4,5. Exercise 6 is not used for MODULO but as part of 
# a different work on nonlinear decompositions. Exercise 7 and 8 are used in the examples of MODULO v2.

# To handle the different numbering, in the examples from modulo the variable "FOLDER" is renamed accordingly.


#%% Exercise 1
# This exercise presents an example of the use of mPOD and POD on a 1D scalar dataset.
# This dataset is the velocity profile of a pulsating Poiseuille Flow. 
# More info on the test case is presented in Mendez & Buchlin, TN215, 10.13140/RG.2.2.32209.02404 '''
import os 
import urllib.request
FOLDER='Ex_1'
print('Downloading Data for Ex 1...')
url = 'https://osf.io/zqrp5/download'
urllib.request.urlretrieve(url, 'Ex_1_1D_Analytical.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_1_1D_Analytical.zip'
zf = ZipFile(String,'r')
zf.extractall('./')
zf.close()
os.rename('Ex_1_1D_Analytical', FOLDER) # rename the data flolder to FOLDER
os.remove(String) # Delete the zip file with the data 
print('Data set unzipped and ready ! ')



#%% Exercise 2
# This exercise presents an example of the use of mPOD and POD on a 2D scalar dataset.
# This dataset is constructed by adding three modes of similar energy content, to challenge the POD.
# Data from Section 4 of https://arxiv.org/abs/1804.09646

import os
import urllib.request
FOLDER='Ex_2'
print('Downloading Data for Ex 2...')
url = 'https://osf.io/jhfmn/download'
urllib.request.urlretrieve(url, 'Ex_2_Synthetic.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_2_Synthetic.zip'
zf = ZipFile(String,'r')
zf.extractall('./')
zf.close()
os.rename('Ex_2_Synthetic', FOLDER) # rename the data flolder to FOLDER
os.remove(String) # Delete the zip file with the data 
print('Data set unzipped and ready ! ')



#%% Exercise 3

# This exercise presents an example of the use of mPOD and POD on a 2D scalar dataset.
# This dataset is constructed from the CFD simulation of the 2D Navier-stokes problem in the vorticity-stream function
# formulation. Data from Section 5 of https://arxiv.org/abs/1804.09646

import os
import urllib.request
FOLDER='Ex_3'
print('Downloading Data for Ex 3...')
url = 'https://osf.io/zgujk/download'
urllib.request.urlretrieve(url, 'Ex_3_CFD_Vortices.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_3_CFD_Vortices.zip'
zf = ZipFile(String,'r')
zf.extractall(FOLDER)
zf.close()
os.remove(String) # Delete the zip file with the data 
print('Data set unzipped and ready ! ')


#%% Exercise 4

# This exercise presents an example of the use of mPOD and POD on a 2D vector dataset.
# This dataset is the TR-PIV of an impinging jet.
# More info on the test case is presented in Sec. 6 of https://arxiv.org/abs/1804.09646
# About the data from the Tr-PIV analysis: the sampling frequency is 2kHz.
# Each file contains the X and Y coordinates (in mm) in the first two columns;
#  the U and V components in m/s in the third and fourth.

import os
import urllib.request
FOLDER='Ex_4'
print('Downloading Data for Ex 4...')
url = 'https://osf.io/c28de/download'
urllib.request.urlretrieve(url, 'Ex_4_TR_PIV_Jet.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_4_TR_PIV_Jet.zip'
zf = ZipFile(String,'r')
zf.extractall('./')
zf.close() 
os.rename('Ex_4_TR_PIV_Jet', FOLDER) # rename the data flolder to FOLDER
os.remove(String) # Delete the zip file with the data 
print('Data set unzipped and ready ! ')



#%% Exercise 5

#This exercise presents an example of the use of mPOD and POD with memory saving options on a 2D vector dataset.
# This dataset is the TR-PIV of the flow past a cylinder in transient conditions.
# More info on the test case is presented in https://arxiv.org/abs/2001.01971
# About the data from the from the Tr-PIV analysis: the sampling frequency is 3kHz.
# MESH.dat contains the X and Y coordinates (in mm). Each file called Res*.dat containes the U and V components in m/s .

import os
import urllib.request
FOLDER='Ex_5'
print('Downloading Data for Ex 5...')
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

#%% Exercise 6

# This is a database of synthetic PIV images taken from https://doi.org/10.1016/j.expthermflusci.2016.08.021
# It contains two folders: X and X_P, both containing 400 images (200 PIV pairs).
# X contains the noisy images while X_P contains the ideal ones (noise free) linked to these.
# In the notation of the paper, X_B=X-X_P is the background one would ideally remove.

FOLDER='Ex_6'
# First we unzip the file 
import urllib.request
print('Downloading Data for Ex 6...')
url = 'https://osf.io/g7asz/download'
urllib.request.urlretrieve(url, 'Ex_6_PIV_Images_Denoising.zip ')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_6_PIV_Images_Denoising.zip'
zf = ZipFile(String,'r')
zf.extractall('./')
zf.close() 
os.rename('Ex_6_PIV_Images_Denoising', FOLDER) # rename the data flolder to FOLDER
os.remove(String) # Delete the zip file with the data 
print('Data set unzipped and ready ! ')



#%% Exercise 7
# This database collects the velocity field from a 2D simulations of the flow past a cylinder.
# The dataset is organized in 4 txt files. Some key information: the cylinder has a diameter of 15mm, in an overly
# large domain of 300 x 600 mm. The simulations were carried out in Openfoam then exported in a regulard grid.
# This was part of the STP of Denis Dumoulin (see VKI Library for more info).

# The inlet velocity is 10m/s, with a TI of 5%. The sampling frequency in the data is Fs=100 Hz.

# It is important to note that the dataset contains NANs in the location of the 
# cylinder. These can be safely ignored using numpy's nan_to_num, which 
# replaces the Nans with zeros.
import os 
import urllib.request
from zipfile import ZipFile

FOLDER='Ex_7'

url = 'https://osf.io/emgv2/download'
urllib.request.urlretrieve(url, 'Ex_7_2D_CFD.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
String='Ex_7_2D_CFD.zip'
zf = ZipFile(String,'r')
zf.extractall('./')
zf.close() 
os.rename('Ex_7_2D_Cylinder_CFD', FOLDER) # rename the data flolder to FOLDER
os.remove(String) # Delete the zip file with the data 
print('Data set unzipped and ready ! ')



#%% Exercise 8

# This database collects a portion of the domain in a 3D LES simulation of a turbulent impinging jet.
# The nozzle has a width of D=XX mm and is placed at a distance Z=XX  from the wall.
# The mean velocity is approximately U= XXX m/s, resulting in a Reynolds number Re= U D/nu = XXXX
# The dataset is sampled with dt = 5e-05 s.

# Note: To visualize this data (VTK format) it is strongly adviced to use 
# pyvista (# https://docs.pyvista.org/version/stable/examples/99-advanced/openfoam-tubes.html)



import os 
import urllib.request
from zipfile import ZipFile

FOLDER='Ex_8'

url = 'https://osf.io/aqkc8/download'
urllib.request.urlretrieve(url, 'Ex_8_2D_Impinging_JET_CFD.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
String='Ex_8_2D_Impinging_JET_CFD.zip'
zf = ZipFile(String,'r')
zf.extractall('./')
zf.close() 
os.rename('Ex_8_2D_Impinging_JET_CFD', FOLDER) # rename the data flolder to FOLDER
os.remove(String) # Delete the zip file with the data 
print('Data set unzipped and ready ! ')









