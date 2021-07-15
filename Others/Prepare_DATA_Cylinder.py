# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:14:18 2020

@author: mendez
"""


import numpy as np
import matplotlib.pyplot as plt
import os 
from Others import Plot_Field_TEXT_Cylinder

#%% Download the data from the repository

# Extract all the Zip Files
FOLDER='Ex_5_TR_PIV_Cylinder'

import urllib.request
print('Downloading Data for Tutorial 5...')
url = 'https://osf.io/47ftd/download'
urllib.request.urlretrieve(url, 'Ex_5_TR_PIV_Cylinder.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_5_TR_PIV_Cylinder.zip'
zf = ZipFile(String,'r'); zf.extractall('./Ex_5_TR_PIV_Cylinder'); zf.close()


#%% Input data for the decomposition and repartition
# Number of time steps and sampling frequency (in Hz)
n_t=13200; Fs=3000; dt=1/Fs 

Name=FOLDER+os.sep+'Res%05d'%10+'.dat' # Check it out: print(Name)
Name_Mesh=FOLDER+os.sep+'MESH.dat'
n_s, Xg, Yg, Vxg, Vyg, X_S,Y_S=Plot_Field_TEXT_Cylinder(Name,Name_Mesh) 
Name='Snapshot_Cylinder.png'
plt.savefig(Name, dpi=200) 
