# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:29:55 2019

This is the first version of the Ex 1- Same structure as in the Matlab exercise.
@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt
import os 
from Others import Plot_Field_TEXT
from Others import Plot_Field

Anim=False # Decide if you want to construct animation of the data or not

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


## Set Up and Create Data Matrix D
# 1 . Data Preparation
# For info, these data were collected with the following parameters
n_t=2000 # number of steps.
Fs=2000 # Sampling frequency
dt=1/Fs # This data was sampled at 2kHz.
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 

# Read file number 10 (Check the string construction)
Name=FOLDER+os.sep+'Res%05d'%10+'.dat' # Check it out: print(Name)
n_s, Xg, Yg, Vxg, Vyg, X_S,Y_S=Plot_Field_TEXT(Name) 
nxny=n_s/2 
####################### 1. CONSTRUCT THE DATA MATRIX D #################
# Initialize the data matrix D
D=np.zeros([n_s,n_t])
#####################################################################

for k in range(0,n_t):
  Name=FOLDER+os.sep+'Res%05d'%(k+1)+'.dat' # Name of the file to read
  # Read data from a file
  DATA = np.genfromtxt(Name) # Here we have the four colums
  Dat=DATA[1:,:] # Here we remove the first raw, containing the header
  V_X=Dat[:,2]; # U component
  V_Y=Dat[:,3]; # V component
  D[:,k]=np.concatenate([V_X,V_Y],axis=0) # Reshape and assign
  # Obs: the file count starts from 1 but the index must start from 0
  print('Loading Step '+str(k+1)+'/'+str(n_t)) 


# Save as numpy array all the data
np.savez('Data',D=D,t=t,dt=dt,n_t=n_t,\
         Xg=Xg,Yg=Yg,X_S=X_S,Y_S=Y_S)


# For a stationary test case like this, you might want to remove the mean  
D_MEAN=np.mean(D,1) # Temporal average (along the columns)
D_Mr=D-np.array([D_MEAN,]*n_t).transpose() # Mean Removed

# Check the mean flow
nxny=int(nxny)
V_X_m=D_MEAN[0:nxny]
V_Y_m=D_MEAN[nxny::]
fig, ax = plt.subplots(figsize=(8, 5)) # This creates the figure
# Put both components as fields in the grid
_,_,Vxg,Vyg,Magn=Plot_Field(X_S,Y_S,V_X_m,V_Y_m,True,2,None)
ax.set_aspect('equal') # Set equal aspect ratio
ax.set_xlabel('$x[mm]$',fontsize=16)
ax.set_ylabel('$y[mm]$',fontsize=16)
ax.set_title('Mean Velocity Field via TR-PIV',fontsize=18)
ax.set_xticks(np.arange(0,40,10))
ax.set_yticks(np.arange(10,30,5))
ax.set_xlim([0,35])
ax.set_ylim([10,29])
ax.invert_yaxis() # Invert Axis for plotting purpose
 
NameOUT='Mean_FLOW.png'
plt.savefig(NameOUT, dpi=100)   



from Others import Animation

if Anim:
 plt.ioff() # To disable interactive plotting   
## Visualize entire evolution (Optional)
 Animate=Animation('Data.npz','Exercise_4.gif')
else:
 print('No animation requested')    
    
    
    