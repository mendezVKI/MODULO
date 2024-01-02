# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:29:55 2019

This is the first version of the Ex 1- Same structure as in the Matlab exercise.
@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt
import os 
from Others import Plot_Field_Cylinder
from Others import Plot_Field_TEXT_Cylinder
from src.download_data import download_data
from src.load_data_methods import load_from_columns
from src.load_data_methods import load_from_columns_parallel

Anim=False # Decide if you want to construct animation of the data or not

# Extract all the Zip Files


# import urllib.request
# print('Downloading Data for Tutorial 2...')
# urllib.request.urlretrieve(url, 'Ex_5_TR_PIV_Cylinder.zip')
# print('Download Completed! I prepare data Folder')
# # Unzip the file
# from zipfile import ZipFile
# zf = ZipFile(String,'r'); zf.extractall('./'); zf.close()

folder ='Ex_5_TR_PIV_Cylinder'
url = 'https://osf.io/qa8d4/download'
download_data(url=url, destination=folder, force = False)

# Construct Time discretization
n_t=13200; Fs=3000; dt=1/Fs
n_t=2000; Fs=3000; dt=1/Fs
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 


# Read file number 10 (Check the string construction)
Name = folder + os.sep + 'Res%05d'%10+'.dat' # Check it out: print(Name)
print(Name)
Name_Mesh = folder+os.sep+'MESH.dat'
n_s, Xg, Yg, Vxg, Vyg, X_S,Y_S=Plot_Field_TEXT_Cylinder(Name,Name_Mesh) 
Name='Snapshot_Cylinder.png'
plt.savefig(Name, dpi=200) 


nxny=int(n_s/2) 
####################### 1. CONSTRUCT THE DATA MATRIX D #################
# Initialize the data matrix D
D=np.zeros([n_s,n_t])
#####################################################################

# D = load_from_columns(D, 'Res*',folder=folder, skip_lines=1, columns_in_order=[0,1], timesteps=n_t)
D = load_from_columns_parallel('Res*',folder=folder, skip_lines=1, columns_in_order=[0,1], timesteps=n_t)

V_X = D[4,:]
V_Y = D[int(n_s/2+4),:]
PROBE=np.sqrt(V_X**2+V_Y**2)

# Plot velocity in the top corner
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=13)
plt.rc('ytick',labelsize=13)
plt.plot(t,PROBE,'k',)
plt.xlabel('$t[s]$',fontsize=16)
plt.ylabel('$U_{\infty}[m/s]$',fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.title('Centerline Vel Evolution',fontsize=13)
plt.xlim([0,4.56])
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.01, pos1.y0 + 0.01,  pos1.width *0.95, pos1.height *0.95] 
ax.set_position(pos2) # set a new position
Name='U_Infty.pdf'
plt.savefig(Name, dpi=200) 

  
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
fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
# Put both components as fields in the grid
_,_,Vxg,Vyg,Magn=Plot_Field_Cylinder(X_S,Y_S,V_X_m,V_Y_m,True,2,None)
ax.set_aspect('equal') # Set equal aspect ratio
ax.set_xlabel('$x[mm]$',fontsize=16)
ax.set_ylabel('$y[mm]$',fontsize=16)
ax.set_xticks(np.arange(0,70,10))
ax.set_yticks(np.arange(-10,11,10))
ax.set_xlim([0,50])
ax.set_ylim(-10,10)
circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
plt.gcf().gca().add_artist(circle)

NameOUT='Mean_FLOW.png'
plt.savefig(NameOUT, dpi=100)   



from Others import Animation

if Anim:
 plt.ioff() # To disable interactive plotting   
## Visualize entire evolution (Optional)
 Animate=Animation('Data.npz','Exercise_4.gif')
else:
 print('No animation requested')    
    
    
    