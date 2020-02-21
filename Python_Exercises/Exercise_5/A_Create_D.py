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

Anim=False # Decide if you want to construct animation of the data or not

# Extract all the Zip Files
FOLDER='Ex_5_TR_PIV_Cylinder'


import urllib.request
print('Downloading Data for Tutorial 2...')
url = 'https://osf.io/qa8d4/download'
urllib.request.urlretrieve(url, 'Ex_5_TR_PIV_Cylinder.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_5_TR_PIV_Cylinder.zip'
zf = ZipFile(String,'r'); zf.extractall('./'); zf.close()



FOLDER='Ex_5_TR_PIV_Cylinder'
# Construct Time discretization
n_t=13200; Fs=3000; dt=1/Fs 
n_t=10000; Fs=3000; dt=1/Fs
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 


# Read file number 10 (Check the string construction)
Name=FOLDER+os.sep+'Res%05d'%10+'.dat' # Check it out: print(Name)
Name_Mesh=FOLDER+os.sep+'MESH.dat'
n_s, Xg, Yg, Vxg, Vyg, X_S,Y_S=Plot_Field_TEXT_Cylinder(Name,Name_Mesh) 
Name='Snapshot_Cylinder.png'
plt.savefig(Name, dpi=200) 


nxny=int(n_s/2) 
####################### 1. CONSTRUCT THE DATA MATRIX D #################
# Initialize the data matrix D
D=np.zeros([n_s,n_t])
#####################################################################
PROBE=np.zeros(n_t)


for k in range(0,n_t):
  Name=FOLDER+os.sep+'Res%05d'%(k+1)+'.dat' # Name of the file to read
  # Read data from a file
  DATA = np.genfromtxt(Name,usecols=np.arange(0,2),max_rows=nxny+1) # Here we have the two colums
  Dat=DATA[1:,:] # Here we remove the first raw, containing the header
  V_X=Dat[:,0]; # U component
  V_Y=Dat[:,1]; # V component
  D[:,k]=np.concatenate([V_X,V_Y],axis=0) # Reshape and assign
  # Obs: the file count starts from 1 but the index must start from 0
  print('Loading Step '+str(k+1)+'/'+str(n_t)) 
  PROBE[k]=np.sqrt(V_X[4]**2+V_Y[4]**2)
  

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
    
    
    