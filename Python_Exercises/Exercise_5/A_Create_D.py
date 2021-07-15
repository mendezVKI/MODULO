# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:29:55 2019

This is the first version of the Ex 5- Same structure as in the Matlab exercise.
The main feature of this exercise is that we will perform the decomposition
using the memory saving option. In other words, we split the matrix D
into 5 partitions and we proceed with the computation in blocks.

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt
import os 
from Others import Plot_Field_Cylinder
from Others import Plot_Field_TEXT_Cylinder

#%% Download the data from the repository
Anim=False # Decide if you want to construct animation of the data or not

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
FOLDER='Ex_5_TR_PIV_Cylinder'
# Construct Time discretization
n_t=13200; Fs=3000; dt=1/Fs 
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 

# Parameters for the memory saving
N_B=11# This is the number of blocks in the partition.
import math
n_t_B=math.floor(n_t/(N_B-1))
n_T_R= n_t-n_t_B*(N_B-1)# Reminder in the last partition
# This creates a new index (for the partitions):

if n_T_R==0:
# You will have N_B partitions with n_t_B entries each
 n_PAR=N_B
 Indices=np.zeros(n_PAR) 
 Indices[0:N_B]=np.arange(N_B)*n_t_B; 
else:
#You will have N_B partitions with n_t_B entries each + 1 partition with   
 n_PAR=N_B+1 
 Indices=np.zeros(n_PAR)
 Indices[0:N_B]=np.arange(N_B)*n_t_B; Indices[n_PAR-1]=n_t 
 
 
#%% Plot a sample snapshot
# Read file number 10 (Check the string construction)
Name=FOLDER+os.sep+'Res%05d'%10+'.dat' # Check it out: print(Name)
Name_Mesh=FOLDER+os.sep+'MESH.dat'
n_s, Xg, Yg, Vxg, Vyg, X_S,Y_S=Plot_Field_TEXT_Cylinder(Name,Name_Mesh) 
Name='Snapshot_Cylinder.png'
plt.savefig(Name, dpi=200) 


nxny=int(n_s/2) 
####################### 1. CONSTRUCT THE DATA MATRIX D #################
# Initialize the data matrix D

#####################################################################
PROBE=np.zeros(n_t)


#%% Run the loop to mount the matrices D1, D2, D3, etc..
for j in range(1,len(Indices)):
 R1=int(Indices[j-1]); R2=int(Indices[j]) # Range of indices
 D=np.zeros([n_s,R2-R1])
 for k in range(R1,R2):
  Name=FOLDER+os.sep+'Res%05d'%(k+1)+'.dat' # Name of the file to read
  # Read data from a file
  DATA = np.genfromtxt(Name,usecols=np.arange(0,2),max_rows=nxny+1) # Here we have the two colums
  Dat=DATA[1:,:] # Here we remove the first raw, containing the header
  V_X=Dat[:,0]; # U component
  V_Y=Dat[:,1]; # V component
  D[:,k-R1]=np.concatenate([V_X,V_Y],axis=0) # Reshape and assign
  # Obs: the file count starts from 1 but the index must start from 0
  print('Loading Step '+str(k+1)+'/'+str(n_t)) 
  PROBE[k]=np.sqrt(V_X[4]**2+V_Y[4]**2)
  # Save as numpy array the data for the block
 String='D'+str(j)
 if j==1:
    np.savez(String,D=D,t=t,dt=dt,n_t=n_t,\
          Xg=Xg,Yg=Yg,X_S=X_S,Y_S=Y_S,N_B=N_B)
 else:
    np.savez(String,D=D)  
 print('Storing Block '+str(j)+' of '+str(n_PAR-1)) 


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

# # This is not a stationary test case so it does not make sense to remove the mean
# # If you are looking for a code to remove the mean on the partitions,
# # Check out the code below  
# Anyway, we compute and show the mean ( the commented part is for substracting it)

D_SUM=np.mean(D,1)*0 # Initialize the mean vector
for j in range(1,len(Indices)):
  R1=int(Indices[j-1]); R2=int(Indices[j]) # Range of indices
  N_P=R2-R1  
  String='D'+str(j)+'.npz' 
  data = np.load(String)
  D=data['D']
  D_SUM=D_SUM+np.mean(D,1)*N_P # Summ everything
  print('Reloading for Mean n_B='+str(j)) 

D_MEAN=D_SUM/n_t # Temporal average (along the columns)
# Reload the entire set removing to it the mean
# for j in range(1,len(Indices)):
#   R1=int(Indices[j-1]); R2=int(Indices[j]) # Range of indices
#   N_P=R2-R1    
#   String='D'+str(j)+'.npz' 
#   data = np.load(String)
#   D=data['D']-np.array([D_MEAN,]*N_P).transpose() # Mean Removed
#   print('Re-saving Centered n_B='+str(j)) 
#   # In the first partition, we put all the info
#   String='D'+str(j)
#   if j==1:
#     np.savez(String,D=D,t=t,dt=dt,n_t=n_t,\
#           Xg=Xg,Yg=Yg,X_S=X_S,Y_S=Y_S,N_B=N_B)
#   else:
#     np.savez(String,D=D)   


#%% Plot the Mean Flow
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
 Animate=Animation('D1.npz','Exercise_5.gif')
else:
 print('No animation requested')    
    
    
    