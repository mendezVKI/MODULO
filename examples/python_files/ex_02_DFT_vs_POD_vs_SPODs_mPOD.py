# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:20:59 2023

@author: mendez
"""

#%% Exercise 2.

# We study the decomposition of a TR-PIV dataset of an impinging jet.


'''This second tutorial considers a dataset which is dynamically much richer than the previous. This time 3 POD modes cannot the essence of what is happening. This data is the TR-PIV of an impinging gas jet and was extensively analyzed in previous tutorials on MODULO.
It was also discussed in the [Mendez et al, 2018](
https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/multiscale-proper-orthogonal-decomposition-of-complex-fluid-flows/D078BD2873B1C30B6DD9016E30B62DA8 ) and Chapter 8 of the book [Mendez et al, 2022](https://www.cambridge.org/core/books/datadriven-fluid-mechanics/0327A1A43F7C67EE88BB13743FD9DC8D).

We refer you to those work for details on the experimental set up and the flow conditions.
'''

import numpy as np # we use this to manipulate data 
import matplotlib.pyplot as plt # this is for plotting
import os  # this is to create/rename/delete folders
from modulo.modulo import MODULO # this is to create modulo objects

# these are some utility functions 
from modulo.utils.others import Plot_Field_TEXT_JET, Plot_Field_JET # plotting
from modulo.utils.others import Animation_JET # for animations 
from modulo.utils.read_db import ReadData # to read the data


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
    
# Folder where we extract the data 
FOLDER='Tutorial_2_JET_PIV'

# First we unzip the file (note that this is the 7th exercise in the old enumeration)
import urllib.request
print('Downloading Data for Tutorial 2...')
url = 'https://osf.io/c28de/download'
urllib.request.urlretrieve(url, 'Ex_4_TR_PIV.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_4_TR_PIV.zip'
zf = ZipFile(String,'r')
zf.extractall('./')
zf.close() 
os.rename('Ex_4_TR_PIV_Jet', FOLDER) # rename the data flolder to FOLDER
os.remove(String) # Delete the zip file with the data 
print('Data set unzipped and ready ! ')

    
#%% Load one snapshot and plot it 

# Construct Time discretization
n_t=2000; Fs=2000; dt=1/Fs 
t=np.linspace(0,dt*(n_t-1),n_t) # prepare the time axis# 

# Read file number 10 (Check the string construction)
SNAP=10
Name=FOLDER+os.sep+'Res%05d'%SNAP+'.dat' # Check it out: print(Name)
n_s, Xg, Yg, Vxg, Vyg, X_S,Y_S=Plot_Field_TEXT_JET(Name) 
# Shape of the grid
n_y,n_x=np.shape(Xg)

# Plot the vector field
fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
Magn=np.sqrt(Vxg**2+Vyg**2); 
CL=plt.contourf(Xg,Yg,Magn,levels=np.arange(0,9,1))
STEPx=2; STEPy=2; 
plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
           Vxg[::STEPx,::STEPy],Vyg[::STEPx,::STEPy],color='k',scale=100) # Create a quiver (arrows) plot
    
plt.rc('text', usetex=True)      # This is Miguel's customization
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
fig.colorbar(CL,pad=0.05,fraction=0.025)
ax.set_aspect('equal') # Set equal aspect ratio
ax.set_xlabel('$x[mm]$',fontsize=13)
ax.set_ylabel('$y[mm]$',fontsize=13)
ax.set_title('Tutorial 2: Impinging Jet',fontsize=16)
ax.set_xticks(np.arange(0,40,10))
ax.set_yticks(np.arange(10,30,5))
ax.set_xlim([0,35])
ax.set_ylim(10,29)
ax.invert_yaxis() # Invert Axis for plotting purpose
plt.tight_layout()
Name=FOLDER+os.sep+'Snapshot_JET_'+str(SNAP)+'.png'
plt.savefig(Name, dpi=200) 
plt.show()


#%% Step 1: Prepare the snapshot matrix

# --- Component fields (N=2 for 2D velocity fields, N=1 for pressure fields)
N = 2 
# --- Number of mesh points
N_S = 6840
# --- Header (H) and footer (F) to be skipped during acquisition
H = 1; F = 0
# --- Read one sample snapshot (to get N_S)
Name = "./Tutorial_2_JET_PIV/Res00001.dat"
Dat = np.genfromtxt(Name, skip_header=H, skip_footer=F)

D = ReadData._data_processing(database=None,
                              FOLDER_OUT='./',
                              FOLDER_IN='./Tutorial_2_JET_PIV/', 
                              filename='Res%05d', 
                              h=H,f=F,c=2,
                              N=2, N_S=2*Dat.shape[0],N_T=n_t)


#%% we also make an animation of the full dataset
Name_GIF=FOLDER+os.sep+'Velocity_Field.gif'
Mex=Animation_JET(Name_GIF,D,X_S,Y_S,500,600,1)

#%% Step 2: Perform the DFT 

'''We perform the (modal) DFT (see previous modulo 
[tutorial](https://www.youtube.com/watch?v=8fhupzhAR_M)). 
We fist create the folder where to export the results, then run the 
decomposition and export the results'''

FOLDER_DFT_RESULTS=FOLDER+os.sep+'DFT_Results_Jet'
if not os.path.exists(FOLDER_DFT_RESULTS):
    os.mkdir(FOLDER_DFT_RESULTS)

# We perform the DFT first
# --- Remove the mean from this dataset (stationary flow )!
D,D_MEAN=ReadData._data_processing(D,MR=True)
# We create a matrix of mean flow:
D_MEAN_mat=np.array([D_MEAN, ] * n_t).transpose()    

# --- Initialize MODULO object
m = MODULO(data=D)
# Compute the DFT
Sorted_Freqs, Phi_F, Sorted_Sigmas = m.compute_DFT(Fs)

# Shape of the grid
nxny=m.D.shape[0]//2; 

#%% DFT Post Processing

# Check the spectra of the DFT
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(Sorted_Freqs,20*np.log10((Sorted_Sigmas/np.sqrt((nxny*n_t)))),'ko')
plt.xlim([0,1000])
plt.ylim([-45,2])
plt.xlabel('$ f [Hz]$',fontsize=18)
plt.ylabel('$\sigma_{\mathcal{F}r}/(n_s n_r)$ [dB]',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name=FOLDER_DFT_RESULTS+os.sep+'DFT_Spectra_Impinging_JET.png'
plt.savefig(Name, dpi=200) 

# Check the spectra of the DFT
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(Sorted_Sigmas[0:n_t-1]/np.max(Sorted_Sigmas),'ko:')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim([0,n_t-1])
plt.xlabel('$r$',fontsize=18)
plt.ylabel('$\sigma_{\mathcal{F}r}/(\sigma_{\mathcal{F}1})$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name=FOLDER_DFT_RESULTS+os.sep+'DFT_R_Impinging_JET.png'
plt.savefig(Name, dpi=200) 

#% The modes come in pair. We plot the spatial structures of some modes

# Export the first r modes
Fol_Out=FOLDER_DFT_RESULTS+os.sep+'DFT_Results'
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)
  

for r in range(200,220):
  print('Exporting (real part of) Mode '+str(r))  
  Phi=np.real(Phi_F[:,r])

  # Check the mean flow
  V_X_m=Phi[0:nxny]
  V_Y_m=Phi[nxny::]
   # Put both components as fields in the grid
  Mod=np.sqrt(V_X_m**2+V_Y_m**2)
  Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
  Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
  Magn=np.transpose(Mod.reshape((n_x,n_y)))

  # Show The Mean Flow

  fig = plt.subplots(figsize=(8,8))
   # fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True,figsize=(10,8))
  ax1=plt.subplot(2,1,1,position=[0.18,0.55,0.7,0.35])
  plt.contourf(Xg,Yg,Magn)
  # One possibility is to use quiver
  STEPx=2; STEPy=2
  plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
            Vxg[::STEPx,::STEPy]*0.2,\
                -Vyg[::STEPx,::STEPy]*0.2,color='k') # Create a quiver (arrows) plot
  # plt.streamplot(Xg,Yg,Vxg,Vyg)
  ax1.set_aspect('equal') # Set equal aspect ratio
  
  ax1.set_xlabel('$x[mm]$',fontsize=14)
  ax1.set_ylabel('$y[mm]$',fontsize=14)
  TIT='$\phi_{'+str(r)+'}(x,y)$'
  ax1.set_title(TIT,fontsize=12)
  ax1.set_xticks(np.arange(0,40,10))
  ax1.set_yticks(np.arange(10,30,5))
  ax1.set_xlim([0,35])
  ax1.set_ylim(10,29)
  ax1.invert_yaxis() # Invert Axis for plotting purpose
     
  ax2=plt.subplot(2,1,2,position=[0.18,0.1,0.7,0.33])
  plt.plot(Sorted_Freqs,20*np.log10((Sorted_Sigmas/np.sqrt((nxny*n_t)))),'ko')
  plt.plot(Sorted_Freqs[r],20*np.log10((Sorted_Sigmas[r]/np.sqrt((nxny*n_t)))),'rs')
  
  ax2.set_xlabel('$f[Hz]$',fontsize=14)
  ax2.set_ylabel('$\sigma_{\mathcal{F}}$',fontsize=14)
  # TIT='$\sigma_{'+str(r)+'}$'
  # ax2.set_title(TIT,fontsize=12)
  ax2.set_xlim([-1000,1000])
  ax2.set_ylim([-45,10])

  Name=Fol_Out+ os.sep +'DFT_Mode_'+str(r)+'.png'
  plt.savefig(Name, dpi=200)      
  plt.close()


#%% DFT Conclusions

# Every DFT mode has one frequency. The convergence of the decomposition is 
# poor. We can plot an animation using the first 50 DFT modes. 
# First, construct an approximation of D in the DFT spectra:
    
#%% Build an animation of the Data and the approximation D_f
plt.ioff()
Name_GIF=FOLDER_DFT_RESULTS+os.sep+'Full_data.gif'
Mex=Animation_JET(Name_GIF,m.D,X_S,Y_S,500,600,1)

# We can build an approximation as follows
R=50
# Re-build the Temporal Basis from the frequencies
Psi_F_t=np.zeros((n_t,R),dtype=complex)
for r in range(R):
    Psi_F_t[:,r]=1/np.sqrt(n_t)*np.exp(2*np.pi*1j*Sorted_Freqs[r]*t)
Phi_F_t=Phi_F[:,0:R]
Sigma_F_t=Sorted_Sigmas[0:R]
# Summation over the selected leading modes
D_F=np.real(np.linalg.multi_dot([Phi_F_t,np.diag(Sigma_F_t),Psi_F_t.T]) )

Error=np.linalg.norm(m.D-D_F)/np.linalg.norm(m.D)

print('Convergence Error: E_C='+"{:.2f}".format(Error*100)+' %')

Name_GIF=FOLDER_DFT_RESULTS+os.sep+'DFT_Approximation_R50.gif'

Mex=Animation_JET(Name_GIF,D_F+D_MEAN_mat,X_S,Y_S,500,600,1)

#%% Perfom a POD Analysis

# The POD provides the best decomposition convergence.
# Here is how to perform it:
    
# --- Initialize MODULO object
m2 = MODULO(data=m.D,n_Modes=50)
# --- Check for D
D = m2._data_processing()

Phi_P, Psi_P, Sigma_P = m2.compute_POD_svd() # POD via svd


FOLDER_POD_RESULTS=FOLDER+os.sep+'POD_Results_Jet'
if not os.path.exists(FOLDER_POD_RESULTS):
    os.mkdir(FOLDER_POD_RESULTS)

# Plot the decomposition convergence
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(Sigma_P/np.max(Sigma_P),'ko:')
# ax.set_yscale('log'); ax.set_xscale('log')
plt.xlabel('$r$',fontsize=18)
plt.ylabel('$\sigma_{\mathcal{P}r}/(\sigma_{\mathcal{P}1})$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name=FOLDER_POD_RESULTS+os.sep+'POD_R_Impinging_JET.png'
plt.savefig(Name, dpi=200) 

# Plot the leading POD modes and their spectra:
    
# Show modes
for j in range(1,10):
 plt.close(fig='all') 
 fig, ax3= plt.subplots(figsize=(5,6))   
 ax=plt.subplot(2,1,1)
 plt.rc('text', usetex=True)    
 plt.rc('font', family='serif')
 plt.rc('xtick',labelsize=12)
 plt.rc('ytick',labelsize=12)
 V_X=Phi_P[0:nxny,j-1]
 V_Y=Phi_P[nxny::,j-1]
 Plot_Field_JET(X_S,Y_S,V_X,V_Y,True,2,1)
 #plt.quiver(X_S,Y_S,V_X,V_Y)
 ax.set_aspect('equal') # Set equal aspect ratio
 ax.set_xticks(np.arange(0,40,10))
 ax.set_yticks(np.arange(10,30,5))
 ax.set_xlim([0,35])
 ax.set_ylim([10,29])
 ax.set_xlabel('$x[mm]$',fontsize=13)
 ax.set_ylabel('$y[mm]$',fontsize=13)
 ax.invert_yaxis() # Invert Axis for plotting purpose
 String_y='$\phi_{\mathcal{S}'+str(j)+'}$'
 plt.title(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 
 ax=plt.subplot(2,1,2)
 Signal=Psi_P[:,j-1]
 s_h=np.abs((np.fft.fft(Signal-Signal.mean())))
 Freqs=np.fft.fftfreq(int(n_t))*Fs
 plt.plot(Freqs*(4/1000)/6.5,s_h,'-',linewidth=1.5)
 plt.xlim(0,0.38)    
 plt.xlabel('$St[-]$',fontsize=18)
 String_y='$\widehat{\psi}_{\mathcal{S}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 Name=FOLDER_POD_RESULTS+os.sep+'POD_s_Mode_'+str(j)+'.png'
 print(Name+' Saved')
 plt.savefig(Name, dpi=300) 


#%% Conclusion from POD analysis

# Here is the approximation with the leading 10 POD modes
D_P=np.real(np.linalg.multi_dot([Phi_P,np.diag(Sigma_P),Psi_P.T]) )
Error=np.linalg.norm(m.D-D_P)/np.linalg.norm(m.D)

print('Convergence Error: E_C='+"{:.2f}".format(Error*100)+' %')

Name_GIF=FOLDER_POD_RESULTS+os.sep+'POD_Approximation_R50.gif'
Mex=Animation_JET(Name_GIF,D_P+D_MEAN_mat,X_S,Y_S,500,600,1)

# The extreme convergence comes at the prices of spectra mixing. 
# Modes are characterized by a large range of frequencies and thus their spatial
# structures cannot be associated to specific ranges of frequencies.


#%% Perform Towne's SPOD with MODULO

# We perform Towne's SPOD (see tutorial). This is an hybrid between Pwelch's method
# and the POD. In short, we compute the DFT in various chunks of the data.
# for each of these we then perform a POD frequency by frequency.

FOLDER_sPOD_RESULTS=FOLDER+os.sep+'sPOD_t_Results_Jet'
if not os.path.exists(FOLDER_sPOD_RESULTS):
    os.mkdir(FOLDER_sPOD_RESULTS)

# Initialize a 'MODULO Object'
m3 = MODULO(data=m.D,MEMORY_SAVING=False)
# Prepare (partition) the dataset
# Compute the POD
Phi_SP, Sigma_SP, Freqs_Pos = m3.compute_SPOD_t(F_S=2000, # sampling frequency
                                                L_B=200, # Length of the chunks for time average
                                                O_B=150, # Overlap between chunks
                                                n_Modes=3) # number of modes PER FREQUENCY

# Plot the SPOD Spectra:
f_hat=Freqs_Pos*(4/1000)/6.5

fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(f_hat, (Sigma_SP[0,:]/Sigma_SP[0,0]),'ko:')
plt.xlim([0,0.5])
plt.xlabel('$St= f H/U [-]$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name=FOLDER_sPOD_RESULTS+os.sep+'SPOD_t_Jet.png'
plt.savefig(Name, dpi=200) 

# We here plot the structures of some of the modes

Ind_1=5 # Mode 5

fig, ax3= plt.subplots(figsize=(5,3))   
V_X=Phi_SP[0:nxny,0,Ind_1]
V_Y=Phi_SP[nxny::,0,Ind_1]
Plot_Field_JET(X_S,Y_S,V_X,V_Y,True,2,1)
  #plt.quiver(X_S,Y_S,V_X,V_Y)
ax3.set_aspect('equal') # Set equal aspect ratio
ax3.set_xticks(np.arange(0,40,10))
ax3.set_yticks(np.arange(10,30,5))
ax3.set_xlim([0,35])
ax3.set_ylim([10,29])
ax3.set_xlabel('$x[mm]$',fontsize=8)
ax3.set_ylabel('$y[mm]$',fontsize=8)
ax3.invert_yaxis() # Invert Axis for plotting purpose
String_y='$\phi_{\mathcal{S}'+str(Ind_1)+'}$'
plt.title(String_y,fontsize=18)
plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
Name=FOLDER_sPOD_RESULTS+os.sep+'SPOD_t_Jet_'+str(Ind_1)+'.png'
plt.savefig(Name, dpi=200) 


Ind_2=25 # Mode 25

fig, ax3= plt.subplots(figsize=(5,3))   
V_X=Phi_SP[0:nxny,0,Ind_2]
V_Y=Phi_SP[nxny::,0,Ind_2]
Plot_Field_JET(X_S,Y_S,V_X,V_Y,True,2,1)
  #plt.quiver(X_S,Y_S,V_X,V_Y)
ax3.set_aspect('equal') # Set equal aspect ratio
ax3.set_xticks(np.arange(0,40,10))
ax3.set_yticks(np.arange(10,30,5))
ax3.set_xlim([0,35])
ax3.set_ylim([10,29])
ax3.set_xlabel('$x[mm]$',fontsize=8)
ax3.set_ylabel('$y[mm]$',fontsize=8)
ax3.invert_yaxis() # Invert Axis for plotting purpose
String_y='$\phi_{\mathcal{S}'+str(Ind_2)+'}$'
plt.title(String_y,fontsize=18)
plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
Name=FOLDER_sPOD_RESULTS+os.sep+'SPOD_t_Jet_'+str(Ind_2)+'.png'
plt.savefig(Name, dpi=200) 
    



Ind_3=48 # Mode 48


fig, ax3= plt.subplots(figsize=(5,3))   
V_X=Phi_SP[0:nxny,0,Ind_3]
V_Y=Phi_SP[nxny::,0,Ind_3]
Plot_Field_JET(X_S,Y_S,V_X,V_Y,True,2,2)
  #plt.quiver(X_S,Y_S,V_X,V_Y)
ax3.set_aspect('equal') # Set equal aspect ratio
ax3.set_xticks(np.arange(0,40,10))
ax3.set_yticks(np.arange(10,30,5))
ax3.set_xlim([0,35])
ax3.set_ylim([10,29])
ax3.set_xlabel('$x[mm]$',fontsize=8)
ax3.set_ylabel('$y[mm]$',fontsize=8)
ax3.invert_yaxis() # Invert Axis for plotting purpose
String_y='$\phi_{\mathcal{S}'+str(Ind_3)+'}$'
plt.title(String_y,fontsize=18)
plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
Name=FOLDER_sPOD_RESULTS+os.sep+'SPOD_t_Jet_'+str(Ind_3)+'.png'
plt.savefig(Name, dpi=200) 
    
#%% Conclusion about Towne's SPOD 

# We have a smoother spectra and slightly smooter spatial structures
# thanks to the averaging procedure. We can't say much in terms of convergence
# because it is not trivial to rebuild the flow: we would need many modes.


#%% Sieber's SPOD

# This is an hybrid between DFT and POD. We filter the correlation matrix
# to make it more circulant. At the limit of perfectly circulant matrix,
 # the POD is a DFT.The filtering is carried out along the diagonals


FOLDER_SPOD_RESULTS=FOLDER+os.sep+'SPOD_s_Results_Jet'
if not os.path.exists(FOLDER_SPOD_RESULTS):
    os.mkdir(FOLDER_SPOD_RESULTS)


# Initialize a 'MODULO Object'
m = MODULO(data=m.D)
# Prepare (partition) the dataset
# Compute the POD
Phi_S, Psi_S, Sigma_S = m.compute_SPOD_s(Fs,N_O=100,
                                               f_c=0.01,
                                               n_Modes=25,
                                               SAVE_SPOD=True)
 
# The rest of the plotting is IDENTICAL to the POD part

fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(Sigma_S/np.max(Sigma_S),'ko:')
# ax.set_yscale('log'); ax.set_xscale('log')
plt.xlabel('$r$',fontsize=18)
plt.ylabel('$\sigma_{\mathcal{S}r}/(\sigma_{\mathcal{S}1})$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name=FOLDER_SPOD_RESULTS+os.sep+'SPOD_R_Impinging_JET.png'
plt.savefig(Name, dpi=200) 

# Plot the leading SPOD modes and their spectra:
    
# Show modes
for j in range(1,10):
 plt.close(fig='all') 
 fig, ax3= plt.subplots(figsize=(5,6))   
 ax=plt.subplot(2,1,1)
 plt.rc('text', usetex=True)    
 plt.rc('font', family='serif')
 plt.rc('xtick',labelsize=12)
 plt.rc('ytick',labelsize=12)
 V_X=Phi_S[0:nxny,j-1]
 V_Y=Phi_S[nxny::,j-1]
 Plot_Field_JET(X_S,Y_S,V_X,V_Y,True,2,1)
 #plt.quiver(X_S,Y_S,V_X,V_Y)
 ax.set_aspect('equal') # Set equal aspect ratio
 ax.set_xticks(np.arange(0,40,10))
 ax.set_yticks(np.arange(10,30,5))
 ax.set_xlim([0,35])
 ax.set_ylim([10,29])
 ax.set_xlabel('$x[mm]$',fontsize=13)
 ax.set_ylabel('$y[mm]$',fontsize=13)
 ax.invert_yaxis() # Invert Axis for plotting purpose
 String_y='$\phi_{\mathcal{S}'+str(j)+'}$'
 plt.title(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 
 ax=plt.subplot(2,1,2)
 Signal=Psi_S[:,j-1]
 s_h=np.abs((np.fft.fft(Signal-Signal.mean())))
 Freqs=np.fft.fftfreq(int(n_t))*Fs
 plt.plot(Freqs*(4/1000)/6.5,s_h,'-',linewidth=1.5)
 plt.xlim(0,0.38)    
 plt.xlabel('$St[-]$',fontsize=18)
 String_y='$\widehat{\psi}_{\mathcal{S}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 Name=FOLDER_SPOD_RESULTS+os.sep+'SPOD_s_Mode_'+str(j)+'.png'
 print(Name+' Saved')
 plt.savefig(Name, dpi=300) 


#%% Conclusion from SPOD analysis

# Here is the approximation with the leading 10 SPOD modes
D_P=np.real(np.linalg.multi_dot([Phi_S,np.diag(Sigma_S),Psi_S.T]) )
Error=np.linalg.norm(m.D-D_P)/np.linalg.norm(m.D)

print('Convergence Error: E_C='+"{:.2f}".format(Error*100)+' %')

Name_GIF=FOLDER_SPOD_RESULTS+os.sep+'sPOD_Approximation.gif'
Mex=Animation_JET(Name_GIF,D_P+D_MEAN_mat,X_S,Y_S,500,600,1)


#%% Mendez's mPOD

# Here we go with our legacy ;P

FOLDER_MPOD_RESULTS=FOLDER+os.sep+'MPOD_Results_Jet'
if not os.path.exists(FOLDER_MPOD_RESULTS):
    os.mkdir(FOLDER_MPOD_RESULTS)

# We here perform the mPOD as done in the previous tutorials.
# This is mostly a copy paste from those, but we include it for completenetss

Keep = np.array([1, 1, 1, 1])
Nf = np.array([201, 201, 201, 201])
# --- Test Case Data:
# + Stand off distance nozzle to plate
H = 4 / 1000  
# + Mean velocity of the jet at the outlet
U0 = 6.5  
# + Input frequency splitting vector in dimensionless form (Strohual Number)
ST_V = np.array([0.1, 0.2, 0.25, 0.4])  
# + Frequency Splitting Vector in Hz
F_V = ST_V * U0 / H
# + Size of the extension for the BC (Check Docs)
Ex = 203  # This must be at least as Nf.
dt = 1/2000
boundaries = 'reflective'
MODE = 'reduced'
#K = np.load("./MODULO_tmp/correlation_matrix/k_matrix.npz")['K']
Phi_M, Psi_M, Sigmas_M = m.compute_mPOD(Nf, Ex, F_V, Keep, boundaries, MODE, dt, False)


# The rest of the plotting is IDENTICAL to the POD part

fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(Sigmas_M/np.max(Sigmas_M),'ko:')
# ax.set_yscale('log'); ax.set_xscale('log')
plt.xlabel('$r$',fontsize=18)
plt.ylabel('$\sigma_{\mathcal{M}r}/(\sigma_{\mathcal{M}1})$',fontsize=18)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
Name=FOLDER_MPOD_RESULTS+os.sep+'mPOD_R_Impinging_JET.png'
plt.savefig(Name, dpi=200) 

# Plot the leading SPOD modes and their spectra:
    
# Show modes
for j in range(1,10):
 plt.close(fig='all') 
 fig, ax3= plt.subplots(figsize=(5,6))   
 ax=plt.subplot(2,1,1)
 plt.rc('text', usetex=True)    
 plt.rc('font', family='serif')
 plt.rc('xtick',labelsize=12)
 plt.rc('ytick',labelsize=12)
 V_X=Phi_M[0:nxny,j-1]
 V_Y=Phi_M[nxny::,j-1]
 Plot_Field_JET(X_S,Y_S,V_X,V_Y,True,2,1)
 #plt.quiver(X_S,Y_S,V_X,V_Y)
 ax.set_aspect('equal') # Set equal aspect ratio
 ax.set_xticks(np.arange(0,40,10))
 ax.set_yticks(np.arange(10,30,5))
 ax.set_xlim([0,35])
 ax.set_ylim([10,29])
 ax.set_xlabel('$x[mm]$',fontsize=13)
 ax.set_ylabel('$y[mm]$',fontsize=13)
 ax.invert_yaxis() # Invert Axis for plotting purpose
 String_y='$\phi_{\mathcal{M}'+str(j)+'}$'
 plt.title(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 
 ax=plt.subplot(2,1,2)
 Signal=Psi_M[:,j-1]
 s_h=np.abs((np.fft.fft(Signal-Signal.mean())))
 Freqs=np.fft.fftfreq(int(n_t))*Fs
 plt.plot(Freqs*(4/1000)/6.5,s_h,'-',linewidth=1.5)
 plt.xlim(0,0.38)    
 plt.xlabel('$St[-]$',fontsize=18)
 String_y='$\widehat{\psi}_{\mathcal{M}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 Name=FOLDER_MPOD_RESULTS+os.sep+'MPOD_s_Mode_'+str(j)+'.png'
 print(Name+' Saved')
 plt.savefig(Name, dpi=300) 


#%% Conclusion from mPOD analysis

# Here is the approximation with the leading 10 mPOD modes
D_P=np.real(np.linalg.multi_dot([Phi_M,np.diag(Sigmas_M),Psi_M.T]) )
Error=np.linalg.norm(m.D-D_P)/np.linalg.norm(m.D)

print('Convergence Error: E_C='+"{:.2f}".format(Error*100)+' %')

Name_GIF=FOLDER_MPOD_RESULTS+os.sep+'mPOD_Approximation.gif'
Mex=Animation_JET(Name_GIF,D_P+D_MEAN_mat,X_S,Y_S,500,600,1)

# The approximation is not bad! 

































