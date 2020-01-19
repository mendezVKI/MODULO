# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 14:01:17 2020

@author: mendez
"""


## D. Compute Projection.

# Given the temporal Basis for the POD and the mPOD we compute their
# spatial structures


import numpy as np
import matplotlib.pyplot as plt
from Others import Plot_Field
# Load Data
data = np.load('Data.npz')
D=data['D']; t=data['t']; dt=data['dt']; n_t=data['n_t'];
Fs=1/dt
Xg=data['Xg']; Yg=data['Yg']; X_S=data['X_S']; Y_S=data['Y_S']
n_y,n_x=Yg.shape; nxny=n_x*n_y; n_s=nxny*2

# Load mPOD basis
data=np.load('Psis_mPOD.npz')
Psi_M=data['PSI_M']
 
# Compute the spatial basis for mPOD 
# Projection from Eq. 2.6
R=Psi_M.shape[1];
PHI_M_SIGMA_M=np.dot(D,(Psi_M))
# Initialize the output
PHI_M=np.zeros((n_s,R))
SIGMA_M=np.zeros((R))

for i in range(0,R):     
  print('Completing mPOD Mode '+str(i))
  #Assign the norm as amplitude
  SIGMA_M[i] = np.linalg.norm(PHI_M_SIGMA_M[:,i])
  #Normalize the columns of C to get spatial modes
  PHI_M[:,i] = PHI_M_SIGMA_M[:,i]/SIGMA_M[i]
  
Indices=np.flipud(np.argsort(SIGMA_M)) # find indices for sorting in decreasing order
Sorted_Sigmas=SIGMA_M[Indices] # Sort all the sigmas
Phi_M=PHI_M[:,Indices] # Sorted Spatial Structures Matrix
Psi_M=Psi_M[:,Indices] # Sorted Temporal Structures Matrix
Sigma_M=Sorted_Sigmas # Sorted Amplitude Matrix
  
# Show some exemplary modes for mPOD. 

fig = plt.subplots(figsize=(14,8))
   # fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True,figsize=(10,8))

plt.rc('text', usetex=True)      # This is Miguel's customization
plt.rc('font', family='serif')
   
for j in range(1,4):
 ax=plt.subplot(3,2,2*j)
 Signal=Psi_M[:,j-1]-np.mean(Psi_M[:,j-1])
 Signal_FFT = np.fft.fft(Signal)/np.sqrt(n_t) # Compute the DFT
 Freqs=np.fft.fftfreq(len(Signal))*Fs # Compute the frequency bins
 plt.plot(Freqs,np.abs(Signal_FFT),linewidth=1.5)    
 plt.xlim([2,1050])
 plt.xscale('log')
 #ax.set_xticks([10, 50,150,300, 600])
 #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
 plt.xlabel('$f[Hz]$',fontsize=18)
 String_y='$\widehat{\psi}_{\mathcal{M}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 #plt.tight_layout(pad=1.1, w_pad=1, h_pad=1.0)
 plt.tight_layout()
 

for j in range(1,4):
 ax=plt.subplot(3,2,2*j-1)
 V_X=Phi_M[0:nxny,j-1]
 V_Y=Phi_M[nxny::,j-1]
 Plot_Field(X_S,Y_S,V_X,V_Y,True,2,0.8)
 ax.set_aspect('equal') # Set equal aspect ratio
# ax.set_xlabel('$x[mm]$',fontsize=16)
# ax.set_ylabel('$y[mm]$',fontsize=16)
#ax.set_title('Velocity Field via TR-PIV',fontsize=18)
 ax.set_xticks(np.arange(0,40,10))
 ax.set_yticks(np.arange(10,30,5))
 ax.set_xlim([0,35])
 ax.set_ylim([10,29])
 ax.invert_yaxis() # Invert Axis for plotting purpose
 String_y='$\phi_{\mathcal{M}'+str(j)+'}$'
 plt.title(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 
    
plt.savefig('Results_mPOD_1_3.pdf', dpi=100)      


fig = plt.subplots(figsize=(14,8))
   # fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True,figsize=(10,8))

plt.rc('text', usetex=True)      # This is Miguel's customization
plt.rc('font', family='serif')
   
for j in range(1,4):
 ax=plt.subplot(3,2,2*j)
 Signal=Psi_M[:,j-1+3]-np.mean(Psi_M[:,j-1])
 Signal_FFT = np.fft.fft(Signal)/np.sqrt(n_t) # Compute the DFT
 Freqs=np.fft.fftfreq(len(Signal))*Fs # Compute the frequency bins
 plt.plot(Freqs,np.abs(Signal_FFT),linewidth=1.5)    
 plt.xlim([2,1050])
 plt.xscale('log')
 #ax.set_xticks([10, 50,150,300, 600])
 #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
 plt.xlabel('$f[Hz]$',fontsize=18)
 String_y='$\widehat{\psi}_{\mathcal{M}'+str(j+3)+'}$'
 plt.ylabel(String_y,fontsize=18)
 #plt.tight_layout(pad=1.1, w_pad=1, h_pad=1.0)
 plt.tight_layout()
 

for j in range(1,4):
 ax=plt.subplot(3,2,2*j-1)
 V_X=Phi_M[0:nxny,j-1+3]
 V_Y=Phi_M[nxny::,j-1+3]
 Plot_Field(X_S,Y_S,V_X,V_Y,True,2,0.8)
 ax.set_aspect('equal') # Set equal aspect ratio
# ax.set_xlabel('$x[mm]$',fontsize=16)
# ax.set_ylabel('$y[mm]$',fontsize=16)
#ax.set_title('Velocity Field via TR-PIV',fontsize=18)
 ax.set_xticks(np.arange(0,40,10))
 ax.set_yticks(np.arange(10,30,5))
 ax.set_xlim([0,35])
 ax.set_ylim([10,29])
 ax.invert_yaxis() # Invert Axis for plotting purpose
 String_y='$\phi_{\mathcal{M}'+str(j+3)+'}$'
 plt.title(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 
    
plt.savefig('Results_mPOD_4_6.pdf', dpi=100)      


# Compute the spatial basis for the POD

# Load POD basis
data=np.load('Psis_POD.npz')
Psi_P=data['Psi_P']
Sigma_P=data['Sigma_P']

# Projection from Eq. 2.6
R=Psi_P.shape[1];
PHI_P_SIGMA_P=np.dot(D,(Psi_P))
# Initialize the output
Phi_P=np.zeros((n_s,R))

for i in range(0,R):     
  print('Completing POD Mode '+str(i))
  #Normalize the columns of C to get spatial modes
  Phi_P[:,i] = PHI_P_SIGMA_P[:,i]/Sigma_P[i]
  
 
fig = plt.subplots(figsize=(14,8))
   # fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True,figsize=(10,8))

plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
   
for j in range(1,4):
 ax=plt.subplot(3,2,2*j)
 Signal=Psi_P[:,j-1]-np.mean(Psi_P[:,j-1])
 Signal_FFT = np.fft.fft(Signal)/np.sqrt(n_t) # Compute the DFT
 Freqs=np.fft.fftfreq(len(Signal))*Fs # Compute the frequency bins
 plt.plot(Freqs,np.abs(Signal_FFT),linewidth=1.5)    
 plt.xlim([2,1050])
 plt.xscale('log')
 #ax.set_xticks([10, 50,150,300, 600])
 #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
 plt.xlabel('$f[Hz]$',fontsize=18)
 String_y='$\widehat{\psi}_{\mathcal{P}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 #plt.tight_layout(pad=1.1, w_pad=1, h_pad=1.0)
 plt.tight_layout()
 

for j in range(1,4):
 ax=plt.subplot(3,2,2*j-1)
 V_X=Phi_P[0:nxny,j-1]
 V_Y=Phi_P[nxny::,j-1]
 Plot_Field(X_S,Y_S,V_X,V_Y,True,2,0.8)
 ax.set_aspect('equal') # Set equal aspect ratio
# ax.set_xlabel('$x[mm]$',fontsize=16)
# ax.set_ylabel('$y[mm]$',fontsize=16)
#ax.set_title('Velocity Field via TR-PIV',fontsize=18)
 ax.set_xticks(np.arange(0,40,10))
 ax.set_yticks(np.arange(10,30,5))
 ax.set_xlim([0,35])
 ax.set_ylim([10,29])
 ax.invert_yaxis() # Invert Axis for plotting purpose
 String_y='$\phi_{\mathcal{P}'+str(j)+'}$'
 plt.title(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 
    
plt.savefig('Results_POD_1_3.pdf', dpi=100)      

 
fig = plt.subplots(figsize=(14,8))
   # fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True,figsize=(10,8))

plt.rc('text', usetex=True)      # This is Miguel's customization
plt.rc('font', family='serif')
   
for j in range(1,4):
 ax=plt.subplot(3,2,2*j)
 Signal=Psi_P[:,j-1+3]-np.mean(Psi_P[:,j-1])
 Signal_FFT = np.fft.fft(Signal)/np.sqrt(n_t) # Compute the DFT
 Freqs=np.fft.fftfreq(len(Signal))*Fs # Compute the frequency bins
 plt.plot(Freqs,np.abs(Signal_FFT),linewidth=1.5)    
 plt.xlim([2,1050])
 plt.xscale('log')
 #ax.set_xticks([10, 50,150,300, 600])
 #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
 plt.xlabel('$f[Hz]$',fontsize=18)
 String_y='$\widehat{\psi}_{\mathcal{P}'+str(j+3)+'}$'
 plt.ylabel(String_y,fontsize=18)
 #plt.tight_layout(pad=1.1, w_pad=1, h_pad=1.0)
 plt.tight_layout()
 

for j in range(1,4):
 ax=plt.subplot(3,2,2*j-1)
 V_X=Phi_P[0:nxny,j-1+3]
 V_Y=Phi_P[nxny::,j-1+3]
 Plot_Field(X_S,Y_S,V_X,V_Y,True,2,0.8)
 ax.set_aspect('equal') # Set equal aspect ratio
# ax.set_xlabel('$x[mm]$',fontsize=16)
# ax.set_ylabel('$y[mm]$',fontsize=16)
#ax.set_title('Velocity Field via TR-PIV',fontsize=18)
 ax.set_xticks(np.arange(0,40,10))
 ax.set_yticks(np.arange(10,30,5))
 ax.set_xlim([0,35])
 ax.set_ylim([10,29])
 ax.invert_yaxis() # Invert Axis for plotting purpose
 String_y='$\phi_{\mathcal{P}'+str(j+3)+'}$'
 plt.title(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 
    
plt.savefig('Results_POD_4_6.pdf', dpi=100)      

 

plt.close(fig='all')
# Amplitude Modes POD vs mPOD
RR=np.arange(1,Sigma_P.shape[0]+1)
## DFT Spectra and convergence
fig, ax = plt.subplots(figsize=(6,3))
plt.rc('text', usetex=True)     
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
sigma_P_v=(Sigma_P)/(n_s*n_t)*1000
sigma_M_v=(Sigma_M)/(n_s*n_t)*1000
plt.plot(RR,sigma_P_v,'ko',label='POD')
plt.plot(RR,sigma_M_v,'rs',label='mPOD')
plt.legend(fontsize=20,loc='upper right') 
plt.xlabel('$r$',fontsize=12)
plt.xlim(0.5,1001)
plt.xscale('log')
plt.ylabel('$\sigma_{\mathcal{P}r}/\sigma_{\mathcal{M}r}*1000$',fontsize=14)
plt.title('POD vs mPOD Amplitudes ',fontsize=16)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
#pos1 = ax.get_position() # get the original position 
#pos2 = [pos1.x0 + 0.01, pos1.y0 + 0.01,  pos1.width *0.95, pos1.height *0.95] 
#ax.set_position(pos2) # set a new position
plt.savefig('POD_mPOD_Amplitude.png', dpi=200)     
plt.close(fig)






