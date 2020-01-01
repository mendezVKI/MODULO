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

# Load Data
data = np.load('Data.npz')
D=data['D']; t=data['t']; dt=data['dt']; n_t=data['n_t']
y=data['y']; dy=data['dy']; n_y=data['n_y']

# Load mPOD basis
data=np.load('Psis_mPOD.npz')
Psi_M=data['PSI_M']
 
# Compute the spatial basis for mPOD 
# Projection from Eq. 2.6
R=Psi_M.shape[1];
PHI_M_SIGMA_M=np.dot(D,(Psi_M))
# Initialize the output
PHI_M=np.zeros((n_y,R))
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
Sigma_M=np.diag(Sorted_Sigmas) # Sorted Amplitude Matrix
  
# Show some exemplary modes for mPOD. We take from 2 to 7.

fig = plt.subplots(figsize=(12,8))
   # fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True,figsize=(10,8))

plt.rc('text', usetex=True)      # This is Miguel's customization
plt.rc('font', family='serif')
for j in range(1,4):
 ax1=plt.subplot(3,2,2*j-1)
 plt.plot(y,Phi_M[:,j-1],linewidth=1.5)    
 plt.xlabel('$\hat{y}$',fontsize=18)
 String_y='$\phi_{\mathcal{M}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 plt.tight_layout()
    
for j in range(1,4):
 ax1=plt.subplot(3,2,2*j)
 plt.plot(t,Psi_M[:,j-1],linewidth=1.5)    
 plt.xlabel('$\hat{t}$',fontsize=18)
 String_y='$\psi_{\mathcal{M}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 plt.tight_layout()
    
plt.savefig('Results_mPOD.pdf', dpi=100)      

# Compute the spatial basis for the POD

# Load POD basis
data=np.load('Psis_POD.npz')
Psi_P=data['Psi_P']
Sigma_P=data['Sigma_P']

# Projection from Eq. 2.6
R=Psi_P.shape[1];
PHI_P_SIGMA_P=np.dot(D,(Psi_P))
# Initialize the output
Phi_P=np.zeros((n_y,R))

for i in range(0,R):     
  print('Completing POD Mode '+str(i))
  #Normalize the columns of C to get spatial modes
  Phi_P[:,i] = PHI_P_SIGMA_P[:,i]/Sigma_P[i]
  
Sigma_P=np.diag(Sorted_Sigmas) # Sorted Amplitude Matrix
 
fig = plt.subplots(figsize=(12,8))
   
for j in range(1,4):
 ax1=plt.subplot(3,2,2*j-1)
 plt.plot(y,Phi_P[:,j-1],linewidth=1.5)    
 plt.xlabel('$\hat{y}$',fontsize=18)
 String_y='$\phi_{\mathcal{P}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 plt.tight_layout()
    
for j in range(1,4):
 ax1=plt.subplot(3,2,2*j)
 plt.plot(t,Psi_P[:,j-1],linewidth=1.5)    
 plt.xlabel('$\hat{t}$',fontsize=18)
 String_y='$\psi_{\mathcal{P}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 plt.tight_layout()
    
plt.savefig('Results_POD.pdf', dpi=100)      
####### DFT ########
Fs=2 # Sampling frequency
## For comparison putposes we also perform here a DFT.
print('Compute also the DFT')
PSI_F=np.fft.fft(np.eye(n_t))/np.sqrt(n_t) # Prepare the Fourier Matrix.
print('Projecting Data')
#D_Complex=D+1j*np.zeros(D.shape)
PHI_SIGMA=np.dot(D,np.conj(PSI_F)) # This is PHI * SIGMA
PHI_F=np.zeros((D.shape[0],n_t),dtype=complex) # Initialize the PHI_F MATRIX
SIGMA_F=np.zeros(n_t) # Initialize the SIGMA_F MATRIX
# Now we proceed with the normalization. This is also intense so we time it
for r in range(0,n_t): # Loop over the PHI_SIGMA to normalize
  MEX='Proj '+str(r)+' /'+str(n_t) 
  print(MEX)
  SIGMA_F[r]=abs(np.vdot(PHI_SIGMA[:,r],PHI_SIGMA[:,r]))**0.5
  PHI_F[:,r]=PHI_SIGMA[:,r]/SIGMA_F[r]

n_t=int(n_t)
Freqs=np.fft.fftfreq(n_t)*Fs # Compute the frequency bins
sigmas_V=SIGMA_F/(n_y*n_t) # Carefull with the normalization

fig, ax = plt.subplots(figsize=(10,4)) # Create Signal Noisy and Clean
plt.plot(np.fft.fftshift(Freqs),np.fft.fftshift(sigmas_V)*1000)
plt.rc('text', usetex=True)      # This is Miguel's customization
#plt.rc('font', family='serif')
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.xlabel('$f[-] $',fontsize=20)
plt.ylabel('$\sigma_{\mathcal{F}}[-]$',fontsize=20)
# plt.title('Eigen_Function_Sol_N',fontsize=18)
plt.xlim([-0.5,0.5])
plt.tight_layout()
plt.show()
plt.savefig('DFT_Modal_Spectra.pdf', dpi=100)      
plt.close(fig)


Indices=np.flipud(np.argsort(sigmas_V)) # find indices for sorting in decreasing order
Sorted_Sigmas=sigmas_V[Indices] # Sort all the sigmas
Sorted_Freqs=Freqs[Indices] # Sort all the frequencies accordingly.
Phi_F=PHI_F[:,Indices] # Sorted Spatial Structures Matrix
Psi_F=PSI_F[:,Indices] # Sorted Temporal Structures Matrix
SIGMA=np.diag(Sorted_Sigmas) # Sorted Amplitude Matrix


# Export the first r modes
Fol_Out='DFT_MODES'
import os
if not os.path.exists(Fol_Out):
    os.mkdir(Fol_Out)
  
for r in range(0,20):
 print('Exporting Mode '+str(r))  
 Phi_plot=Phi_F[:,r]

 fig = plt.subplots(figsize=(10,10))
 # fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True,figsize=(10,8))
 ax1=plt.subplot(2,1,1,position=[0.18,0.55,0.7,0.35])
 plt.plot(y,np.real(Phi_plot),linewidth=1.5,label='Real')    
 plt.plot(y,np.imag(Phi_plot),linewidth=1.5,label='Imag')     
 plt.legend(loc='lower left',fontsize=18)
 plt.xlabel('$\hat{y}$',fontsize=18)
 String_y='$\phi_{\mathcal{F}'+str(r)+'}$'
 plt.ylabel(String_y,fontsize=18)
# plt.tight_layout()
 ax1.set_xlim([-1,1])
   
 ax2=plt.subplot(2,1,2,position=[0.18,0.1,0.7,0.33])
 plt.plot(Sorted_Freqs,Sorted_Sigmas,'ko')
 plt.plot(Sorted_Freqs[r],Sorted_Sigmas[r],'rs',Markersize=8)
 ax2.set_xlabel('$f[-]$',fontsize=16)
 ax2.set_ylabel('$\sigma_{\mathcal{F}}$',fontsize=16)
 TIT='$\sigma_{\mathcal{F}'+str(r)+'}$'
 ax2.set_title(TIT,fontsize=24)
 ax2.set_xlim([-0.5,0.5])

 Name=Fol_Out+ os.sep +'DFT_Mode_'+str(r)+'.png'
 plt.savefig(Name, dpi=100)      
 plt.close()





