# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 01:11:36 2019

@author: mendez
"""
## C. Compute Temporal Basis

# This script computes the temporal basis via standard Proper
# Orthogonal Decomposition and via Multiscale Proper Orthogonal
# Decomposition.

## mPOD Decomposition

import numpy as np
import matplotlib.pyplot as plt
from mPOD_Functions import mPOD_K


# Load all the dataset and the correlation matrix
data = np.load('Data.npz')
D=data['D']; t=data['t']; dt=data['dt']; n_t=data['n_t'];
Xg=data['Xg']; Yg=data['Yg']
n_y,n_x=Yg.shape; nxny=n_x*n_y
# Load the correlation matrix


## Study the frequency content of K
## We plot || K_hat || to look for frequencies
# This could be done via Matrix Multiplication (as it is done in the paper)
# but we use fft2 for fast computation. If you want to see the algebraic form of the code,
# do not hesitate to contact me at mendez@vki.ac.be

data=np.load('Correlation_K.npz')
K=data['K']

plt.ioff() # To enable interactive plotting
Fs=1/dt; # Sampling frequency
Freq=np.fft.fftshift(np.fft.fftfreq(int(n_t)))*Fs # Frequency Axis
# Compute the 2D FFT of K
K_HAT_ABS=np.fliplr(np.abs(np.fft.fftshift(np.fft.fft2(data['K']-np.mean(data['K'])))));

fig, ax = plt.subplots(figsize=(4,4))
#ax.set_xlim([-0.5,0.5])
#ax.set_ylim([-0.5,0.5])
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.pcolor(Freq,Freq,K_HAT_ABS/np.size(D)) # We normalize the result
ax.set_aspect('equal') # Set equal aspect ratio
ax.set_xlabel('$\hat{f}[Hz]$',fontsize=14)
ax.set_ylabel('$\hat{f}[Hz]$',fontsize=14)
ax.set_xticks(np.arange(-600,610,200))
ax.set_yticks(np.arange(-600,610,200))
ax.set_xlim(-600,600)
ax.set_ylim(-600,600)
plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
plt.clim(0,0.5) # From here  downward is just for plotting purposes
plt.tight_layout()
plt.savefig('Correlation_Spectra.png', dpi=100)  


# This matrix shows that there are two dominant frequencies in the problem.
# We set a frequency splitting to divide these two portions of the
# spectra. (See Sec. 3.1-3.2 of the article)

H=4/1000; # Stand off distance nozzle to plate
U0=6.5; # Mean velocity of the jet at the outlet


ST_V=np.array([0.1,0.2,0.25,0.4]) # Define scales based on St
F_V=ST_V*U0/H

Keep=np.array([1,1,1,1]); #These are the band-pass you want to keep (Approximation is always kept).
Nf=np.array([201,201,201,201]); # This vector collects the length of the filter kernels.

Ex=203 # This must be at least as Nf.
#These are the number of points that will be padded on the two sides of the signal
#following the specified BC.
##

# We can visualize where these are acting
F_Bank_r = F_V*2/Fs; #(Fs/2 mapped to 1)
M=len(F_Bank_r); # Number of scales
# Plot the transfer functions along the diagonal of K (similar to Fig 1)


fig, ax = plt.subplots(figsize=(8,5))
plt.rc('text', usetex=True)      # This is Miguel's customization
#plt.rc('font', family='serif')
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

# Extract the diagonal of K_F
K_HAT_ABS=np.fliplr(np.abs(np.fft.fftshift(np.fft.fft2(K-np.mean(K)))));
# For Plotting purposes remove the 0 freqs.
ZERO_F=np.where(Freq==0)[0]
ZERO_F=ZERO_F[0]; diag_K=np.abs(np.diag((K_HAT_ABS)));
diag_K[ZERO_F-1:ZERO_F+1]=0;
plt.plot(Freq,diag_K/np.max(diag_K),linewidth=1.2);


from scipy.signal import firwin # To create FIR kernels
# Loop over the scales to show the transfer functions
for m in range(0,len(Keep)):
  # Generate the 1d filter for this 
    if m==0:
       #Low Pass Filter
       h_A=firwin(Nf[m], F_Bank_r[m], window='hamming')
       # First Band Pass Filter
       h1d_H= firwin(Nf[m],[F_Bank_r[m],F_Bank_r[m+1]],pass_zero=False) # Band-pass
       # Assign the first band-pass
       plt.plot(Freq,np.fft.fftshift(np.abs(np.fft.fft(h_A,n_t))),linewidth=1.5)
       plt.plot(Freq,np.fft.fftshift(np.abs(np.fft.fft(h1d_H,n_t))),linewidth=1.5)
       List_h=[h_A]
       List_h.append(h1d_H)
    elif m>0 and m<M-1:
      if Keep[m]==1:
       # This is the 1d Kernel for Band pass
       h1d_H= firwin(Nf[m],[F_Bank_r[m],F_Bank_r[m+1]],pass_zero=False) # Band-pass
       plt.plot(Freq,np.fft.fftshift(np.abs(np.fft.fft(h1d_H,n_t))),linewidth=1.5)
       List_h.append(h1d_H)
    else:
      if Keep[m]==1:
       # This is the 1d Kernel for High Pass (last scale)
       h1d_H = firwin(Nf[m],F_Bank_r[m],pass_zero=False)
       List_h.append(h1d_H)
       plt.plot(Freq,np.fft.fftshift(np.abs(np.fft.fft(h1d_H,n_t))),linewidth=1.5)
       
plt.xlim([0,600])    
plt.xlabel('$f[Hz]$',fontsize=18)
plt.ylabel('Normalized Spectra',fontsize=18)


plt.tight_layout()
plt.savefig('Frequency_Splitting.png', dpi=200)  


# Compute the mPOD Temporal Basis
PSI_M,Ks = mPOD_K(K,dt,Nf,Ex,F_V,Keep,'nearest','reduced');

# Save the correlation matrices of each scale:
for i in range(0,Ks.shape[2]):
 K=Ks[:,:,i]
 K_HAT_ABS=np.fliplr(np.abs(np.fft.fftshift(np.fft.fft2(K-np.mean(K)))));

 fig, ax = plt.subplots(figsize=(4,4))
 #ax.set_xlim([-0.5,0.5])
 #ax.set_ylim([-0.5,0.5])
 plt.rc('text', usetex=True)      
 plt.rc('font', family='serif')
 plt.rc('xtick',labelsize=12)
 plt.rc('ytick',labelsize=12)
 plt.pcolor(Freq,Freq,K_HAT_ABS/np.size(D)) # We normalize the result
 ax.set_aspect('equal') # Set equal aspect ratio
 ax.set_xlabel('${f}[Hz]$',fontsize=14)
 ax.set_ylabel('${f}[Hz]$',fontsize=14)
 ax.set_xticks(np.arange(-600,610,200))
 ax.set_yticks(np.arange(-600,610,200))
 ax.set_xlim(-600,600)
 ax.set_ylim(-600,600)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 plt.clim(0,0.5) # From here  downward is just for plotting purposes
 plt.tight_layout()
 Name='CS_Scale_'+str(i+1)
 print('Saving Fig '+Name)
 plt.savefig(Name, dpi=200) 

# Save as numpy array all the data
np.savez('Psis_mPOD',PSI_M=PSI_M)
 
# To make a comparison later, we also compute the POD basis
# Temporal structures are eigenvectors of K
data=np.load('Correlation_K.npz')
K=data['K']
Psi_P, Lambda_P, _ = np.linalg.svd(K)
# The POD has the unique feature of providing the amplitude of the modes
# with no need of projection. The amplitudes are:
Sigma_P=(Lambda_P)**0.5; 

# Obs: svd and eig on a symmetric positive matrix are equivalent.
np.savez('Psis_POD',Psi_P=Psi_P,Sigma_P=Sigma_P)
