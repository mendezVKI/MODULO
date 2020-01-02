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
data=np.load('Correlation_K.npz')
K=data['K']

## Study the frequency content of K
## We plot || K_hat || to look for frequencies
# This could be done via Matrix Multiplication (as it is done in the paper)
# but we use fft2 for fast computation. If you want to see the algebraic form of the code,
# do not hesitate to contact me at mendez@vki.ac.be

plt.ion() # To enable interactive plotting
Fs=1/dt; # Sampling frequency
Freq=np.fft.fftshift(np.fft.fftfreq(int(n_t)))*Fs # Frequency Axis
# Compute the 2D FFT of K
K_HAT_ABS=np.abs(np.fft.fftshift(np.fft.fft2(K-np.mean(K))));

fig, ax = plt.subplots(figsize=(5,5))
plt.pcolor(Freq,Freq,K_HAT_ABS/np.size(D)) # We normalize the result
plt.ylabel('$\hat{f}[-]$',fontsize=18)
plt.xlabel('$\hat{f}[-]$',fontsize=18)
plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
plt.clim(0,0.5) # From here  downward is just for plotting purposes
plt.xticks(np.arange(-600,610,200))
plt.yticks(np.arange(-600,610,200))
plt.axis('equal')
plt.ylim([-600,600]);plt.xlim([-600,600])
plt.tight_layout()

# This matrix shows that there are two dominant frequencies in the problem.
# We set a frequency splitting to divide these two portions of the
# spectra. (See Sec. 3.1-3.2 of the article)

H=4/1000; # Stand off distance nozzle to plate
U0=6.5; # Mean velocity of the jet at the outlet

F_V=np.array([100,200,300,400]); # This will generate four scales: H_A, H_H_1, H_H_2, H_H_3. See Sec. 3.2
St=F_V*H/U0;

Keep=np.array([1,1,1,1]); #These are the band-pass you want to keep (Approximation is always kept).
Nf=np.array([201,201,201,201]); # This vector collects the length of the filter kernels.


# We can visualize where these are acting
F_Bank_r = F_V*2/Fs; #(Fs/2 mapped to 1)
M=len(F_Bank_r); # Number of scales
# Plot the transfer functions along the diagonal of K (similar to Fig 1)


fig, ax = plt.subplots(figsize=(8,5))
plt.rc('text', usetex=True)      # This is Miguel's customization
#plt.rc('font', family='serif')
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)

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
plt.xlabel('$\hat{f}[Hz]$',fontsize=18)
plt.ylabel('Normalized Spectra',fontsize=18)

plt.tight_layout()
plt.savefig('Frequency_Splitting.pdf', dpi=100)  


Ex=201
# Compute the mPOD Temporal Basis
PSI_M = mPOD_K(K,dt,Nf,Ex,F_V,Keep,'nearest','reduced');


# Save as numpy array all the data
np.savez('Psis_mPOD',PSI_M=PSI_M)
 
# To make a comparison later, we also compute the POD basis
# Temporal structures are eigenvectors of K
Psi_P, Lambda_P, _ = np.linalg.svd(K)
# The POD has the unique feature of providing the amplitude of the modes
# with no need of projection. The amplitudes are:
Sigma_P=(Lambda_P)**0.5; 

# Obs: svd and eig on a symmetric positive matrix are equivalent.
np.savez('Psis_POD',Psi_P=Psi_P,Sigma_P=Sigma_P)
