# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:47:48 2019

@author: mendez
"""


import numpy as np
import matplotlib.pyplot as plt
from mPOD_Functions import Bound_EXT
from mPOD_Functions import conv_m
from scipy.signal import firwin # To create FIR kernels
from scipy.signal import convolve2d


# DEbuggig K filtering

# Load all the dataset and the correlation matrix
data = np.load('Data.npz')
D=data['D']; t=data['t']; dt=data['dt']; n_t=data['n_t']
y=data['y']; dy=data['dy']; n_y=data['n_y']
# Load the correlation matrix
data=np.load('Correlation_K.npz')
K=data['K']




Nf=np.array([201,201,201]); # This vector collects the length of the filter kernels.
# Observe that Nf could be set as it is usually done in Wavelet Theory.
# For example, using eq. A.5.
F_V=np.array([0.1,0.25,0.35]) # This will generate three scales: H_A, H_H_1, H_H_2. See Sec. 3.2
Keep=np.array([1, 0,1]) #These are the band-pass you want to keep (Approximation is always kept).
Fs=2
# We can visualize where these are acting
F_Bank_r = F_V*2/Fs; #(Fs/2 mapped to 1)
M=len(F_Bank_r); # Number of scales
# Plot the transfer functions along the diagonal of K (similar to Fig 1)
m=0
h_A=firwin(Nf[m], F_Bank_r[m], window='hamming')
h1d_H= firwin(Nf[m],[F_Bank_r[m],F_Bank_r[m+1]],pass_zero=False) # Band-pass
K_L=conv_m(K,h1d_H,int(n_t/2),"nearest")


fig, ax = plt.subplots(figsize=(8,5))
plt.subplot(2, 1, 1)
plt.plot(K[444,:])
plt.plot(K_L[444,:])
plt.subplot(2, 1, 2)
plt.pcolor(K_L)

#K_Ld=ndimage.convolve(K,np.outer(h1d_H,h1d_H),mode="nearest")
#K_Ld2=convolve2d(K,np.outer(h1d_H,h1d_H),mode="same")

#K_H1=conv_m(K,h1d_H,int(n_t/2),"nearest")

#K_H2=ndimage.convolve(K, np.outer(h1d_H,h1d_H), mode='nearest')

#K_DIFF=K_H1-K_H2


