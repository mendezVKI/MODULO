# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 15:46:44 2019

@author: mendez
"""

import numpy as np
from scipy.signal import firwin # To create FIR kernels
from scipy import signal
from scipy.sparse.linalg import svds


def Bound_EXT(S,Ex,boundaries):  
     """
    This function computes the extension of a signal for 
    filtering purposes
      
    :param S: The Input signal
    :param Nf: The Size of the Kernel (must be an odd number!) 
    :param boundaries: The type of extension:
         ‘reflect’ (d c b a | a b c d | d c b a)       The input is extended by reflecting about the edge of the last pixel.
            ‘nearest’ (a a a a | a b c d | d d d d)    The input is extended by replicating the last pixel.
            ‘wrap’ (a b c d | a b c d | a b c d)       The input is extended by wrapping around to the opposite edge.
            ‘extrap’ Extrapolation (not yet available) The input is extended via linear extrapolation.
            
   
    """
    # We first perform a zero padding
     #Ex=int((Nf-1)/2) # Extension on each size
     size_Ext=2*Ex+len(S) # Compute the size of the extended signal
     S_extend=np.zeros((size_Ext)) # Initialize extended signal
     S_extend[Ex:(int((size_Ext)-Ex))]=S; # Assign the Signal on the zeroes    
    
     if boundaries=="reflect":
        LEFT=np.flip(S[0:Ex]) # Prepare the reflection on the left
        RIGHT=np.flip(S[len(S)-Ex:len(S)]) # Prepare the reflectino on the right
        S_extend[0:Ex]=LEFT; 
        S_extend[len(S_extend)-Ex:len(S_extend)]=RIGHT
     elif boundaries=="nearest":
        LEFT=np.ones(Ex)*S[0] # Prepare the constant on the left
        RIGHT=np.ones(Ex)*S[len(S)-1] # Prepare the constant on the Right
        S_extend[0:Ex]=LEFT
        S_extend[len(S_extend)-Ex:len(S_extend)]=RIGHT
     elif boundaries=="wrap":
        LEFT=S[len(S)-Ex:len(S)] # Wrap on the Left
        RIGHT=S[0:Ex] # Wrap on the Right
        S_extend[0:Ex]=LEFT 
        S_extend[len(S_extend)-Ex:len(S_extend)]=RIGHT
     elif boundaries=="extrap":
        LEFT=np.ones(Ex)*S[0] # Prepare the constant on the left
        RIGHT=np.ones(Ex)*S[len(S)-1] # Prepare the constant on the Right
        S_extend[0:Ex]=LEFT
        S_extend[len(S_extend)-Ex:len(S_extend)]=RIGHT
        print('Not active yet, replaced by nearest')
     return S_extend

def conv_m(K,h,Ex,boundaries):  
     """
    This function computes the 2D convolution by perfoming 2 sets of 1D convolutions.
    Moreover, we here use the fft with an appropriate extension 
    that avoids the periodicity condition. 
      
    :param K: Matrix to be filtered
    :param h: The 1D Kernel of the filter 
    :param boundaries: The type of extension:
            ‘reflect’ (d c b a | a b c d | d c b a)    The input is extended by reflecting about the edge of the last pixel.
            ‘nearest’ (a a a a | a b c d | d d d d)    The input is extended by replicating the last pixel.
            ‘wrap’ (a b c d | a b c d | a b c d)       The input is extended by wrapping around to the opposite edge.
            ‘extrap’ Extrapolation (not yet available)
    """
    # Filter along the raws
     n_t=np.shape(K)[0]
     #Ex=int(n_t/2)
     K_F1=np.zeros(np.shape(K))
     K_F2=np.zeros(np.shape(K))
    # K_F=np.zeros(np.shape(K))
     for k in range(0,n_t):
       S=K[:,k]
       S_Ext=Bound_EXT(S,Ex,boundaries)
       S_Filt=signal.fftconvolve(S_Ext, h, mode='valid')
       # Compute where to take the signal
       Ex1=int((len(S_Filt)-len(S))/2)
     #  K_F1[k,:]=S_Filt[Ex:(len(S_Filt)-Ex)]
       K_F1[:,k]=S_Filt[Ex1:(len(S_Filt)-Ex1)]
     for k in range(0,n_t):
       S=K_F1[k,:]
       S_Ext=Bound_EXT(S,Ex,boundaries)
       S_Filt=signal.fftconvolve(S_Ext, h, mode='valid')
       # Compute where to take the signal
       Ex1=int((len(S_Filt)-len(S))/2)
      # K_F2[:,k]=S_Filt[Ex:(len(S_Filt)-Ex)]
       K_F2[k,:]=S_Filt[Ex1:(len(S_Filt)-Ex1)]       
     #K_F=K_F1+K_F2
     return K_F2


# Define the function to produce a gif file
def mPOD_K(K,dt,Nf,Ex,F_V,Keep,boundaries,MODE,SAVE_KS):
    """
    This function computes the Psi_M for the mPOD
    taking as input:
      
    :param K: The Temporal Correlation Matrix
    :param dt: The Time step 
    :param Nf: Nf the vector with order (must be odd!) of the kernels
    :param Ex: This is about the BC extension (must be odd!) of the signal before the convolution
    :param F_V: Frequency splitting vector
    :boundaries: this is a string defining the treatment for the BC
                 Options are (from scipy.ndimage.convolve):                  
            ‘reflect’ (d c b a | a b c d | d c b a)    The input is extended by reflecting about the edge of the last pixel.
            ‘nearest’ (a a a a | a b c d | d d d d)    The input is extended by replicating the last pixel.
            ‘wrap’ (a b c d | a b c d | a b c d)       The input is extended by wrapping around to the opposite edge.
    :MODE: this is about the final QR. Options
            ‘reduced’ In this case the final basis will not necessarely be full
            ‘complete’ In this case the final basis will always be full
    :SAVE_KS: This is 1 or 0. If 1, you will save and store all the intermediate scales
            
    :return: PSI_M: the mPOD temporal structures
    
    """
    if Ex<np.max(Nf):
     raise ValueError("Ex must be larger or equal to Nf") 
     return -1
    
   # Convert F_V in radiants
    Fs=1/dt
    F_Bank_r = F_V*2/Fs #(Fs/2 mapped to 1)
    M=len(F_Bank_r) # Number of scales
    
    # Loop over the scales to show the transfer functions
    Psi_M=np.array([])
    Lambda_M=np.array([])
    n_t=K.shape[1]
    if SAVE_KS==1:
     Ks=np.zeros((n_t,n_t,M+1))
    else:
     print('No Ks prepared')
          
 # Now you could filter and then compute the eigenvectors   
    for m in range(0,M):
       # Generate the 1d filter for this 
      if m<1:
       #Low Pass Filter
       h_A=firwin(Nf[m], F_Bank_r[m], window='hamming')
       #h_A2d=np.outer(h_A,h_A) # Create 2D Kernel
       # First Band Pass Filter
       # Create 1D Kernel
       # Filter K_LP
       print('Filtering Largest Scale')
       K_L=conv_m(K,h_A,Ex,boundaries)
       
       if SAVE_KS==1:
         Ks[:,:,m]=K_L # First large scale
       else:
         Ks=K_L 
       print('Diagonalizing Largest Scale')
       R_K=np.linalg.matrix_rank(K_L, tol=None, hermitian=True)
       Lambda_P,Psi_P= np.linalg.eigh(K_L)
       Psi_M=Psi_P[:,0:R_K] # In the first scale we take it as is
       Lambda_M=Lambda_P[0:R_K]
       
        # Construct first band pass
       if M>1:
        h1d_H= firwin(Nf[m],[F_Bank_r[m],F_Bank_r[m+1]],pass_zero=False) # Band-pass
       else:
        h1d_H= firwin(Nf[m],F_Bank_r[m],pass_zero=False) # Band-pass
  
       print('Filtering H Scale '+str(m+1)+'/'+str(M))
       K_H=conv_m(K,h1d_H,Ex,boundaries)
       if SAVE_KS==1:
         Ks[:,:,m+1]=K_H # Last (high pass) scale
       else:
         Ks=K_H 
       print('Diagonalizing H Scale '+str(m+1)+'/'+str(M))
       R_K=np.linalg.matrix_rank(K_H, tol=None, hermitian=True)
#       Psi_P, Lambda_P, _ = svds(K_H,R_K) # Diagonalize scale
#       Psi_M=np.concatenate((Psi_M, Psi_P), axis=1) # append to the previous
#       Lambda_M=np.concatenate((Lambda_M, Lambda_P), axis=0)
       Lambda_P,Psi_P= np.linalg.eigh(K_H)
       Psi_M=Psi_P[:,0:R_K] # In the first scale we take it as is
       Lambda_M=Lambda_P[0:R_K]
       
       #method = signal.choose_conv_method(K, h2d, mode='same')
      elif m>0 and m<M-1:         
       if Keep[m]==1:
       # print(m)
       # This is the 1d Kernel for Band pass
        h1d_H= firwin(Nf[m],[F_Bank_r[m],F_Bank_r[m+1]],pass_zero=False) # Band-pass
        print('Filtering H Scale '+str(m+1)+'/'+str(M))
        K_H=conv_m(K,h1d_H,Ex,boundaries)
        if SAVE_KS==1:
         Ks[:,:,m+1]=K_H # Last (high pass) scale
        else:
         Ks=K_H 
        print('Diagonalizing H Scale '+str(m+1)+'/'+str(M))
        R_K=np.linalg.matrix_rank(K_H, tol=None, hermitian=True)
#        Psi_P, Lambda_P, _ = svds(K_H,R_K) # Diagonalize scale
        Lambda_P,Psi_P= np.linalg.eigh(K_H)
        Lambda_M=np.concatenate((Lambda_M, Lambda_P[0:R_K]), axis=0)
        Psi_M=np.concatenate((Psi_M, Psi_P[:,0:R_K]), axis=1) # append to the previous
        
       else: 
         print('Scale Jumped') 
      else:
       if Keep[m]==1:
     # This is the 1d Kernel for High Pass (last scale)
        h1d_H = firwin(Nf[m],F_Bank_r[m],pass_zero=False)
        print('Filtering H Scale '+str(m+1)+'/ '+str(M))
        K_H=conv_m(K,h1d_H,Ex,boundaries)
        if SAVE_KS==1:
         Ks[:,:,m+1]=K_H # Last (high pass) scale
        else:
         Ks=K_H   
        print('Diagonalizing H Scale '+str(m+1)+'/ '+str(M))
        R_K=np.linalg.matrix_rank(K_H, tol=None, hermitian=True)
       # Psi_P, Lambda_P, _ = svds(K_H,R_K) # Diagonalize scale
       # Psi_M=np.concatenate((Psi_M, Psi_P), axis=1) # append to the previous
        Lambda_P,Psi_P= np.linalg.eigh(K_H)
        Lambda_M=np.concatenate((Lambda_M, Lambda_P[0:R_K]), axis=0)
        Psi_M=np.concatenate((Psi_M, Psi_P[:,0:R_K]), axis=1) # append to the previous
       else:
        print('Scale Jumped')
        
    # Now Order the Scales
    Indices=np.flip(np.argsort(Lambda_M)) # find indices for sorting in decreasing order
    Psi_M=Psi_M[:,Indices] # Sort the temporal structures
    # Now we complete the basis via re-orghotonalization
    print('QR Polishing...')
    PSI_M,R=np.linalg.qr(Psi_M,mode=MODE)
    print('Done!')
    if SAVE_KS==1:
     return PSI_M,Ks       
    else:
     return PSI_M   




