# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 02:32:03 2019

@author: mendez
"""

import numpy as np
from scipy.signal import firwin # To create FIR kernels
from mPOD_Functions import conv_m

# Define the function to produce a gif file
def mPOD_K(K,dt,Nf,F_V,Keep,boundaries):
    """
    This function computes the Psi_M for the mPOD
    taking as input:
      
    :param K: The Temporal Correlation Matrix
    :param dt: The Time step 
    :param pair: Nf the vector with order (must be odd!) of the kernels
    :param F_V: Frequency splitting vector
    :boundaries: this is a string defining the treatment for the BC
                 Options are (from scipy.ndimage.convolve):                  
            ‘reflect’ (d c b a | a b c d | d c b a)    The input is extended by reflecting about the edge of the last pixel.
            ‘nearest’ (a a a a | a b c d | d d d d)    The input is extended by replicating the last pixel.
            ‘wrap’ (a b c d | a b c d | a b c d)       The input is extended by wrapping around to the opposite edge.
    
    :return: PSI_M: the mPOD temporal structures
    
    """
   # Convert F_V in radiants
    n_t=np.shape(K)[0] 
    Fs=1/dt
    F_Bank_r = F_V*2/Fs #(Fs/2 mapped to 1)
    M=len(F_Bank_r) # Number of scales
    
    # Loop over the scales to show the transfer functions
    List_Psi=[]
    List_Lambdas=[]
              
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
       K_L=conv_m(K,h_A,int(n_t/2),"reflect")
       print('Diagonalizing Largest Scale')
       Psi_P, Lambda_P, _ = np.linalg.svd(K_L)
       List_Psi=[Psi_P]
       List_Lambdas=[Lambda_P]
       
       h1d_H= firwin(Nf[m],[F_Bank_r[m],F_Bank_r[m+1]],pass_zero=False) # Band-pass
       print('Filtering H Scale '+str(m+1)+'/'+str(M))
       K_H=conv_m(K,h1d_H,int(n_t/2),boundaries)
       print('Diagonalizing H Scale '+str(m+1)+'/'+str(M))
       Psi_P, Lambda_P, _ = np.linalg.svd(K_H)
       List_Psi.append(Psi_P)
       List_Lambdas.append(Lambda_P)
       #method = signal.choose_conv_method(K, h2d, mode='same')
      elif m>0 and m<M-1:         
       if Keep[m]==1:
       # print(m)
       # This is the 1d Kernel for Band pass
        h1d_H= firwin(Nf[m],[F_Bank_r[m],F_Bank_r[m+1]],pass_zero=False) # Band-pass
        print('Filtering H Scale '+str(m+1)+'/'+str(M))
        K_H=conv_m(K,h1d_H,int(n_t/2),boundaries)
        print('Diagonalizing H Scale '+str(m+1)+'/'+str(M))
        Psi_P, Lambda_P, _ = np.linalg.svd(K_H)
        List_Psi.append(Psi_P)
        List_Lambdas.append(Lambda_P)
       else: 
        print('Scale Jumped') 
      else:
       if Keep[m]==1:
     # This is the 1d Kernel for High Pass (last scale)
        h1d_H = firwin(Nf[m],F_Bank_r[m],pass_zero=False)
        print('Filtering H Scale '+str(m+1)+'/ '+str(M))
        K_H=conv_m(K,h1d_H,int(n_t/2),boundaries)
        print('Diagonalizing H Scale '+str(m+1)+'/ '+str(M))
        Psi_P, Lambda_P, _ = np.linalg.svd(K_H)
        List_Psi.append(Psi_P)
        List_Lambdas.append(Lambda_P)
       else:
        print('Scale Jumped')
    return List_Psi        
    





