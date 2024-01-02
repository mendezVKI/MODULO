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

for j in range(1,12):
 plt.close(fig='all') 
 fig, ax3= plt.subplots(figsize=(5,7))   
 ax=plt.subplot(2,1,1)
 V_X=Phi_M[0:nxny,j-1]
 V_Y=Phi_M[nxny::,j-1]
 Plot_Field(X_S,Y_S,V_X,V_Y,True,2,1)
 ax.set_aspect('equal') # Set equal aspect ratio
 ax.set_xticks(np.arange(0,40,10))
 ax.set_yticks(np.arange(10,30,5))
 ax.set_xlim([0,35])
 ax.set_ylim([10,29])
 ax.invert_yaxis() # Invert Axis for plotting purpose
 String_y='$\phi_{\mathcal{P}'+str(j)+'}$'
 plt.title(String_y,fontsize=18)
# plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 
 ax=plt.subplot(2,1,2)
 Signal=Psi_M[:,j-1]
 s_h=np.abs((np.fft.fft(Signal-Signal.mean())))
 Freqs=np.fft.fftfreq(int(n_t))*Fs
 plt.plot(Freqs*(4/1000)/6.5,s_h,'-',linewidth=1.5)
 plt.xlim(0,0.5)    
 plt.xlabel('$St[-]$',fontsize=18)
 String_y='$\widehat{\psi}_{\mathcal{M}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 Name='mPOD_Mode_'+str(j)+'.png'
 print(Name+' Saved')
 plt.savefig(Name, dpi=300)  
#
## save POD onto a csv
#import os
#FOL='mPOD'; os.mkdir(FOL)
#np.savetxt(FOL+'/SIGMA.csv',Sigma_M)
#np.savetxt(FOL+'/phi_M_1_u.csv',Phi_M[0:nxny,0].reshape(n_x,n_y),delimiter=',')
#np.savetxt(FOL+'/phi_M_1_v.csv',Phi_M[nxny::,0].reshape(n_x,n_y),delimiter=',')
#np.savetxt(FOL+'/phi_M_2_u.csv',Phi_M[0:nxny,1].reshape(n_x,n_y),delimiter=',')
#np.savetxt(FOL+'/phi_M_2_v.csv',Phi_M[nxny::,1].reshape(n_x,n_y),delimiter=',')
#np.savetxt(FOL+'/phi_M_3_u.csv',Phi_M[0:nxny,2].reshape(n_x,n_y),delimiter=',')
#np.savetxt(FOL+'/phi_M_3_v.csv',Phi_M[nxny::,2].reshape(n_x,n_y),delimiter=',')
#
#np.savetxt(FOL+'/psi_M_1.csv',Psi_M[:,0])
#np.savetxt(FOL+'/psi_M_2.csv',Psi_M[:,1])
#np.savetxt(FOL+'/psi_M_3.csv',Psi_M[:,3])


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
  
for j in range(1,14):
 plt.close(fig='all') 
 fig, ax3= plt.subplots(figsize=(5,7))   
 ax=plt.subplot(2,1,1)
 V_X=Phi_P[0:nxny,j-1]
 V_Y=Phi_P[nxny::,j-1]
 Plot_Field(X_S,Y_S,V_X,V_Y,True,2,0.8)
 ax.set_aspect('equal') # Set equal aspect ratio
 ax.set_xticks(np.arange(0,40,10))
 ax.set_yticks(np.arange(10,30,5))
 ax.set_xlim([0,35])
 ax.set_ylim([10,29])
 ax.invert_yaxis() # Invert Axis for plotting purpose
 String_y='$\phi_{\mathcal{P}'+str(j)+'}$'
 plt.title(String_y,fontsize=18)
# plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 
 ax=plt.subplot(2,1,2)
 Signal=Psi_P[:,j-1]
 s_h=np.abs((np.fft.fft(Signal-Signal.mean())))
 Freqs=np.fft.fftfreq(int(n_t))*Fs
 plt.plot(Freqs*(4/1000)/6.5,s_h,'-',linewidth=1.5)
 plt.xlim(0,0.5)    
 plt.xlabel('$St[-]$',fontsize=18)
 String_y='$\widehat{\psi}_{\mathcal{P}'+str(j)+'}$'
 plt.ylabel(String_y,fontsize=18)
 plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
 Name='POD_Mode_'+str(j)+'.png'
 print(Name+' Saved')
 plt.savefig(Name, dpi=300)  
 

plt.close(fig='all')
# Amplitude Modes POD vs mPOD
RR_P=np.arange(1,Sigma_P.shape[0]+1)
RR_M=np.arange(1,Sigma_M.shape[0]+1)
## DFT Spectra and convergence
fig, ax = plt.subplots(figsize=(6,3))
plt.rc('text', usetex=True)     
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
sigma_P_v=(Sigma_P)/((n_s*n_t)**0.5)
sigma_M_v=(Sigma_M)/((n_s*n_t)**0.5)
plt.plot(RR_P,sigma_P_v,'ko',label='POD')
plt.plot(RR_M,sigma_M_v,'rs',label='mPOD')
plt.legend(fontsize=20,loc='upper right') 
plt.xlabel('$r$',fontsize=12)
plt.xlim(0.5,1001)
plt.xscale('log')
plt.ylabel('$\sigma_{\mathcal{P}r},\sigma_{\mathcal{M}r}*1000$',fontsize=14)
plt.title('POD vs mPOD Amplitudes ',fontsize=16)
plt.tight_layout(pad=0.6, w_pad=0.3, h_pad=0.8)
#pos1 = ax.get_position() # get the original position 
#pos2 = [pos1.x0 + 0.01, pos1.y0 + 0.01,  pos1.width *0.95, pos1.height *0.95] 
#ax.set_position(pos2) # set a new position
plt.savefig('POD_mPOD_Amplitude.png', dpi=200)     
plt.close(fig)

#
#FOL='POD'; os.mkdir(FOL)
#np.savetxt(FOL+'/SIGMA.csv',Sigma_P)
#np.savetxt(FOL+'/phi_P_1_u.csv',Phi_P[0:nxny,0].reshape(n_x,n_y),delimiter=',')
#np.savetxt(FOL+'/phi_P_1_v.csv',Phi_P[nxny::,0].reshape(n_x,n_y),delimiter=',')
#np.savetxt(FOL+'/phi_P_2_u.csv',Phi_P[0:nxny,1].reshape(n_x,n_y),delimiter=',')
#np.savetxt(FOL+'/phi_P_2_v.csv',Phi_P[nxny::,1].reshape(n_x,n_y),delimiter=',')
#np.savetxt(FOL+'/phi_P_3_u.csv',Phi_P[0:nxny,2].reshape(n_x,n_y),delimiter=',')
#np.savetxt(FOL+'/phi_P_3_v.csv',Phi_P[nxny::,2].reshape(n_x,n_y),delimiter=',')
#
#np.savetxt(FOL+'/psi_P_1.csv',Psi_P[:,0])
#np.savetxt(FOL+'/psi_P_2.csv',Psi_P[:,1])
#np.savetxt(FOL+'/psi_P_3.csv',Psi_P[:,3])

# Finally, perform also the DFT
n_t=int(n_t)
Freqs=np.fft.fftfreq(n_t)*Fs # Compute the frequency bins
####### DFT ########
## For comparison putposes we also perform here a DFT.
print('Compute also the DFT')
PSI_F=np.conj(np.fft.fft(np.eye(n_t))/np.sqrt(n_t)) # Prepare the Fourier Matrix.
print('Projecting Data')
#D_Complex=D+1j*np.zeros(D.shape)
# Method 1 (didactic!)
PHI_SIGMA=np.dot(D,np.conj(PSI_F)) # This is PHI * SIGMA

# Method 2
#PHI_SIGMA=(np.fft.fft(D,n_t,1))/(n_t**0.5)

PHI_F=np.zeros((D.shape[0],n_t),dtype=complex) # Initialize the PHI_F MATRIX
SIGMA_F=np.zeros(n_t) # Initialize the SIGMA_F MATRIX

# Now we proceed with the normalization. This is also intense so we time it
for r in range(0,n_t): # Loop over the PHI_SIGMA to normalize
  MEX='Proj '+str(r)+' /'+str(n_t) 
  print(MEX)
  SIGMA_F[r]=abs(np.vdot(PHI_SIGMA[:,r],PHI_SIGMA[:,r]))**0.5
  PHI_F[:,r]=PHI_SIGMA[:,r]/SIGMA_F[r]

Sigma_F_n=SIGMA_F/(n_y*n_t) # Carefull with the normalization

Indices=np.flipud(np.argsort(SIGMA_F)) # find indices for sorting in decreasing order
Sorted_Sigmas=SIGMA_F[Indices] # Sort all the sigmas
Sorted_Freqs=Freqs[Indices] # Sort all the frequencies accordingly.
Phi_F=PHI_F[:,Indices] # Sorted Spatial Structures Matrix
Psi_F=PSI_F[:,Indices] # Sorted Temporal Structures Matrix
SIGMA_F=np.diag(Sorted_Sigmas) # Sorted Amplitude Matrix


#
#FOL='DFT_product'; os.mkdir(FOL)
#SIGMA_F_OUT=np.zeros((len(Freqs),2));
#SIGMA_F_OUT[:,0]=Freqs; SIGMA_F_OUT[:,1]=Sigma_F_n
#np.savetxt(FOL+'/SIGMA.csv',Sigma_P)
#np.savetxt(FOL+'/phi_F_1_u_R.csv',np.real(Phi_P[0:nxny,0].reshape(n_x,n_y)),delimiter=',')
#np.savetxt(FOL+'/phi_F_1_v_R.csv',np.real(Phi_P[nxny::,0].reshape(n_x,n_y)),delimiter=',')
#np.savetxt(FOL+'/phi_F_2_u_R.csv',np.real(Phi_P[0:nxny,1].reshape(n_x,n_y)),delimiter=',')
#np.savetxt(FOL+'/phi_F_2_v_R.csv',np.real(Phi_P[nxny::,1].reshape(n_x,n_y)),delimiter=',')
#np.savetxt(FOL+'/phi_F_3_u_R.csv',np.real(Phi_P[0:nxny,2].reshape(n_x,n_y)),delimiter=',')
#np.savetxt(FOL+'/phi_F_3_v_R.csv',np.real(Phi_P[nxny::,2].reshape(n_x,n_y)),delimiter=',')
#




