# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:25:47 2026

@author: belda
"""

'''
This sixth tutorial illustrates how to compute the fast multiscale POD of a simple dataset. The dataset is artificially generated and contains 3 modes with different temporal structures (two oscillatory modes with different frequencies and slopes, and one mode containing a single wavelet). The dataset is designed to be simple enough to allow for a clear comparison of the results of the fast mPOD with the classical FIR-based mPOD. The fast mPOD is called in two ways: once with the tapering and once without (i.e. rectangular spectral mask). The results are compared in terms of singular values and temporal modes.

For an overview of the theory of fast mPOD, see [Belda et al. (2026)] (https://arxiv.org/abs/2604.12077).

It is important to note that for fortran support for correlation matrix assembly (useFortran=True in the fast mPOD call), the fast mPOD routine requires the data to be passed as a np.float64 array during modulo object initialization. This is because the Fortran code is written assuming double precision format. When the fortran support is not used, the data can be passed in any format. To run this tutorial with fortran support, simply change the 'useFortran' flag to 'useFortran = True' in the code below.
'''

import numpy as np
from modulo_vki.modulo import ModuloVKI
import os
from math import pi
import matplotlib.pyplot as plt


##### PLOT CUSTOMIZATION #####
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
size1 = (8, 2.5)     # figure size for overall evolution
size2 = (3, 2.5)       # figure size for details
resolution = 600    # figure resolution in DPI

target_dir = "./ex6_results"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

fig_target = target_dir + "/tempEvolFigs"
if not os.path.exists(fig_target):
    os.makedirs(fig_target)

# Create artificial spatial structure
toposes = np.zeros((3000,3))
toposes[:1000,0] = 1
toposes[1000:2000,1] = 1
toposes[2000:,2] = 1
toposes = toposes/np.linalg.norm(toposes,axis=0)

# Associate importance with the test modes
sigmas = np.diag([6000,4000,2000])

# Create artificial temporal structure
nt = 6000   # number of time steps
chronoses = np.zeros((nt,3))
for i in range(2000):
    chronoses[i,0] = np.sin(2*pi*i/60)+(i-500)/1200     # f1/fs = 1/60 + slope
for i in range(4000,6000):
    chronoses[i,1] = np.sin(2*pi*(i-4000)/30)-(i-4000)/800   # f2/fs = 1/30 + slope

# single wavelet
for i in range(2200,3000):
    chronoses[i,2] = 1
for i in range(3000,3800):
    chronoses[i,2] = -1    

for i in range(3):
    chronoses[:,i] = chronoses[:,i]/np.linalg.norm(chronoses[:,i])

# Assemble the data matrix
D = toposes @ sigmas @ chronoses.T

# Save temporal evolutions
chronTot = np.zeros((chronoses.shape[0],chronoses.shape[1]+1))
chronTot[:,0] = np.arange(nt)
chronTot[:,1:] = chronoses
np.save(f"{target_dir}/chronoses.npy", chronTot)

# Frequency split and selection of scales
useFortran = True  # Whether to use Fortran support for correlation matrix assembly in fast mPOD.
nModes = 10     # Number of modes to compute
F_V = [1/120,3/120,5/120,10/120]   # Frequency splitting vector in Hz, assuming fs = 1 (normalized frequencies) for this tutorial
Keep = [1, 1, 1, 1, 0]  # Which scales to keep (0 means not processed, 1 means processed)
winType = "hann" # Type of window for the tapering (hann, hamming, blackman, etc)
taper = [1/300, 1/300, 1/300, 1/300, 1/300] # Taper width in Hz for each scale, assuming fs = 1 (normalized frequencies) for this tutorial
mode = "fullK" # Type of computation (supports "fullK", "bandK", "fullSVD", "randSVD")


# MODULO object initialization
if useFortran:
    m = ModuloVKI(data=np.nan_to_num(D), n_Modes = nModes, dtype=np.float64)  # dtype = np.float64 required for fortran support
else:
    m = ModuloVKI(data=np.nan_to_num(D), n_Modes = nModes)

# Fast mPOD call
phi, psi, sig = m.fastmPOD(F_V, 1, Keep, winType = winType, taper = taper, mode = mode, GThresh=5, ncpus=9, useFortran=useFortran) 
np.save(f"{target_dir}/phi.npy", phi)
np.save(f"{target_dir}/psi.npy", psi)
np.save(f"{target_dir}/sig.npy", sig)

# Rectangular mask
phiNT, psiNT, sigNT = m.fastmPOD(F_V, 1, Keep, winType = winType, taper = None, mode = mode, GThresh=5, ncpus=9, useFortran=useFortran)
np.save(f"{target_dir}/phiNT.npy", phiNT)
np.save(f"{target_dir}/psiNT.npy", psiNT)
np.save(f"{target_dir}/sigNT.npy", sigNT)

# Parameters for FIR approach (classic mPOD approach)
Nf = [1001, 1001, 1001, 1001, 1001]    # FIR orders for all scales
Nf = np.array(Nf)
Ex = np.max(Nf)+2  # This must be at least as Nf.
dt = 1; boundaries = 'wrap'; MODE = 'reduced'

Phi_M, Psi_M, Sigmas_M = m.mPOD(Nf, Ex, F_V, Keep, nModes ,boundaries, MODE, dt, False)
np.save(f"{target_dir}/phiFIR.npy", Phi_M)
np.save(f"{target_dir}/Psi_M.npy", Psi_M)
np.save(f"{target_dir}/sigFIR.npy", Sigmas_M)


# Compare singular values
sigDiff = sig - Sigmas_M
sigDiffNT = sigNT - Sigmas_M
print("\n--------------------------")
print("RESULTS:")
print("\nSingular values fast mPOD:")
print(sig)
print("\nSingular values rectangular mask fast mPOD:")
print(sigNT)
print("\nSingular values classical (FIR-based) mPOD:")
print(Sigmas_M)


print("\nSingular values difference (absolute) fast vs FIR:")
print(sigDiff)
print("\nSingular values difference (relative) fast vs FIR:")
print(sigDiff/Sigmas_M)


print("\nSingular values difference (absolute) rectangular mask vs FIR:")
print(sigDiffNT)
print("\nSingular values difference (relative) rectangular mask vs FIR:")
print(sigDiffNT/Sigmas_M)


plt.figure(figsize=size1)
plt.plot(np.arange(nt),chronoses[:,0],label=r"$\Psi_{T1} \; [-]$")
plt.plot(np.arange(nt),chronoses[:,1],linestyle='--',label=r"$\Psi_{T2} \; [-]$")
plt.plot(np.arange(nt),chronoses[:,2],linestyle='-.',label=r"$\Psi_{T3} \; [-]$")
ax = plt.gca()
ax.set_aspect('auto', adjustable='box')
ax.grid(which = 'both')
#plt.title("Time evolution mode %d"%(i+1))
plt.xlabel(r"$i \; [-]$")
plt.ylabel(r'$\Psi \; [-]$')
plt.xlim([0,nt])
plt.legend()
plt.savefig(fig_target + "/chronoses.pdf", dpi=resolution, bbox_inches="tight")
plt.close()

# Compare temporal modes
for i in range(nModes):
    # Plot time evolution of mode

    minus = np.linalg.norm(psi[:,i]-Psi_M[:,i])
    plus = np.linalg.norm(psi[:,i]+Psi_M[:,i])

    if plus < minus:
        psi[:,i] = -1*psi[:,i]

    minus = np.linalg.norm(psiNT[:,i]-Psi_M[:,i])
    plus = np.linalg.norm(psiNT[:,i]+Psi_M[:,i])

    if plus < minus:
        psiNT[:,i] = -1*psiNT[:,i]
        
    plt.figure(figsize=size1)
    plt.plot(np.arange(nt),Psi_M[:,i],label="FIR-based mPOD")
    plt.plot(np.arange(nt),psi[:,i],linestyle='--',label="fast mPOD")
    plt.plot(np.arange(nt),psiNT[:,i],linestyle='-.',label="rectangular spectral mask")
    ax = plt.gca()
    ax.set_aspect('auto', adjustable='box')
    ax.grid(which = 'both')
    #plt.title("Time evolution mode %d"%(i+1))
    plt.xlabel(r"$i \; [-]$")
    plt.ylabel(r'$\Psi_{%d} \; [-]$'%(i+1))
    plt.xlim([0,nt])
    plt.legend()
    plt.savefig(fig_target + "/Mode%d.pdf"%(i+1), dpi=resolution, bbox_inches="tight")
    plt.close()
    

    # Details around jumps
    plt.figure(figsize=size2)
    plt.plot(np.arange(500),Psi_M[:500,i],label="FIR-based mPOD")
    plt.plot(np.arange(500),psi[:500,i],linestyle='--',label="fast mPOD")
    plt.plot(np.arange(500),psiNT[:500,i],linestyle='-.',label="rectangular spectral mask")
    ax = plt.gca()
    ax.set_aspect('auto', adjustable='box')
    ax.grid(which = 'both')
    #plt.title("Time evolution mode %d"%(i+1))
    plt.xlabel(r"$i \; [-]$")
    #plt.ylabel(r'$\Psi_{%d} \; [-]$'%(i+1))
    plt.xlim([0,499])
    #plt.legend(loc='lower right')
    plt.savefig(fig_target + "/Mode%dDet1.pdf"%(i+1), dpi=resolution, bbox_inches="tight")
    plt.close()

    # Details around jumps
    plt.figure(figsize=size2)
    plt.plot(np.arange(1500,2500),Psi_M[1500:2500,i],label="FIR-based mPOD")
    plt.plot(np.arange(1500,2500),psi[1500:2500,i],linestyle='--',label="fast mPOD")
    plt.plot(np.arange(1500,2500),psiNT[1500:2500,i],linestyle='-.',label="rectangular spectral mask")
    ax = plt.gca()
    ax.set_aspect('auto', adjustable='box')
    ax.grid(which = 'both')
    #plt.title("Time evolution mode %d"%(i+1))
    plt.xlabel(r"$i \; [-]$")
    #plt.ylabel(r'$\Psi_{%d} \; [-]$'%(i+1))
    plt.xlim([1500,2500])
    #plt.legend()
    plt.savefig(fig_target + "/Mode%dDet2.pdf"%(i+1), dpi=resolution, bbox_inches="tight")
    plt.close()

    # Details around jumps
    plt.figure(figsize=size2)
    plt.plot(np.arange(2750,3250),Psi_M[2750:3250,i],label="FIR-based mPOD")
    plt.plot(np.arange(2750,3250),psi[2750:3250,i],linestyle='--',label="fast mPOD")
    plt.plot(np.arange(2750,3250),psiNT[2750:3250,i],linestyle='-.',label="rectangular spectral mask")
    ax = plt.gca()
    ax.set_aspect('auto', adjustable='box')
    ax.grid(which = 'both')
    #plt.title("Time evolution mode %d"%(i+1))
    plt.xlabel(r"$i \; [-]$")
    #plt.ylabel(r'$\Psi_{%d} \; [-]$'%(i+1))
    plt.xlim([2750,3250])
    #plt.legend()
    plt.savefig(fig_target + "/Mode%dDet3.pdf"%(i+1), dpi=resolution, bbox_inches="tight")
    plt.close()

    # Details around jumps
    plt.figure(figsize=size2)
    plt.plot(np.arange(3500,4500),Psi_M[3500:4500,i],label="FIR-based mPOD")
    plt.plot(np.arange(3500,4500),psi[3500:4500,i],linestyle='--',label="fast mPOD")
    plt.plot(np.arange(3500,4500),psiNT[3500:4500,i],linestyle='-.',label="rectangular spectral mask")
    ax = plt.gca()
    ax.set_aspect('auto', adjustable='box')
    ax.grid(which = 'both')
    #plt.title("Time evolution mode %d"%(i+1))
    plt.xlabel(r"$i \; [-]$")
    #plt.ylabel(r'$\Psi_{%d} \; [-]$'%(i+1))
    #plt.legend()
    plt.xlim([3500,4500])
    plt.savefig(fig_target + "/Mode%dDet4.pdf"%(i+1), dpi=resolution, bbox_inches="tight")
    plt.close()

    # Details around jumps
    plt.figure(figsize=size2)
    plt.plot(np.arange(5500,6000),Psi_M[5500:,i],label="FIR-based mPOD")
    plt.plot(np.arange(5500,6000),psi[5500:,i],linestyle='--',label="fast mPOD")
    plt.plot(np.arange(5500,6000),psiNT[5500:,i],linestyle='-.',label="rectangular spectral mask")
    ax = plt.gca()
    ax.set_aspect('auto', adjustable='box')
    ax.grid(which = 'both')
    #plt.title("Time evolution mode %d"%(i+1))
    plt.xlabel(r"$i \; [-]$")
    #plt.ylabel(r'$\Psi_{%d} \; [-]$'%(i+1))
    #plt.legend()
    plt.xlim([5500,5999])
    plt.savefig(fig_target + "/Mode%dDet5.pdf"%(i+1), dpi=resolution, bbox_inches="tight")
    plt.close()