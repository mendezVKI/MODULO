# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:05:53 2023

@author: loren
"""

import numpy as np
import pickle

FOLDER = './modulo_vki/examples/ex_4_dataJet/'
#
# Name = FOLDER + 'field0.0.csv'
# Dat = np.loadtxt(Name, skiprows=1, delimiter=',')

with open(FOLDER + 'dataJet', 'rb') as f:
    Dat = pickle.load(f)

data = Dat['U']
grid = Dat['grid']

with open(FOLDER + 'timeJet', 'rb') as f:
    time = pickle.load(f)
# %% Enforce uniform sampling

dts, counts= np.unique(np.diff(time), return_counts=True)
dt = dts[2]

indexes = np.where(np.diff(time) == dt)[0]

# %%
N_S = grid.shape[0] # Number of mesh points
N = 2

U_resampled = [data[idx] for idx in indexes]

# %% Load the data and the mesh
import os

D = np.zeros((N_S * 2, counts[2]))

for i in range(counts[2]):
    tmp = U_resampled[i]
    d_i = np.concatenate([tmp[:, 0], tmp[:, 1]], axis=0)
    D[:, i] = d_i

# %%
from modulo_vki.modulo.modulo import MODULO
X, Y = grid[:, 0], grid[:, 1] #np.meshgrid(

m = MODULO(data=np.nan_to_num(D), svd_solver='svd_scipy_sparse', is_cartesian=(False, (X, Y)))
# %%
Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_K()

# %%
import matplotlib.pyplot as plt
from modulo_vki.modulo.utils.others import Plot_Field_JET

# plt.rc('text', usetex=True)  # This is Miguel's customization
# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize=16)
# plt.rc('ytick', labelsize=16)

nxny=int(len(Phi_POD[:,0])/2)

U_D=np.real(Phi_POD)

for k in range(1, 6):
    fig, axs = plt.subplots(2, 1, figsize=(5, 7))
    n_x, n_y = np.shape(grid)
    U_g=U_D[:nxny ,k]
    V_g=U_D[nxny:, k]

    GRAD_X = np.diff(X)
    # GRAD_Y=np.diff(Y_S);
    # Depending on the reshaping performed, one of the two will start with
    # non-zero gradient. The other will have zero gradient only on the change.
    IND_X = np.where(GRAD_X != 0);
    DAT = IND_X[0];
    n_y = DAT[0] + 1;
    nxny = X.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
    # n_s=2*nxny
    # Reshaping the grid from the data
    n_x = (nxny // (n_y))  # Carefull with integer and float!
    Xg = np.transpose(X.reshape((n_x, n_y)))
    Yg = np.transpose(Y.reshape((n_x, n_y)))  # This is now the mesh! 60x114.
    Mod = np.sqrt(U_g ** 2 + V_g ** 2)
    Vxg = np.transpose(U_g.reshape((n_x, n_y)))
    Vyg = np.transpose(V_g.reshape((n_x, n_y)))
    Magn = np.transpose(Mod.reshape((n_x, n_y)))

    STEPx = 10
    STEPy = 10
    # axs[0].contourf(Xg, Yg, Magn)
    axs[0].quiver(Xg[::STEPx, ::STEPy], Yg[::STEPx, ::STEPy], \
              Vxg[::STEPx, ::STEPy], -Vyg[::STEPx, ::STEPy], color='k',
              scale=0.5)  # Create a quiver (arrows) plot

    #Xg, Yg, Vxg, Vyg, Magn = Plot_Field_JET(X, Y, V_g, U_g,  True, 10, 0.2)
    #ax.set_aspect('equal')  # Set equal aspect ratio
    #ax.invert_yaxis()  # Invert Axis for plotting purpose
    # ax.invert_xaxis()  # Invert Axis for plotting purpose
    String_y = '$\phi_{\mathcal{M}' + str(k) + '}$'
    axs[0].set_title(String_y, fontsize=18)
    #axs[0].tight_layout(pad=1, w_pad=0.5, h_pad=1.0)

    Signal = Psi_POD[:, k]
    s_h = np.abs((np.fft.fft(Signal - Signal.mean())))
    Freqs = np.fft.fftfreq(int(counts[2])) / dt
    axs[1].plot(Freqs, s_h, '-', linewidth=1.5) #* 0.001 / 100
    axs[1].set_xlim(-0.001, 5000)
    axs[1].set_xlabel('$f$ (Hz)', fontsize=18)
    String_y = '$\widehat{\psi}_{\mathcal{P}' + str(k) + '}$'
    plt.ylabel(String_y, fontsize=18)
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
    plt.show()
