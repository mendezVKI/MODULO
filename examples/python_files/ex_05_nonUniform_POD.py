
"""
Created on Tue Jan  9 18:00:15 2024

@author: Poletti, mendez
"""


import numpy as np
import pyvista as pv  
import pickle
import random
import matplotlib.pyplot as plt
import os
import urllib.request
from zipfile import ZipFile

from modulo.utils.read_db import ReadData # to read the data

''' 

This exercise performs POD decomposition on a non-uniform grid.
The database is the velocity fields exctracted from an unsteady CFD simulation of a jet impinging a flat plate (in OpenFOAM).
The simulation is 3D but the data are extracted from a 2D slice, neglecting the (small) out-of-plane velocity component.

The grid of this slice is non-uniform and the velocity fields are saved with constant time steps.
The non-uniformity of the data adds the need for a weight vectors that accounts for the differences in mesh side.

We assume that the data is exported in vtk format and we use pyvista to interact with this data.
For more info about pyvista, please see:

    https://docs.pyvista.org/version/stable/examples/99-advanced/openfoam-tubes.html
    
Note: the pyvista windows must be closed to let the code continue to run.

A special note is required for the memory saving in this case:
The file was stored in a single vtk. Therefore, it is here loaded as a single block.
If the memory saving is active (N_PARTITIONS>1), we will split the D matrix in block.   

'''

# To activate memory saving feature put N_PARTITIONS > 1.
N_PARTITIONS=10

# Plot settings
camera_position = [(-0.004, 0.007, 0.03), 
                   (-0.004, 0.007, 0.0035),
                   (-1, 0, 0)]
font_size = 12

# Download files
FOLDER = 'Tutorial_5_non_uniform_GRID'

url = 'https://osf.io/aqkc8/download'
urllib.request.urlretrieve(url, 'Ex_8_2D_Impinging_JET_CFD.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file
String = 'Ex_8_2D_Impinging_JET_CFD.zip'
zf = ZipFile(String, 'r')
zf.extractall('./')
zf.close()
os.rename('Ex_8_2D_Impinging_JET_CFD', FOLDER)  # rename the data flolder to FOLDER
os.remove(String)  # Delete the zip file with the data
print('Data set unzipped and ready ! ')



# Folder where data are taken from
folder_in_path = FOLDER+os.sep

# Folder where results are saved in
folder_out_name = "RESULTS_ex_05"

folder_out_path = os.path.join(FOLDER, folder_out_name)
if not os.path.exists(folder_out_path):
    os.makedirs(folder_out_path)

#%%
# Velocity field
with open(folder_in_path + 'dataJet_uniform', 'rb') as f:
    Dat = pickle.load(f)
UList  = Dat['U']

# Grid
gridData = pv.read(folder_in_path + 'grid.vtk')

#%% Plot of the velocity field of the jet

# Random pick of a time step
randSample = random.randint(0, len(UList))
gridData['Urandom']=UList[randSample].astype('float32')

# Fancy color map
cmap = plt.cm.gist_rainbow
cmap_reversed = plt.cm.get_cmap('gist_rainbow_r')

print('Close the window to continue ...')
pl = pv.Plotter()
pl.add_mesh(gridData, scalars='Urandom',lighting=False, cmap=cmap_reversed,
            scalar_bar_args={'title': 'Velocity magnitude (m/s)'}, 
            show_edges=False,copy_mesh=True) #,edge_opacity=0)
pl.enable_anti_aliasing()
pl.show_axes_all() # show frame
pl.show_grid(font_size=12)
pl.camera_position =camera_position
pl.show()

#%% # Make a gif of the jet and save it in folder_out_path
pl = pv.Plotter(notebook=False, off_screen=True)
pl.open_gif(folder_out_path + "/jet_velocity.gif")
pl.camera_position = camera_position

nframe = 4
print("Writing gif")
for i in range(0,len(UList),nframe):
    print(i,'/',len(UList),end='\r')
    gridData['Ugif'] = UList[i].astype('float32')
    pl.add_mesh(gridData, scalars="Ugif", lighting=False, show_edges=False, cmap=cmap_reversed, scalar_bar_args={'title': 'Velocity magnitude (m/s)'}, clim=[0, 160], copy_mesh=True)
    pl.write_frame()
pl.close()

#%% From points to cell centres
# https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.CompositeFilters.point_data_to_cell_data.html

# For the grid ...
cellCentres  = (gridData.cell_centers()).points

# ... and for the velocity field
for i in range(len(UList)):
    gridData['U']=UList[i].astype('float32')            # Load the velocity into the grid
    gridData_cC = gridData.point_data_to_cell_data()    # Interpolate U from the points to the cell centres
    UList[i] = gridData_cC['U']                         # Save the cell centres velocity field in the grid

#%% Create matrix D
print("Creating the data matrix D")
N_P = cellCentres.shape[0]       # Number of mesh points
N   = np.shape(UList[0])[1]      # Number of component of the U vectors (2 because 2D)
N_T = len(UList)                 # Number of time steps


D = np.zeros((N_P * N, N_T))

for i in range(N_T):
    tmp = UList[i]
    d_i = np.concatenate([tmp[:, 0], tmp[:, 1]], axis=0)
    D[:, i] = d_i

#%% Weight computation

# Compute area
a_dataSet = gridData.compute_cell_sizes()
area = a_dataSet['Area']

# Compute weights
areaTot = np.sum(area)
weights = area/areaTot # sum should be equal to 1

# Duplicate the weights to match the length of the D columns
weights_for_D = np.concatenate((weights,weights))

print('Close the window to continue ...')
#Plot the area of each cell
gridData['area'] = area
pl = pv.Plotter()
pl.open_gif(folder_out_path + "/cell_area.gif")
pl.add_mesh(gridData, scalars='area',lighting=False, cmap=cmap_reversed,scalar_bar_args={'title': 'Area (m2)'}, show_edges=False)
pl.show_axes_all()           # show frame
pl.show_grid(font_size=12)
pl.camera_position =camera_position
pl.write_frame()
pl.show()

#%% # Compute the POD of the jet
#!with or without memory saving ! 

from modulo.modulo import MODULO

if N_PARTITIONS==1:
    m = MODULO(data=D, n_Modes=5, dtype='float32' ,weights =weights_for_D)
    Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_K()
else:
    # Prepare 10 partitions, see ex_04 for more details
    D = ReadData._data_processing(D=D,N_PARTITIONS=N_PARTITIONS,FOLDER_OUT='./MODULO_tmp')
    # Make sure to give the dimensions of D as input to MODULO when D is no more saved in the RAM
    m = MODULO(data=None, n_Modes=5, dtype='float32', weights =weights_for_D,N_PARTITIONS=N_PARTITIONS,N_S=N_P*N,N_T=N_T)
    Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_K()

#%%
# Plot the first nMode_toPlot modes

# Select the number of modes you want to plot
nMode_toPlot = 4

# Half the length of Phi_POD equals the length of the U and V vectors
nxny=int(len(Phi_POD[:,0])/2)

# Plot of the modes using pyvista
Phi_w_0 = np.zeros((nxny,1))  # dummy mode used only to please the plot function
print("Close to end the execution")
pl = pv.Plotter(shape=(nMode_toPlot, 2))
pl.open_gif(folder_out_path + "/modes.gif")
for i in range(nMode_toPlot):
    Phi_u_0 = (Phi_POD[:nxny, i]).reshape(-1, 1)
    Phi_v_0 = (Phi_POD[nxny:, i]).reshape(-1, 1)

    gridData['UV_POD'] = np.hstack((Phi_u_0, Phi_v_0, Phi_w_0))
    # X velocity
    pl.subplot(i, 0)
    pl.add_mesh(gridData, scalars=gridData['UV_POD'][:, 0],copy_mesh=True,show_edges=False)
    pl.show_grid(font_size=font_size)
    pl.camera_position =camera_position
    pl.add_title('POD Mode {}: U'.format(i+1), font='times', color='k', font_size=font_size)
    # Y velocity
    pl.subplot(i, 1)
    pl.add_mesh(gridData, scalars=gridData['UV_POD'][:, 1],copy_mesh=True,show_edges=False)
    pl.show_grid(font_size=font_size)
    pl.camera_position =camera_position
    pl.add_title('POD Mode {}: V'.format(i+1), font='times', color='k', font_size=font_size)
pl.write_frame()
pl.show()

# Frequency analysis
dt = 5e-05
fig, ax = plt.subplots(1, nMode_toPlot, figsize=(16, 5))
for k in range(nMode_toPlot):
    Signal = Psi_POD[:, k]
    s_h = np.abs((np.fft.fft(Signal - Signal.mean())))
    Freqs = np.fft.fftfreq(len(UList)) / dt
    ax[k].plot(Freqs, s_h, '-', linewidth=1.5)  # * 0.001 / 100
    ax[k].set_xlim(-0.001, 5000)
    ax[k].set_xlabel('$f$ (Hz)', fontsize=18)
    ax[k].set_ylabel('$\Psi_{}$'.format(k + 1))