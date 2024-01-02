# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:20:59 2023

@author: loren, mendez
"""


import numpy as np
import matplotlib.pyplot as plt 
import os 


# We use the files in the current directory

current_directory = os.getcwd() # current directory
os.chdir(current_directory)    
print(current_directory)


# go one level back
os.chdir(os.path.join(current_directory, os.pardir))


from modulo.utils.others import Animation_2D_CFD_Cyl
from modulo.utils.others import Plot_2D_CFD_Cyl
from modulo.modulo import MODULO

# Then come back to this folder:
os.chdir(current_directory)    


# In this excercise we consider a very simple yet classic problem: the vortex
# shedding behind a 2D cylinder. 

# The dataset contains the velocity component and the grid information in 4 txt
# files. Some key information: the cylinder has a diameter of 15mm, in an overly
# large domain of 300 x 600 mm. The simulations were carried out in Openfoam
# then exported in a regulard grid.


# The inlet velocity is 10m/s, with a TI of 5%.
# The sampling frequency in the data is Fs=100 Hz.

# It is important to note that the dataset contains NANs in the location of the 
# cylinder. These can be safely ignored using numpy's nan_to_num, which 
# replaces the Nans with zeros.



# This is for plot customization
fontsize = 16
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['font.size'] = fontsize


#%% Download the data
 
# Extract all the Zip Files
FOLDER = 'Tutorial_3_kPOD_vs_POD_Cyl'

# First we unzip the file 
import urllib.request
print('Downloading Data for Ex 3...')
url = 'https://osf.io/emgv2/download'
urllib.request.urlretrieve(url, 'Ex_7_2D_CFD.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_7_2D_CFD.zip'
zf = ZipFile(String,'r')
zf.extractall('./')
zf.close()   
os.rename('Ex_7_2D_Cylinder_CFD', FOLDER) # rename the data flolder to FOLDER
os.remove(String) # Delete the zip file with the data 
print('Data set unzipped and ready ! ')

# Read one snapshot and plot it 
U=np.loadtxt(FOLDER + os.sep+ 'U_Cyl.txt')                              # U component
V=np.loadtxt(FOLDER + os.sep+ 'V_Cyl.txt')                              # V component
X=np.loadtxt(FOLDER + os.sep+ 'X_Cyl.txt')                            # X coordinates
Y=np.loadtxt(FOLDER + os.sep+ 'Y_Cyl.txt')                            # Y coordinates

# We rebuild the mesh
Xg,Yg=np.meshgrid(X,Y)
n_x=len(Y); n_y=len(X); nxny=n_x*n_y
n_s=2*nxny; n_t=np.shape(U)[1]

t=np.arange(0,n_t)*1/1000

# Crete the snapshot Matrix:
D = np.zeros((n_s, n_t))

for k in range(0, n_t):
    D[:int(n_s/2), k] = U[:, k]
    D[int(n_s/2):, k] = V[:, k]


# We can plot/export one of the snapshot
Plot_2D_CFD_Cyl(Xg,Yg,U,V,k=11,
                Name=FOLDER+os.sep+'Snapshot_11_Test.png')
# And we can make an animation from snapshot 1 to 100 in steps of 1:
Name_GIF=FOLDER+os.sep+'Animation_TEST.gif'
Animation_2D_CFD_Cyl(Name_GIF,D,Xg,Yg,1,100,1)

# Prepare the folder with the output from the modal decompositions

decompositions = ['kPOD', 'POD']

for decomp in decompositions:
    os.makedirs(FOLDER+ os.sep + f'{decomp}_Results_Cylinder_CFD', 
                exist_ok=True)

# %% 


# --- Initialize MODULO object
m = MODULO(data=np.nan_to_num(D), svd_solver='svd_scipy_sparse')

# %% 


FOLDER_POD_RESULTS=FOLDER+os.sep+'POD_Results_Cylinder_CFD'
if not os.path.exists(FOLDER_POD_RESULTS):
    os.mkdir(FOLDER_POD_RESULTS)


Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_K()

# We here plot the POD modes and their structures

U_D=Phi_POD[0:nxny,:]
V_D=Phi_POD[nxny::,:]

for K in range(6):
 Name=FOLDER_POD_RESULTS+os.sep+'POD_Mode_'+str(K)+'.png'
 plt.title('$\Phi_{\mathcal{P}}(\mathbf{x}_i)$',fontsize=18)
 Plot_2D_CFD_Cyl(Xg,Yg,U_D,V_D,k=K,CL=0,Name=Name)

# Then plot their temporal evolution

for K in range(6):
  fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
  plt.plot(t,Psi_POD[:,K])
  plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
  Name=FOLDER_POD_RESULTS+os.sep+'POD_Mode_PSI_'+str(K)+'.png' 
  print(Name+' Saved')
  plt.savefig(Name, dpi=300)  


# Plot the sigma POD
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(Sigma_POD/Sigma_POD[0],'ko') 
ax.set_xlabel('$r$',fontsize=16)
ax.set_ylabel('$\sigma_{\mathcal{P}}$',fontsize=16)
plt.tight_layout()
Name=FOLDER_POD_RESULTS+os.sep+'Sigma_P.png'
plt.show()
plt.savefig(Name, dpi=300) 


# Here is the approximation with the leading 3 POD modes
R=3
D_P=np.real(np.linalg.multi_dot([Phi_POD[:,0:R],
                                 np.diag(Sigma_POD[0:R]),
                                 Psi_POD[:,0:R].T]) )
Error=np.linalg.norm(m.D-D_P)/np.linalg.norm(m.D)


Name_GIF=FOLDER_POD_RESULTS+os.sep+'Animation_Approximation.gif'   
Animation_2D_CFD_Cyl(Name_GIF,D_P,Xg,Yg,1,100,1)

# %% 
FOLDER_kPOD_RESULTS=FOLDER+os.sep+'kPOD_Results_Cylinder_CFD'
# -- Now we proceed with the new kernel POD
Phi_kPOD, Psi_kPOD, Sigma_kPOD = m.compute_kPOD(M_DIST=[1,19],k_m=1e-12,cent=None)

# Game: start with k=0.1 then decrease it to almost 0


# We here plot the POD modes and their structures

U_D=Phi_kPOD[0:nxny,:]
V_D=Phi_kPOD[nxny::,:]

for K in range(6):
  Name=FOLDER_kPOD_RESULTS+os.sep+'kPOD_Mode_'+str(K)+'.png'
  plt.title('$\Phi_{\mathcal{K}}(\mathbf{x}_i)$',fontsize=18)
  Plot_2D_CFD_Cyl(Xg,Yg,U_D,V_D,k=K,CL=0,Name=Name)

# Then plot their temporal evolution

for K in range(6):
  fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
  plt.plot(t,Psi_kPOD[:,K])
  plt.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)
  Name=FOLDER_kPOD_RESULTS+os.sep+'kPOD_Mode_PSI_'+str(K)+'.png' 
  print(Name+' Saved')
  plt.savefig(Name, dpi=300)  


# Plot the sigma kPOD
fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
plt.plot(Sigma_kPOD/Sigma_kPOD[0],'ko') 
ax.set_xlabel('$r$',fontsize=16)
ax.set_ylabel('$\sigma_{\mathcal{K}}$',fontsize=16)
plt.tight_layout()
Name=FOLDER_kPOD_RESULTS+os.sep+'Sigma_P.png'
plt.show()
plt.savefig(Name, dpi=300) 


# Here is the approximation with the leading 3 POD modes
R=3
D_kP=np.real(np.linalg.multi_dot([Phi_kPOD[:,0:R],
                                 np.diag(Sigma_kPOD[0:R]),
                                 Psi_kPOD[:,0:R].T]) )
error_kpod=np.linalg.norm(m.D-D_kP)/np.linalg.norm(m.D)

print('Error kPOD = {} (R={})'.format(error_kpod, R))
Name_GIF=FOLDER_kPOD_RESULTS+os.sep+'Animation_Approximation.gif'   
Animation_2D_CFD_Cyl(Name_GIF,D_kP,Xg,Yg,1,100,1)


# %% 

fig, axs = plt.subplots(2, 4)

for i in range(4):
    
    Signal_kpod=Psi_kPOD[:, i]
    s_h_kpod=np.abs((np.fft.fft(Signal_kpod-Signal_kpod.mean())))
    Freqs_kpod=np.fft.fftfreq(313)*(1/0.001)

    Signal_pod=Psi_POD[:, i]
    s_h_pod=np.abs((np.fft.fft(Signal_pod-Signal_pod.mean())))
    Freqs_pod=np.fft.fftfreq(313)*(1/0.001)
    
    
    axs[0, i].plot(Freqs_pod, s_h_pod)
    axs[1, i].plot(Freqs_kpod, s_h_kpod)
    
    if i == 0: 
        axs[0, i].set(ylabel='$\widehat{\psi}_{\mathrm{POD}}$')
        axs[1, i].set(ylabel='$\widehat{\psi}_{\mathrm{kPOD}}$')
        
    axs[0, i].set(title='Mode {}'.format(int(i + 1)))
    axs[0, i].set(xlabel='$f$ (Hz)')
    axs[1, i].set(xlabel='$f$ (Hz)')

fig.tight_layout()
plt.show()

# %% 

# Modes comparison for different decompositions
def get_psi_plot_cylinder(Xg,Yg,U,V,k=10,CL=16,Name='', ax=None):
    n_x,n_y=np.shape(Xg)
    U_g=U[:,k].reshape(n_y,n_x).T
    V_g=V[:,k].reshape(n_y,n_x).T
    
    ax.contourf(Xg,Yg,np.sqrt(U_g**2+V_g**2),30)
    # plt.quiver(Xg,Yg,U_g,V_g,scale=10000)
    ax.set_aspect('equal') # Set equal aspect ratio
    ax.set_xlabel('$x[mm]$',fontsize=13)
    ax.set_ylabel('$y[mm]$',fontsize=13)
    #ax.set_title('Tutorial 2: Cylinder Wake',fontsize=12)
    ax.set_xticks(np.arange(-0.1,0.2,0.05))
    ax.set_yticks(np.arange(-0.1,0.1,0.05))
    ax.set_xlim([-0.05,0.2])
    ax.set_ylim(-0.05,0.05)
    
    circle = plt.Circle((0,0),0.0075,fill=True,color='r',edgecolor='k',alpha=0.5)
    ax.add_patch(circle)
    return 

rows, cols = 2, 3
fig, axs = plt.subplots(rows, cols, figsize=(12, 8))

U_POD=Phi_POD[0:nxny,:]
V_POD=Phi_POD[nxny::,:]

U_kPOD=Phi_kPOD[0:nxny,:]
V_kPOD=Phi_kPOD[nxny::,:]


U = [U_POD, U_kPOD] 
V = [V_POD, V_kPOD]

i = 0
# Loop through the rows and columns to populate the subplots
for row in range(rows):
    Ui, Vi = U[row], V[row]
    
    for col in range(cols):
        #plt.sca(axs[row, col])  # Set the current subplot
        get_psi_plot_cylinder(Xg,Yg,Ui,Vi,k=col,CL=0,Name=Name, ax=axs[row, col])  # Call your function to generate a plot
        #axs[row, col].axis('on')
        
    if i == 0: 
        axs[0, i].set(ylabel='$y$ (mm)')
        axs[1, i].set(ylabel='$y$ (mm)')
        
    axs[0, i].set(title='$\phi_\mathcal{P}$' + '{}'.format(int(i + 1)))
    axs[1, i].set(title='$\phi_\mathcal{K}$' + '{}'.format(int(i + 1)))
    axs[0, i].set(xlabel='$x$ (mm)')
    axs[1, i].set(xlabel='$x$ (mm)')
    
    i+=1 
    
fig.tight_layout()
plt.show()


#%% Conclusion
# the more we put k_m to be small, 
# the more we are stressing that the similarity between the selected snapshot
# (in this case 1 and 19) is small.



