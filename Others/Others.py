# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 20:33:42 2019

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt

def Plot_Field_TEXT_Cylinder(File,Name_Mesh):  
   """
   This function plots the vector field from the TR-PIV in Exercise 4.
      
    :param File: Name of the file to load          
   
   """
   # We first perform a zero padding
   Name=File
   # Read data from a file
   DATA = np.genfromtxt(Name) # Here we have the four colums
   Dat=DATA[1:,:] # Here we remove the first raw, containing the header
   nxny=Dat.shape[0] # is the to be doubled at the end we will have n_s=2 * n_x * n_y
   n_s=2*nxny
   ## 1. Reconstruct Mesh from file Name_Mesh
   DATA_mesh=np.genfromtxt(Name_Mesh);
   Dat_mesh=DATA_mesh[1:,:]
   X_S=Dat_mesh[:,0];
   Y_S=Dat_mesh[:,1];
   # Reshape also the velocity components
   V_X=Dat[:,0] # U component
   V_Y=Dat[:,1] # V component
   # Put both components as fields in the grid
   Xg,Yg,Vxg,Vyg,Magn=Plot_Field_Cylinder(X_S,Y_S,V_X,V_Y,False,2,0.6)
   # Show this particular step
   fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
   # Or you can plot it as streamlines
   CL=plt.contourf(Xg,Yg,Magn,levels=np.arange(0,18,2))
   # One possibility is to use quiver
   STEPx=1;  STEPy=1
   plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
               Vxg[::STEPx,::STEPy],Vyg[::STEPx,::STEPy],color='k')
   plt.rc('text', usetex=True)      
   plt.rc('font', family='serif')
   plt.rc('xtick',labelsize=12)
   plt.rc('ytick',labelsize=12)
   plt.colorbar(CL)
   ax.set_aspect('equal') # Set equal aspect ratio
   ax.set_xlabel('$x[mm]$',fontsize=13)
   ax.set_ylabel('$y[mm]$',fontsize=13)
   ax.set_title('Tutorial 2: Cylinder Wake',fontsize=16)
   ax.set_xticks(np.arange(0,70,10))
   ax.set_yticks(np.arange(-10,11,10))
   ax.set_xlim([0,50])
   ax.set_ylim(-10,10)
   circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
   plt.gcf().gca().add_artist(circle)
   plt.tight_layout()
   plt.show()
   Name[len(Name)-12:len(Name)]+' Plotted'
   return n_s, Xg, Yg, Vxg, Vyg, X_S, Y_S


def Plot_Field_Cylinder(X_S,Y_S,V_X,V_Y,PLOT,Step,Scale):
  # Number of n_X/n_Y from forward differences
   GRAD_X=np.diff(Y_S) 
   #GRAD_Y=np.diff(Y_S);
   # Depending on the reshaping performed, one of the two will start with
   # non-zero gradient. The other will have zero gradient only on the change.
   IND_X=np.where(GRAD_X!=0); DAT=IND_X[0]; n_y=DAT[0]+1;
   nxny=X_S.shape[0] # is the to be doubled at the end we will have n_s=2 * n_x * n_y
   #n_s=2*nxny
   # Reshaping the grid from the data
   n_x=(nxny//(n_y)) # Carefull with integer and float!
   Xg=np.transpose(X_S.reshape((n_x,n_y)))
   Yg=np.transpose(Y_S.reshape((n_x,n_y))) # This is now the mesh! 60x114.
   Mod=np.sqrt(V_X**2+V_Y**2)
   Vxg=np.transpose(V_X.reshape((n_x,n_y)))
   Vyg=np.transpose(V_Y.reshape((n_x,n_y)))
   Magn=np.transpose(Mod.reshape((n_x,n_y)))
  
   if PLOT:
   # Show this particular step
#    fig, ax = plt.subplots(figsize=(8, 5)) # This creates the figure
    plt.contourf(Xg,Yg,Magn)
    # One possibility is to use quiver
    STEPx=Step
    STEPy=Step
    plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
               Vxg[::STEPx,::STEPy],-Vyg[::STEPx,::STEPy],color='k',scale=Scale) # Create a quiver (arrows) plot
    
    plt.rc('text', usetex=True)      # This is Miguel's customization
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
#    ax.set_aspect('equal') # Set equal aspect ratio
#    ax.set_xlabel('$x[mm]$',fontsize=16)
#    ax.set_ylabel('$y[mm]$',fontsize=16)
#    ax.set_title('Velocity Field via TR-PIV',fontsize=18)
#    ax.set_xticks(np.arange(0,40,10))
#    ax.set_yticks(np.arange(10,30,5))
#    ax.set_xlim([0,35])
#    ax.set_ylim(10,29)
#    ax.invert_yaxis() # Invert Axis for plotting purpose
#    plt.show()
       
   return Xg,Yg,Vxg,Vyg,Magn  
   

# Define the function to produce a gif file
def Animation(npz_F,Giff_NAME):
    """
    The gif file is created from the data in the npz file
    """
    # Load the data
    #npz_F='Data.npz'
    #Giff_NAME='Exercise_4.gif'
    data = np.load(npz_F)
    D=data['D']    
    # dt=data['dt']
    n_t=data['n_t']
    Xg=data['Xg']
    Yg=data['Yg']   
    n_y,n_x=Yg.shape; nxny=n_x*n_y
    import os
    # Create a temporary file to store the images in the GIF
    Fol_Out = 'Gif_Images_temporary'
    if not os.path.exists(Fol_Out):
      os.mkdir(Fol_Out)

    
    # Prepare Animation of the analytical solution
    for k in range(0,400,1):
     Dat=D[:,k]
     V_X_m=Dat[0:nxny]
     V_Y_m=Dat[nxny::]
     # Put both components as fields in the grid
     Mod=np.sqrt(V_X_m**2+V_Y_m**2)
     Vxg=np.transpose(V_X_m.reshape((n_x,n_y)))
     Vyg=np.transpose(V_Y_m.reshape((n_x,n_y)))
     Magn=np.transpose(Mod.reshape((n_x,n_y)))
     fig, ax = plt.subplots(figsize=(8, 5)) # This creates the figure
   # Or you can plot it as streamlines
     CL=plt.contourf(Xg,Yg,Magn,vmin=0, vmax=16)
   # One possibility is to use quiver
     STEPx=2
     STEPy=2
     plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
               Vxg[::STEPx,::STEPy],-Vyg[::STEPx,::STEPy],color='k') # Create a quiver (arrows) plot
    
     plt.rc('text', usetex=True)      
     plt.rc('font', family='serif')
     plt.rc('xtick',labelsize=12)
     plt.rc('ytick',labelsize=12)
     #plt.clim(0, 16);
     #fig.colorbar(CL,pad=0.05,fraction=0.025)
     plt.clim(0, 16)
     plt.colorbar(CL)
    # fig.colorbar(CL)
     ax.set_aspect('equal') # Set equal aspect ratio
     ax.set_xlabel('$x[mm]$',fontsize=13)
     ax.set_ylabel('$y[mm]$',fontsize=13)
     ax.set_title('Exercise 5: Cylinder Wake',fontsize=16)
     ax.set_xticks(np.arange(0,70,10))
     ax.set_yticks(np.arange(-10,11,10))
     ax.set_xlim([0,50])
     ax.set_ylim(-10,10)
     circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
     plt.gcf().gca().add_artist(circle)
     plt.tight_layout()
     NameOUT=Fol_Out + os.sep + 'Im%03d' % (k) + '.png'
     plt.savefig(NameOUT, dpi=100)      
     plt.close(fig)
     print('Image n ' + str(k) + ' of ' + str(n_t))
    
    # Assembly the GIF
    import imageio  # This used for the animation

    images = []

    for k in range(0,400,1):
      MEX = 'Preparing Im ' + str(k) 
      print(MEX)    
      NameOUT=Fol_Out + os.sep + 'Im%03d' % (k) + '.png'
      images.append(imageio.imread(NameOUT))
    
    # Now we can assembly the video and clean the folder of png's (optional)
    imageio.mimsave(Giff_NAME, images, duration=0.05)
    import shutil  # nice and powerfull tool to delete a folder and its content

    shutil.rmtree(Fol_Out) 
    return 'Gif Created'
   