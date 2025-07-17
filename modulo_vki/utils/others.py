# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 20:33:42 2019
@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt


def Plot_Field_TEXT_JET(File):
    """
    This function plots the vector field from the TR-PIV in Exercise 4.

     :param File: Name of the file to load

    """
    # We first perform a zero padding
    Name = File
    # Read data from a file
    DATA = np.genfromtxt(Name)  # Here we have the four colums
    Dat = DATA[1:, :]  # Here we remove the first raw, containing the header
    nxny = Dat.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
    n_s = 2 * nxny
    ## 1. Reconstruct Mesh from file
    X_S = Dat[:, 0]
    Y_S = Dat[:, 1]
    # Reshape also the velocity components
    V_X = Dat[:, 2]  # U component
    V_Y = Dat[:, 3]  # V component
    # Put both components as fields in the grid
    Xg, Yg, Vxg, Vyg, Magn = Plot_Field_JET(X_S, Y_S, V_X, V_Y, False, 2, 0.6)
    # Show this particular step
    fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
    # Or you can plot it as streamlines
    plt.contourf(Xg, Yg, Magn)
    # One possibility is to use quiver
    STEPx = 2
    STEPy = 2
    plt.quiver(Xg[::STEPx, ::STEPy], Yg[::STEPx, ::STEPy], \
               Vxg[::STEPx, ::STEPy], -Vyg[::STEPx, ::STEPy], color='k')  # Create a quiver (arrows) plot

    plt.rc('text', usetex=True)  # This is Miguel's customization
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    ax.set_aspect('equal')  # Set equal aspect ratio
    ax.set_xlabel('$x[mm]$', fontsize=16)
    ax.set_ylabel('$y[mm]$', fontsize=16)
    ax.set_title('Velocity Field via TR-PIV', fontsize=18)
    ax.set_xticks(np.arange(0, 40, 10))
    ax.set_yticks(np.arange(10, 30, 5))
    ax.set_xlim([0, 35])
    ax.set_ylim(10, 29)
    ax.invert_yaxis()  # Invert Axis for plotting purpose
    plt.show()
    Name[len(Name) - 12:len(Name)] + ' Plotted'
    return n_s, Xg, Yg, Vxg, -Vyg, X_S, Y_S


def Plot_Field_JET(X_S, Y_S, V_X, V_Y, PLOT, Step, Scale):
    # Number of n_X/n_Y from forward differences
    GRAD_X = np.diff(X_S)
    # GRAD_Y=np.diff(Y_S);
    # Depending on the reshaping performed, one of the two will start with
    # non-zero gradient. The other will have zero gradient only on the change.
    IND_X = np.where(GRAD_X != 0);
    DAT = IND_X[0];
    n_y = DAT[0] + 1;
    nxny = X_S.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
    # n_s=2*nxny
    # Reshaping the grid from the data
    n_x = (nxny // (n_y))  # Carefull with integer and float!
    Xg = np.transpose(X_S.reshape((n_x, n_y)))
    Yg = np.transpose(Y_S.reshape((n_x, n_y)))  # This is now the mesh! 60x114.
    Mod = np.sqrt(V_X ** 2 + V_Y ** 2)
    Vxg = np.transpose(V_X.reshape((n_x, n_y)))
    Vyg = np.transpose(V_Y.reshape((n_x, n_y)))
    Magn = np.transpose(Mod.reshape((n_x, n_y)))
    if PLOT:
        # Show this particular step
        #    fig, ax = plt.subplots(figsize=(8, 5)) # This creates the figure
        # Or you can plot it as streamlines
        #fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
        #ax.contourf(Xg, Yg, Magn)
        plt.contourf(Xg,Yg,Magn) 
        # One possibility is to use quiver
        STEPx = Step
        STEPy = Step

        plt.quiver(Xg[::STEPx, ::STEPy], Yg[::STEPx, ::STEPy], \
                   Vxg[::STEPx, ::STEPy], -Vyg[::STEPx, ::STEPy], color='k',
                   scale=Scale)  # Create a quiver (arrows) plot

        plt.rc('text', usetex=True)  # This is Miguel's customization
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
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

    return Xg, Yg, Vxg, Vyg, Magn


# Define the function to produce a gif file
def Animation_JET(Giff_NAME,D,X_S,Y_S,In,Fin,Step):
    """
    The gif file is created from the provided data snapshot
    """
    n_t=Fin-In    
    GRAD_X = np.diff(X_S)
    IND_X = np.where(GRAD_X != 0);
    DAT = IND_X[0];
    n_y = DAT[0] + 1;
    nxny = X_S.shape[0]  # is the to be doubled at the end we will have n_s=2 * n_x * n_y
    # n_s=2*nxny
    # Reshaping the grid from the data
    n_x = (nxny // (n_y))  # Carefull with integer and float!
    Xg = np.transpose(X_S.reshape((n_x, n_y)))
    Yg = np.transpose(Y_S.reshape((n_x, n_y)))  # This is now the mesh! 60x114.    n_y, n_x = Yg.shape;
    nxny = n_x * n_y
    import os
    # Create a temporary file to store the images in the GIF
    Fol_Out = 'Gif_Images_temporary'
    if not os.path.exists(Fol_Out):
        os.mkdir(Fol_Out)

    # Prepare Animation of the analytical solution
    for k in range(1,n_t,Step):
        Dat = D[:, k+In]
        V_X_m = Dat[0:nxny]
        V_Y_m = Dat[nxny::]
        # Put both components as fields in the grid
        Mod = np.sqrt(V_X_m ** 2 + V_Y_m ** 2)
        Vxg = np.transpose(V_X_m.reshape((n_x, n_y)))
        Vyg = np.transpose(V_Y_m.reshape((n_x, n_y)))
        Magn = np.transpose(Mod.reshape((n_x, n_y)))
        fig, ax = plt.subplots(figsize=(8, 5))  # This creates the figure
        # Or you can plot it as streamlines
        plt.contourf(Xg, Yg, Magn)
        # One possibility is to use quiver
        STEPx = 2
        STEPy = 2
        plt.quiver(Xg[::STEPx, ::STEPy], Yg[::STEPx, ::STEPy], \
                   Vxg[::STEPx, ::STEPy], -Vyg[::STEPx, ::STEPy], color='k')  # Create a quiver (arrows) plot

        plt.rc('text', usetex=True)  # This is Miguel's customization
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        ax.set_aspect('equal')  # Set equal aspect ratio
        ax.set_xlabel('$x[mm]$', fontsize=16)
        ax.set_ylabel('$y[mm]$', fontsize=16)
        #ax.set_title('Velocity Field via TR-PIV', fontsize=18)
        ax.set_xticks(np.arange(0, 40, 10))
        ax.set_yticks(np.arange(10, 30, 5))
        ax.set_xlim([0, 35])
        ax.set_ylim(10, 29)
        plt.clim(0, 6)
        ax.invert_yaxis()  # Invert Axis for plotting purpose
        #plt.show()

        NameOUT = Fol_Out + os.sep + 'Im%03d' % (k) + '.png'
        plt.savefig(NameOUT, dpi=100)
        plt.close(fig)
        print('Image n ' + str(k) + ' of ' + str(n_t))

    # Assembly the GIF
    import imageio  # This used for the animation

    images = []

    for k in range(1,n_t,Step):
        MEX = 'Preparing Im ' + str(k)
        print(MEX)
        NameOUT = Fol_Out + os.sep + 'Im%03d' % (k) + '.png'
        images.append(imageio.imread(NameOUT))

    # Now we can assembly the video and clean the folder of png's (optional)
    imageio.mimsave(Giff_NAME, images, duration=0.05)
    import shutil  # nice and powerfull tool to delete a folder and its content

    shutil.rmtree(Fol_Out)
    return 'Gif Created'




def Plot_2D_CFD_Cyl(Xg,Yg,U,V,k=10,CL=16,Name='', verbose=False):
    # Make a 2D plot of the 2D cylinder test case in Openfoam.
    n_x,n_y=np.shape(Xg)
    U_g=U[:,k].reshape(n_y,n_x).T
    V_g=V[:,k].reshape(n_y,n_x).T
    # Prepare the plot        
    fig, ax = plt.subplots(figsize=(6, 3)) # This creates the figure
    contour = plt.contourf(Xg,Yg,np.sqrt(U_g**2+V_g**2),30)
    # plt.quiver(Xg,Yg,U_g,V_g,scale=10000)
    ax.set_aspect('equal') # Set equal aspect ratio
    ax.set_xlabel('$x[mm]$',fontsize=13)
    ax.set_ylabel('$y[mm]$',fontsize=13)
    #ax.set_title('Tutorial 2: Cylinder Wake',fontsize=12)
    ax.set_xticks(np.arange(-0.1,0.2,0.05))
    ax.set_yticks(np.arange(-0.1,0.1,0.05))
    ax.set_xlim([-0.05,0.2])
    ax.set_ylim(-0.05,0.05)
    
    if CL !=0:
     plt.clim(0, CL)

    circle = plt.Circle((0,0),0.0075,fill=True,color='r',edgecolor='k',alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    plt.tight_layout()
              
    if len(Name) !=0:
        plt.savefig(Name, dpi=200)
        plt.close(fig)
        
        if verbose:
            print('Image exported')
    
    return 


def Animation_2D_CFD_Cyl(Giff_NAME,D,Xg,Yg,In,Fin,Step,verbose=False):
    """
    The gif file is created from the provided data snapshot
    """
    n_t=Fin-In 
    n_x,n_y=np.shape(Xg); nxny=n_x*n_y
    U = D[0:nxny,:]
    V = D[nxny::,]
    
    import os
    # Create a temporary file to store the images in the GIF
    Fol_Out = 'Gif_Images_temporary'
    if not os.path.exists(Fol_Out):
        os.mkdir(Fol_Out)
    # Loop to produce the Gifs
    if not verbose:
        print('Exporting images...')
    for k in range(1,n_t,Step):
     NameOUT = Fol_Out + os.sep + 'Im%03d' % (k) + '.png'
     Plot_2D_CFD_Cyl(Xg,Yg,U,V,k=k+In,CL=16,Name=NameOUT,verbose=verbose)

    import imageio  # This used for the animation
    images = []

    if not verbose:
        print('Preparing images...')
    for k in range(1,n_t,Step):
        if verbose:
            MEX = 'Preparing Im ' + str(k)
            print(MEX)
        NameOUT = Fol_Out + os.sep + 'Im%03d' % (k) + '.png'
        images.append(imageio.imread(NameOUT))

    # Now we can assembly the video and clean the folder of png's (optional)
    imageio.mimsave(Giff_NAME, images, duration=0.05)
    import shutil  # nice and powerfull tool to delete a folder and its content

    shutil.rmtree(Fol_Out)
    return 'Gif Created'



def Plot_Field_TEXT_Cylinder(File,Name_Mesh,Name_FIG, show=False):  
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
   fig.colorbar(CL,pad=0.05,fraction=0.025)
   ax.set_aspect('equal') # Set equal aspect ratio
   ax.set_xlabel('$x[mm]$',fontsize=13)
   ax.set_ylabel('$y[mm]$',fontsize=13)
   #ax.set_title('Tutorial 2: Cylinder Wake',fontsize=12)
   ax.set_xticks(np.arange(0,70,10))
   ax.set_yticks(np.arange(-10,11,10))
   ax.set_xlim([0,50])
   ax.set_ylim(-10,10)
   circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
   plt.gcf().gca().add_artist(circle)
   plt.tight_layout()   
   plt.savefig(Name_FIG, dpi=200)
   
   if show: 
    plt.show()
   
   plt.close()
   print(Name_FIG+' printed')
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
    # One possibility is to use quiver
    STEPx=Step;  STEPy=Step
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
    fig.colorbar(CL,pad=0.05,fraction=0.025)
    ax.set_aspect('equal') # Set equal aspect ratio
    ax.set_xlabel('$x[mm]$',fontsize=13)
    ax.set_ylabel('$y[mm]$',fontsize=13)
    #ax.set_title('Tutorial 2: Cylinder Wake',fontsize=12)
    ax.set_xticks(np.arange(0,70,10))
    ax.set_yticks(np.arange(-10,11,10))
    ax.set_xlim([0,50])
    ax.set_ylim(-10,10)
    circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    plt.tight_layout()
    plt.show()
       
   return Xg,Yg,Vxg,Vyg,Magn  
   
def Plot_Scalar_Field_Cylinder(X_S,Y_S,V_X,V_Y,Scalar,PLOT,Step,Scale):
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
   Vxg=np.transpose(V_X.reshape((n_x,n_y)))
   Vyg=np.transpose(V_Y.reshape((n_x,n_y)))
   Magn=np.transpose(Scalar.reshape((n_x,n_y)))
  
   if PLOT:
    # One possibility is to use quiver
    STEPx=Step;  STEPy=Step
    fig, ax = plt.subplots(figsize=(5, 3)) # This creates the figure
    # Or you can plot it as streamlines
    CL=plt.contourf(Xg,Yg,Magn*100,levels=np.arange(0,70,10))
    # One possibility is to use quiver
    STEPx=1;  STEPy=1
    plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
               Vxg[::STEPx,::STEPy],Vyg[::STEPx,::STEPy],color='k')
    plt.rc('text', usetex=True)      
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    fig.colorbar(CL,pad=0.05,fraction=0.025)
    ax.set_aspect('equal') # Set equal aspect ratio
    ax.set_xlabel('$x[mm]$',fontsize=13)
    ax.set_ylabel('$y[mm]$',fontsize=13)
    #ax.set_title('Tutorial 2: Cylinder Wake',fontsize=12)
    ax.set_xticks(np.arange(0,70,10))
    ax.set_yticks(np.arange(-10,11,10))
    ax.set_xlim([0,50])
    ax.set_ylim(-10,10)
    circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    plt.tight_layout()
    plt.show()
       
   return Xg,Yg,Vxg,Vyg,Magn  



def plot_grid_cylinder_flow(Xg,Yg,Vxg,Vyg):
    STEPx=1;  STEPy=1
    # This creates the figure
    fig, ax = plt.subplots(figsize=(6, 3)) 
    Magn=np.sqrt(Vxg**2+Vyg**2)
    # Plot Contour
    #CL=plt.contourf(Xg,Yg,Magn,levels=np.linspace(0,np.max(Magn),5))
    CL=plt.contourf(Xg,Yg,Magn,20,cmap='viridis',alpha=0.95)
    # One possibility is to use quiver
    STEPx=1;  STEPy=1
    plt.quiver(Xg[::STEPx,::STEPy],Yg[::STEPx,::STEPy],\
            Vxg[::STEPx,::STEPy],Vyg[::STEPx,::STEPy],color='k')
    plt.rc('text', usetex=True)      
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)
    #fig.colorbar(CL,pad=0.05,fraction=0.025)
    ax.set_aspect('equal') # Set equal aspect ratio
    ax.set_xlabel('$x[mm]$',fontsize=13)
    ax.set_ylabel('$y[mm]$',fontsize=13)
    #ax.set_title('Tutorial 2: Cylinder Wake',fontsize=12)
    ax.set_xticks(np.arange(0,70,10))
    ax.set_yticks(np.arange(-10,11,10))
    ax.set_xlim([0,50])
    ax.set_ylim(-10,10)
    circle = plt.Circle((0,0),2.5,fill=True,color='r',edgecolor='k',alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    plt.tight_layout()
    
    return fig, ax



