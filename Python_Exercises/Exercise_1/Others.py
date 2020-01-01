# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 20:33:42 2019

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt


# Define the function to produce a gif file
def Animation(npz_F,Giff_NAME):
    """
    The gif file is created from the data in the npz file
    """
    # Load the data
    #npz_F='Data.npz'
    #Giff_NAME='Exercise_1.gif'
    data = np.load(npz_F)
    D=data['D']
    t=data['t']
    # dt=data['dt']
    n_t=data['n_t']
    y=data['y']
    dy=data['dy']
    n_y=data['n_y'] 

    import os
    # Create a temporary file to store the images in the GIF
    Fol_Out = 'Gif_Images_temporary'
    if not os.path.exists(Fol_Out):
      os.mkdir(Fol_Out)

    plt.ioff() # To disable interactive plotting

    # Prepare Animation of the analytical solution
    for k in range(0,934,1):
     Profile=D[:,k]
     fig, ax = plt.subplots(figsize=(8,12))
     plt.subplot(2, 1, 1)
     plt.plot(y,Profile)
     plt.xlabel('$y $',fontsize=22)
     plt.ylabel('$\hat{u}(\hat(y)) $',fontsize=22)
     plt.rc('text', usetex=True)      # This is Miguel's customization
     plt.rc('font', family='serif')
     plt.rc('xtick',labelsize=18)
     plt.rc('ytick',labelsize=18)
     # plt.title('Eigen_Function_Sol_N',fontsize=18)
     plt.xlim([-1,1])
     plt.ylim([-2,4]) 
     plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=1.0)
     # Space-Time Plot
     plt.subplot(2, 1, 2)
    # plt.pcolor(t,y,D)
    # plt.plot(np.ones(len(y))*t[k],y,'r',linewidth=1.5)
     #plt.ylabel('$y $',fontsize=22)
     #plt.xlabel('$t $',fontsize=22) 
     #plt.tight_layout()  
     # plt.show()
     plt.plot(t,D[int(np.floor(n_y/2)),:],'k--')
     plt.plot(t[k],D[int(np.floor(n_y/2)),k],'ko',markersize=9,mfc='red')
     
     plt.xlabel('t[-]',fontsize=18)
     plt.ylabel('$\hat{u}(\hat(y)=0,t) $',fontsize=18)
     plt.title('Centerline Vel Evolution',fontsize=16)
     plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=1.0)
     
     NameOUT=Fol_Out + os.sep + 'Im%03d' % (k) + '.png'
     plt.savefig(NameOUT, dpi=100)      
     plt.close(fig)
     print('Image n ' + str(k) + ' of ' + str(n_t))
    
    # Assembly the GIF
    import imageio  # This used for the animation

    images = []

    for k in range(0,934,1):
      MEX = 'Preparing Im ' + str(k) 
      print(MEX)    
      NameOUT=Fol_Out + os.sep + 'Im%03d' % (k) + '.png'
      images.append(imageio.imread(NameOUT))
    
    # Now we can assembly the video and clean the folder of png's (optional)
    imageio.mimsave(Giff_NAME, images, duration=0.02)
    import shutil  # nice and powerfull tool to delete a folder and its content

    shutil.rmtree(Fol_Out) 
    return 'Gif Created'
   