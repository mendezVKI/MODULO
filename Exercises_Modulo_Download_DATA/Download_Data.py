# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:11:03 2020

@author: mendez
"""

# This is a script to download the data for the exercises 3,4,5.
#  Make sure you have installed request
#  
# The data is available at https://osf.io/5na4h/
# You can access the data as follows


#%% Exercise 1
import urllib.request
print('Downloading Data for Ex 1...')
url = 'https://osf.io/zqrp5/download'
urllib.request.urlretrieve(url, 'Ex_1_1D_Analytical.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_1_1D_Analytical.zip'
zf = ZipFile(String,'r')
zf.extractall('Ex_1_1D_Analytical')
zf.close()
print('Done!')



#%% Exercise 2
import urllib.request
print('Downloading Data for Ex 2...')
url = 'https://osf.io/jhfmn/download'
urllib.request.urlretrieve(url, 'Ex_2_Synthetic.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_2_Synthetic.zip'
zf = ZipFile(String,'r')
zf.extractall('Ex_2_Synthetic')
zf.close()
print('Done!')


#%% Exercise 3
import urllib.request
print('Downloading Data for Ex 3...')
url = 'https://osf.io/zgujk/download'
urllib.request.urlretrieve(url, 'Ex_3_CFD_Vortices.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_3_CFD_Vortices.zip'
zf = ZipFile(String,'r')
zf.extractall('Ex_3_CFD_Vortices')
zf.close()
print('Done!')

#%% Exercise 4
import urllib.request
print('Downloading Data for Ex 4...')
url = 'https://osf.io/c28de/download'
urllib.request.urlretrieve(url, 'Ex_4_TR_PIV_Jet.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_4_TR_PIV_Jet.zip'
zf = ZipFile(String,'r')
zf.extractall('Ex_4_TR_PIV_Jet')
zf.close()
print('Done!')


#%% Exercise 5
import urllib.request
print('Downloading Data for Ex 5...')
url = 'https://osf.io/47ftd/download'
urllib.request.urlretrieve(url, 'Ex_5_TR_PIV_Cylinder.zip')
print('Download Completed! I prepare data Folder')
# Unzip the file 
from zipfile import ZipFile
String='Ex_5_TR_PIV_Cylinder.zip'
zf = ZipFile(String,'r')
zf.extractall('Ex_5_TR_PIV_Cylinder')
zf.close()
print('Done!')






