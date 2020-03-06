import urllib.request
from zipfile import ZipFile
import progressbar
import os.path

class MyProgressBar():
    # progress bar from: https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def download_data(url, destination, force = False):
    # check if it has been downloaded before
    isdir = os.path.isdir(destination)
    if not isdir | force: # if it does not exist or you force it, download it
        zip_file = destination+'.zip'
        print('Downloading Data for'+'destination')
        urllib.request.urlretrieve(url, zip_file,MyProgressBar())
        print('Download Completed! I prepare data Folder')
        # Unzip the file
        zf = ZipFile(zip_file,'r'); zf.extractall('./'); zf.close()
    else:
        print('Seems ' +destination +' was already downloaded...')
        print('If you want to download, add force = True in the arguments')