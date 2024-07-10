import unittest
import numpy as np
import os, shutil
import urllib.request
from zipfile import ZipFile

from modulo_vki.core._k_matrix import CorrelationMatrix
from modulo_vki.core._pod_time import Temporal_basis_POD

from modulo_vki.core._pod_space import Spatial_basis_POD
from modulo_vki.utils.read_db import ReadData # to read the data

class TestPhi(unittest.TestCase):

    def setUp(self):
        # %% Download the data
        # --- same as for tutorial 1 ---
        # Extract all the Zip Files
        FOLDER = 'Tutorial_1_2D_Cylinder_CFD_POD_DMD'
        print('Downloading Data...')
        url = 'https://osf.io/emgv2/download'
        urllib.request.urlretrieve(url, 'Ex_7_2D_CFD.zip')
        print('Download Completed! I prepare data Folder')
        # Unzip the file
        String = 'Ex_7_2D_CFD.zip'
        zf = ZipFile(String, 'r')
        zf.extractall('./')
        zf.close()
        os.rename('Ex_7_2D_Cylinder_CFD', FOLDER)  # rename the data flolder to FOLDER
        os.remove(String)  # Delete the zip file with the data
        print('Data set unzipped and ready ! ')

        # Read one snapshot and plot it
        U = np.loadtxt(FOLDER + os.sep + 'U_Cyl.txt')  # U component
        V = np.loadtxt(FOLDER + os.sep + 'V_Cyl.txt')  # V component
        X = np.loadtxt(FOLDER + os.sep + 'X_Cyl.txt')  # X coordinates
        Y = np.loadtxt(FOLDER + os.sep + 'Y_Cyl.txt')  # Y coordinates

        # We rebuild the mesh
        n_x = len(Y)
        n_y = len(X)
        nxny = n_x * n_y
        n_s = 2 * nxny
        n_t = np.shape(U)[1]

        # Crete the snapshot Matrix:
        D = np.zeros((n_s, n_t))

        for k in range(0, n_t):
            D[:int(n_s / 2), k] = U[:, k]
            D[int(n_s / 2):, k] = V[:, k]

        D = np.nan_to_num(D)
        self.D = D
        self.N_T = n_t

        self.FOLDER_OUT = './MODULO_tmp'
        self.N_PARTITIONS = 10
        self.n_Modes = 5

        print('Computing correlation matrix...')
        K = CorrelationMatrix(self.N_T, D=D)
        print(np.shape(K))
        print("Computing Temporal Basis...")
        self.Psi_P, self.Sigma_P = Temporal_basis_POD(K,n_Modes=self.n_Modes)

        # Assert tolerance
        self.rtol = 1e-4
        self.atol = 1e-4


    def tearDown(self):
        if os.path.isdir("Tutorial_1_2D_Cylinder_CFD_POD_DMD"):
            shutil.rmtree("Tutorial_1_2D_Cylinder_CFD_POD_DMD")


    def test_phi_pod(self):
        """ Test the computation of the spatial structures for POD"""
        # Method 1
        Phi_P_mat = Spatial_basis_POD(self.D, N_T=self.N_T, PSI_P=self.Psi_P, Sigma_P=self.Sigma_P,
                                  MEMORY_SAVING=False,rescale=False)
        # Method 2
        Phi_P_norm = Spatial_basis_POD(self.D, N_T=self.N_T, PSI_P=self.Psi_P, Sigma_P=self.Sigma_P,
                                  MEMORY_SAVING=False,rescale=True)
        # Method 1 = Method2
        np.testing.assert_allclose(Phi_P_mat,Phi_P_norm ,rtol=self.rtol, atol=self.atol)

        # Check if w/ memory saving the same Phi are computed
        print('Computing correlation matrix...')
        _ = ReadData._data_processing(D=self.D,N_PARTITIONS=self.N_PARTITIONS, MR=False,
                                      FOLDER_OUT=self.FOLDER_OUT)
        _ = CorrelationMatrix(np.shape(self.D)[1], N_PARTITIONS=self.N_PARTITIONS,MEMORY_SAVING=True, FOLDER_OUT=self.FOLDER_OUT) #return None
        K = np.load(self.FOLDER_OUT + '/correlation_matrix/k_matrix.npz')['K']
        Psi_P, Sigma_P = Temporal_basis_POD(K,n_Modes=self.n_Modes)

        Phi_P_memorySaving  = Spatial_basis_POD(np.array([1]), N_T=self.N_T,
                             PSI_P=Psi_P,
                             Sigma_P=Sigma_P,
                             MEMORY_SAVING=True,
                             FOLDER_OUT=self.FOLDER_OUT,
                             N_PARTITIONS=self.N_PARTITIONS)

        np.testing.assert_allclose(Phi_P_mat,Phi_P_memorySaving ,rtol=self.rtol, atol=self.atol)




if __name__ == "__main__":
    unittest.main(verbosity=2)