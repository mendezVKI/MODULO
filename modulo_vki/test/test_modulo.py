import unittest
import numpy as np
import os,shutil
import urllib.request
from zipfile import ZipFile

from modulo_vki import ModuloVKI # this is to create modulo objects
from modulo_vki.utils.read_db import ReadData # to read the data

from modulo_vki.core._mpod_time import temporal_basis_mPOD
from modulo_vki.core._k_matrix import CorrelationMatrix
from modulo_vki.utils._utils import overlap

class TestModulo(unittest.TestCase):

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

        self.n_Modes = 5

        # Assert tolerance
        self.rtol = 1e-4
        self.atol = 1e-4

        self.FOLDER_OUT = './MODULO_tmp'
        self.N_PARTITIONS = 10


    def tearDown(self):
        if os.path.isdir("Tutorial_1_2D_Cylinder_CFD_POD_DMD"):
            shutil.rmtree("Tutorial_1_2D_Cylinder_CFD_POD_DMD")
        if os.path.isdir("MODULO_tmp"):
            shutil.rmtree("MODULO_tmp")
        if os.path.isdir("POD"):
            shutil.rmtree("POD")
        if os.path.isdir("SPOD_T"):
            shutil.rmtree("SPOD_T")

    def test_init(self):
        """Test init of MODULO"""
        n_Modes = self.n_Modes

        param = {'data': self.D,
                 'N_PARTITIONS': 1,
                 'FOLDER_OUT': './',
                 'SAVE_K': False,
                 'N_T': np.shape(self.D)[1],
                 'N_S': np.shape(self.D)[0],
                 'n_Modes': self.n_Modes,
                 'dtype': 'float32',
                 'eig_solver':'eigh',
                 'svd_solver': 'svd_sklearn_truncated',
                 'weights':np.ones(np.shape(self.D)[0])*5}


        m = ModuloVKI(data=param['data'],
                 N_PARTITIONS=param['N_PARTITIONS'],
                 FOLDER_OUT=param['FOLDER_OUT'],
                 SAVE_K=param['SAVE_K'],
                 N_T=param['N_T'],
                 N_S=param['N_S'],
                 n_Modes=param['n_Modes'],
                 dtype= param['dtype'],
                 eig_solver=param['eig_solver'],
                 svd_solver= param['svd_solver'],
                 weights=param['weights'])

        np.testing.assert_allclose(param['data'].astype(param['dtype']),m.D,rtol=1e-8, atol=1e-8)
        self.assertEqual(param['N_PARTITIONS'],m.N_PARTITIONS)
        self.assertEqual(param['FOLDER_OUT'],m.FOLDER_OUT)
        self.assertEqual(False,m.MEMORY_SAVING)
        self.assertEqual(param['SAVE_K'],m.SAVE_K)
        self.assertEqual(np.shape(self.D)[1],m.N_T)
        self.assertEqual(np.shape(self.D)[0],m.N_S)
        self.assertEqual(param['n_Modes'],m.n_Modes)
        self.assertEqual(param['eig_solver'],m.eig_solver)
        self.assertEqual(param['svd_solver'],m.svd_solver)
        np.testing.assert_allclose(param['weights'],m.weights,rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(np.transpose(np.transpose(param['data'].astype(param['dtype'])) * np.sqrt(param['weights'])),m.Dstar,rtol=1e-8, atol=1e-8)

        param['N_PARTITIONS']=10
        m = ModuloVKI(data=None,
                      N_PARTITIONS=param['N_PARTITIONS'],
                      FOLDER_OUT=param['FOLDER_OUT'],
                      SAVE_K=param['SAVE_K'],
                      N_T=param['N_T'],
                      N_S=param['N_S'],
                      n_Modes=param['n_Modes'],
                      dtype=param['dtype'],
                      eig_solver=param['eig_solver'],
                      svd_solver=param['svd_solver'],
                      weights=param['weights'])

        self.assertEqual(None,m.D)
        self.assertEqual(param['N_T'],m.N_T)
        self.assertEqual(param['N_S'],m.N_S)
        self.assertEqual(True,m.MEMORY_SAVING)


        # Check if error is raised when no data is provided with only one partition
        param['N_PARTITIONS']=1
        with self.assertRaises(TypeError):
            m = ModuloVKI(data=None,N_PARTITIONS=param['N_PARTITIONS'])

        # Check if error is raised when the weights have a wrong shape
        with self.assertRaises(AttributeError):
            m = ModuloVKI(data=self.D,N_PARTITIONS=param['N_PARTITIONS'],weights=np.ones(2))

        # Check if error when calling an eigh_solver not implemented
        with self.assertRaises(NotImplementedError):
            m = ModuloVKI(data=self.D,N_PARTITIONS=param['N_PARTITIONS'], eig_solver='not_implemented')

        # Check if error when calling a svd_solver not implemented
        with self.assertRaises(NotImplementedError):
            m = ModuloVKI(data=self.D,N_PARTITIONS=param['N_PARTITIONS'], svd_solver='not_implemented')


    def test_output_shape(self):
        """Test the size of the outputs from the different decomposition"""
        n_Modes = self.n_Modes
        m = ModuloVKI(data=self.D,n_Modes=n_Modes)

        # POD K
        Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_K()

        self.assertEqual(np.shape(Phi_POD)[0], np.shape(self.D)[0])
        self.assertEqual(np.shape(Phi_POD)[1], n_Modes)
        self.assertEqual(np.shape(Psi_POD)[0], np.shape(self.D)[1])
        self.assertEqual(np.shape(Psi_POD)[1], n_Modes)
        self.assertEqual(np.shape(Sigma_POD)[0], n_Modes)

        # POD svd
        Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_svd()

        self.assertEqual(np.shape(Phi_POD)[0], np.shape(self.D)[0])
        self.assertEqual(np.shape(Phi_POD)[1], n_Modes)
        self.assertEqual(np.shape(Psi_POD)[0], np.shape(self.D)[1])
        self.assertEqual(np.shape(Psi_POD)[1], n_Modes)
        self.assertEqual(np.shape(Sigma_POD)[0], n_Modes)

        # mPOD
        Keep = np.array([1, 1, 1, 1])
        Nf = np.array([201, 201, 201, 201])
        # --- Test Case Data:
        # + Stand off distance nozzle to plate
        H = 4 / 1000
        # + Mean velocity of the jet at the outlet
        U0 = 6.5
        # + Input frequency splitting vector in dimensionless form (Strohual Number)
        ST_V = np.array([0.1, 0.2, 0.25, 0.4])
        # + Frequency Splitting Vector in Hz
        F_V = ST_V * U0 / H
        # + Size of the extension for the BC (Check Docs)
        Ex = 203  # This must be at least as Nf.
        dt = 1 / 2000;
        boundaries = 'reflective';
        MODE = 'reduced'
        # K = np.load("./MODULO_tmp/correlation_matrix/k_matrix.npz")['K']
        Phi_M, Psi_M, Sigmas_M = m.compute_mPOD(Nf, Ex, F_V, Keep, 20, boundaries, MODE, dt, False)

        self.assertEqual(np.shape(Phi_M)[0], np.shape(self.D)[0])
        self.assertEqual(np.shape(Phi_M)[1], n_Modes)
        self.assertEqual(np.shape(Psi_M)[0], np.shape(self.D)[1])
        self.assertEqual(np.shape(Psi_M)[1], n_Modes)
        self.assertEqual(np.shape(Sigmas_M)[0], n_Modes)

        # DMD pip
        Phi_D, Lambda, freqs, a0s = m.compute_DMD_PIP(False, F_S=1000)

        self.assertEqual(np.shape(Phi_D)[0], np.shape(self.D)[0])
        self.assertEqual(np.shape(Phi_D)[1], n_Modes)
        self.assertEqual(np.shape(Lambda)[0], n_Modes)
        self.assertEqual(np.shape(freqs)[0], n_Modes)
        self.assertEqual(np.shape(a0s)[0], n_Modes)

        # DFT
        Fs = 2000
        Sorted_Freqs, Phi_F, Sorted_Sigmas = m.compute_DFT(Fs)

        self.assertEqual(np.shape(Sorted_Freqs)[0], np.shape(self.D)[1])
        self.assertEqual(np.shape(Phi_F), np.shape(self.D))
        self.assertEqual(np.shape(Sorted_Sigmas)[0], np.shape(self.D)[1])

        # SPOD t
        n_Modes = 3
        L_B = 50
        O_B = 20
        Ind = np.arange(np.shape(self.D)[1])
        Indices = overlap(Ind, len_chunk=L_B, len_sep=O_B)
        Freqs = np.fft.fftfreq(np.shape(Indices)[1]) * Fs  # Compute the frequency bins
        Keep_IND = np.where(Freqs >= 0)
        Phi_SP, Sigma_SP, Freqs_Pos = m.compute_SPOD_t(F_S=Fs,  # sampling frequency
                                                       L_B=L_B,  # Length of the chunks for time average
                                                       O_B=20,  # Overlap between chunks
                                                       n_Modes=n_Modes)  # number of modes PER FREQUENCY

        self.assertEqual(np.shape(Phi_SP)[0], np.shape(self.D)[0])
        self.assertEqual(np.shape(Phi_SP)[1],n_Modes)
        self.assertEqual(np.shape(Phi_SP)[2],np.shape(Keep_IND)[1])
        self.assertEqual(np.shape(Sigma_SP)[0],n_Modes)
        self.assertEqual(np.shape(Sigma_SP)[1],np.shape(Keep_IND)[1])
        self.assertEqual(np.shape(Freqs_Pos)[0],np.shape(Keep_IND)[1])

        # SPOD s
        m = ModuloVKI(data=m.D)
        # Prepare (partition) the dataset
        # Compute the POD
        Phi_S, Psi_S, Sigma_S = m.compute_SPOD_s(Fs, N_O=100,
                                                 f_c=0.01,
                                                 n_Modes=n_Modes,
                                                 SAVE_SPOD=True)
        self.assertEqual(np.shape(Phi_S)[0], np.shape(self.D)[0])
        self.assertEqual(np.shape(Phi_S)[1], n_Modes)
        self.assertEqual(np.shape(Psi_S)[0], np.shape(self.D)[1])
        self.assertEqual(np.shape(Psi_S)[1], n_Modes)
        self.assertEqual(np.shape(Sigma_S)[0], n_Modes)

        # kPOD
        Phi_xi, Psi_xi, Sigma_xi = m.compute_kPOD(n_Modes=n_Modes)

        self.assertEqual(np.shape(Phi_xi)[0], np.shape(self.D)[0])
        self.assertEqual(np.shape(Phi_xi)[1], n_Modes)
        self.assertEqual(np.shape(Psi_xi)[0], np.shape(self.D)[1])
        self.assertEqual(np.shape(Psi_xi)[1], n_Modes)
        self.assertEqual(np.shape(Sigma_xi)[0], n_Modes)



    def test_pod_k_svd(self):
        '''Test POD K method and svd method'''
        n_Modes = self.n_Modes
        m = ModuloVKI(data=self.D,n_Modes=n_Modes)

        # Compute the POD using Sirovinch's method
        Phi_POD, Psi_POD, Sigma_POD = m.compute_POD_K()

        # Compute the POD using svd method
        Phi_POD_svd, Psi_POD_svd, Sigma_POD_svd = m.compute_POD_svd()

        np.testing.assert_allclose(np.abs(Phi_POD),np.abs(Phi_POD_svd),rtol=self.rtol, atol=self.atol)

    def test_mpod(self):
        """Test if error is raised in mPOD if Ex <Nf"""
        Keep = np.array([1, 1, 1, 1])
        Nf = np.array([201, 201, 201, 201])
        # --- Test Case Data:
        # + Stand off distance nozzle to plate
        H = 4 / 1000
        # + Mean velocity of the jet at the outlet
        U0 = 6.5
        # + Input frequency splitting vector in dimensionless form (Strohual Number)
        ST_V = np.array([0.1, 0.2, 0.25, 0.4])
        # + Frequency Splitting Vector in Hz
        F_V = ST_V * U0 / H
        # + Size of the extension for the BC (Check Docs)
        Ex = 203  # This must be at least as Nf.
        dt = 1 / 2000;
        boundaries = 'reflective';
        MODE = 'reduced'
        SAT = 20

        print('Computing correlation matrix D matrix...')
        self.K = CorrelationMatrix(self.N_T, self.FOLDER_OUT, D=self.D)

        with self.assertRaises(RuntimeError):
            PSI_M = temporal_basis_mPOD(K=self.K, Nf=Nf, Ex=100, F_V=F_V, Keep=Keep, boundaries=boundaries,
                                    MODE=MODE, dt=dt, FOLDER_OUT=self.FOLDER_OUT,
                                    n_Modes=self.n_Modes, SAT=SAT)

if __name__ == "__main__":
    unittest.main(verbosity=2)