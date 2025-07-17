import unittest
import numpy as np
import copy,shutil, os
from tqdm import tqdm


from modulo_vki.utils.read_db import ReadData # to read the data

class TestReadDb(unittest.TestCase):

    def setUp(self):
        self.D = np.random.rand(100,10)
        self.D_orig = copy.deepcopy(self.D)
        self.D_MEAN = np.mean(self.D,axis=1)

        self.rtol = 1e-4
        self.atol = 1e-4

    def tearDown(self):
        if os.path.isdir("MODULO_tmp"):
            shutil.rmtree("MODULO_tmp")

    def test_meanComputation(self):
        """Test if mean is computed correctly & substracted from D"""
        N_PARTITIONS = 1
        FOLDER_OUT =  './MODULO_tmp'
        MR = True
        SAVE_D = True
        D, D_MEAN = ReadData._data_processing(D=self.D, N_PARTITIONS=N_PARTITIONS, FOLDER_OUT=FOLDER_OUT,MR=MR,SAVE_D=SAVE_D)
        # Check mean computation
        self.assertListEqual(D_MEAN.tolist(), self.D_MEAN.tolist())

        # Check if mean was removed from D
        D_MR = (self.D_orig - np.ones(np.shape(self.D)) * self.D_MEAN.reshape(-1, 1))
        np.testing.assert_allclose(D,D_MR ,rtol=self.rtol, atol=self.atol)

    def test_saveD(self):
        """Check if the saving of D went ok"""
        N_PARTITIONS = 1
        FOLDER_OUT =  './MODULO_tmp'
        MR = False
        SAVE_D = True
        D = ReadData._data_processing(D=self.D, N_PARTITIONS=N_PARTITIONS, FOLDER_OUT=FOLDER_OUT,MR=MR,SAVE_D=SAVE_D)

        D_test = np.load(FOLDER_OUT + '/data_matrix/database.npz')['D']
        np.testing.assert_allclose(D_test,self.D_orig ,rtol=self.rtol, atol=self.atol)

    def test_split(self):
        """Test decomposition of D """
        N_PARTITIONS = 5
        FOLDER_OUT =  './MODULO_tmp'
        MR = False
        _ = ReadData._data_processing(D=self.D_orig, N_PARTITIONS=N_PARTITIONS, FOLDER_OUT=FOLDER_OUT,MR=MR)
        N_T = np.shape(self.D_orig)[1]

        # Read chunks of D matrix
        if N_T % N_PARTITIONS != 0:
            tot_blocks_col = N_PARTITIONS + 1
        else:
            tot_blocks_col = N_PARTITIONS

        di_list = []
        for k in tqdm(range(tot_blocks_col)):
            di = np.load(FOLDER_OUT + f"/data_partitions/di_{k + 1}.npz")['di']
            di_list.append(di)
        D_rec = np.concatenate(di_list,axis=1)

        # Check if the decomposition went ok
        np.testing.assert_allclose(D_rec,self.D_orig ,rtol=self.rtol, atol=self.atol)

        # Check if the decomposition went ok when MR= TRUE + if the mean was removed from D
        N_PARTITIONS = 5
        FOLDER_OUT =  './MODULO_tmp'
        self.D = copy.deepcopy(self.D_orig)
        D_MEAN = ReadData._data_processing(D=self.D, N_PARTITIONS=N_PARTITIONS, FOLDER_OUT=FOLDER_OUT,MR=True)
        np.testing.assert_allclose(D_MEAN, self.D_MEAN,rtol=self.rtol, atol=self.atol)

        N_T = np.shape(self.D_orig)[1]

        # Read chunks of D matrix
        if N_T % N_PARTITIONS != 0:
            tot_blocks_col = N_PARTITIONS + 1
        else:
            tot_blocks_col = N_PARTITIONS

        di_list = []
        for k in tqdm(range(tot_blocks_col)):
            di = np.load(FOLDER_OUT + f"/data_partitions/di_{k + 1}.npz")['di']
            di_list.append(di)
        D_rec = np.concatenate(di_list,axis=1)

        D_MR = (self.D_orig - np.ones(np.shape(self.D)) * self.D_MEAN.reshape(-1, 1))
        np.testing.assert_allclose(D_rec,D_MR ,rtol=self.rtol, atol=self.atol)

if __name__ == "__main__":
    unittest.main(verbosity=2)
