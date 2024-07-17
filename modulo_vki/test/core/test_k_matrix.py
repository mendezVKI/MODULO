import unittest
import numpy as np
import os, shutil

from modulo_vki.core._k_matrix import CorrelationMatrix
from modulo_vki.utils.read_db import ReadData # to read the data

class TestKMatrix(unittest.TestCase):

    def setUp(self):
        self.D = np.ones((10,10))*2 # dummy data matrix
        self.K = np.ones((10,10))*40 # result of D^T dot D
        # For the tests of the Memory saving feature
        self.FOLDER_OUT = './'
        self.N_PARTITIONS = 10
        # Prepare 10 partitions
        # Divide the D matrix in N_PARTITIONS and save it as npz files in FOLDER_OUT/data_partitions
        _ = ReadData._data_processing(D=self.D,N_PARTITIONS=self.N_PARTITIONS, MR=False,
                                      FOLDER_OUT=self.FOLDER_OUT)
    # Remove test files
    def tearDown(self):
        if os.path.isdir("data_matrix"):
            shutil.rmtree("data_matrix")
        if os.path.isdir("correlation_matrix"):
            shutil.rmtree("correlation_matrix")
        if os.path.isdir("data_partitions"):
            shutil.rmtree("data_partitions")

    def test_dotProduct_memorySavingOff(self):
        """Test the computation of the K matrix w/o memory saving"""
        K_test = CorrelationMatrix(np.shape(self.D)[1], D=self.D)
        self.assertListEqual(K_test.tolist(), self.K.tolist())

    def test_dotProduct_memorySavingOn(self):
        """Test the computation of the K matrix w memory saving"""
        _ = CorrelationMatrix(np.shape(self.D)[1], N_PARTITIONS=self.N_PARTITIONS,MEMORY_SAVING=True, FOLDER_OUT=self.FOLDER_OUT) #return None
        # Load K matrix
        K_test = np.load(self.FOLDER_OUT + '/correlation_matrix/k_matrix.npz')['K']
        self.assertListEqual(K_test.tolist(), self.K.tolist())

if __name__ == "__main__":
    unittest.main(verbosity=2)
