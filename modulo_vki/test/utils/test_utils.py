import unittest
import numpy as np
import os, shutil

from modulo_vki.utils._utils import switch_eigs

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.eig_solvers = ['svd_sklearn_randomized', 'eigh','eigsh']
        self.A = np.array([[3.0, 1.0, 1.0],[1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]) # real symmetric matrix
        self.sqrteigenValues = [np.sqrt(5),np.sqrt(2),np.sqrt(2)] # eigen values computed analytically
        self.eigenVetors = [np.array([1/np.sqrt(3),0,np.sqrt(2/3)]),np.array([1/np.sqrt(3),1/np.sqrt(2),-1/np.sqrt(6)]),np.array([1/np.sqrt(3),-1/np.sqrt(2),-1/np.sqrt(6)])]

    def test_svd(self):
        """ Test the eigen values computed by svd_sklearn_randomized"""
        n_modes = 3
        eig_solver = self.eig_solvers[0]
        Psi_P, Sigma_P  = switch_eigs(self.A, n_modes, eig_solver)
        np.testing.assert_allclose(Sigma_P, self.sqrteigenValues,rtol=1e-4, atol=1e-4)
        #np.testing.assert_allclose(np.abs(Psi_P), np.abs(self.eigenVetors),rtol=1e-4, atol=1e-4)

    def test_eigh(self):
        """ Test the eigen values computed by eigh"""
        n_modes = 3
        eig_solver = self.eig_solvers[1]
        Psi_P, Sigma_P  = switch_eigs(self.A, n_modes, eig_solver)
        np.testing.assert_allclose(Sigma_P, self.sqrteigenValues,rtol=1e-4, atol=1e-4)
        #np.testing.assert_allclose(np.abs(Psi_P), np.abs(self.eigenVetors),rtol=1e-4, atol=1e-4)

    def test_eighs(self):
        """ Test the eigen values computed by eigsh"""
        n_modes = 2
        eig_solver = self.eig_solvers[2]
        Psi_P, Sigma_P  = switch_eigs(self.A, n_modes, eig_solver)
        np.testing.assert_allclose(Sigma_P, self.sqrteigenValues[:-1],rtol=1e-4, atol=1e-4)
        #np.testing.assert_allclose(Psi_P[:-1], self.eigenVetors[:-1],rtol=1e-4, atol=1e-4) # check only the first sqrt eigen value since the others are nan


    def test_eighs(self):
        """ Test the eigen values computed by eigsh"""
        n_modes = 2
        eig_solver = self.eig_solvers[2]
        Psi_P, Sigma_P  = switch_eigs(self.A, n_modes, eig_solver)
        self.assertAlmostEqual(Sigma_P.tolist()[0], self.sqrteigenValues[0])

    def test_unknownEigh(self):
        """Test if an exception is raised with an unknown eigen value solver"""
        with self.assertRaises(ValueError):
            switch_eigs(self.A, 3, 'unknown_eig_solver')

if __name__ == "__main__":
    unittest.main(verbosity=2)