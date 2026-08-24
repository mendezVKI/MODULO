import tempfile
import unittest

import numpy as np

from modulo_vki.core._k_matrix import CorrelationMatrix
from modulo_vki.core.spatial_structures import Spatial_basis_POD
from modulo_vki.core.temporal_structures import Temporal_basis_POD
from modulo_vki.utils.read_db import ReadData


class TestPhi(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.data = rng.normal(size=(23, 30))
        self.n_modes = 5
        self.n_partitions = 4
        self.tmp = tempfile.TemporaryDirectory()
        correlation = CorrelationMatrix(self.data.shape[1], D=self.data, verbose=False)
        self.psi, self.sigma = Temporal_basis_POD(
            correlation, n_Modes=self.n_modes, verbose=False
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_phi_pod_matches_partitioned_computation(self):
        phi = Spatial_basis_POD(
            self.data,
            N_T=self.data.shape[1],
            PSI_P=self.psi,
            Sigma_P=self.sigma,
            MEMORY_SAVING=False,
            rescale=False,
            verbose=False,
        )

        ReadData._data_processing(
            D=self.data,
            N_PARTITIONS=self.n_partitions,
            MR=False,
            FOLDER_OUT=self.tmp.name,
        )
        CorrelationMatrix(
            self.data.shape[1],
            N_PARTITIONS=self.n_partitions,
            MEMORY_SAVING=True,
            FOLDER_OUT=self.tmp.name,
            verbose=False,
        )
        correlation = np.load(
            self.tmp.name + "/correlation_matrix/k_matrix.npz"
        )["K"]
        psi, sigma = Temporal_basis_POD(
            correlation, n_Modes=self.n_modes, verbose=False
        )
        phi_partitioned = Spatial_basis_POD(
            np.array([1]),
            N_T=self.data.shape[1],
            PSI_P=psi,
            Sigma_P=sigma,
            MEMORY_SAVING=True,
            FOLDER_OUT=self.tmp.name,
            N_PARTITIONS=self.n_partitions,
            verbose=False,
        )

        np.testing.assert_allclose(
            np.abs(phi), np.abs(phi_partitioned), rtol=1e-5, atol=1e-7
        )

    def test_rescale_recomputes_sigma_and_normalizes_modes(self):
        sigma = np.full(self.n_modes, np.nan)
        phi = Spatial_basis_POD(
            self.data,
            N_T=self.data.shape[1],
            PSI_P=self.psi,
            Sigma_P=sigma,
            MEMORY_SAVING=False,
            rescale=True,
            verbose=False,
        )

        projected = self.data @ self.psi
        np.testing.assert_allclose(sigma, np.linalg.norm(projected, axis=0))
        np.testing.assert_allclose(np.linalg.norm(phi, axis=0), 1.0)
        np.testing.assert_allclose(phi * sigma, projected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
