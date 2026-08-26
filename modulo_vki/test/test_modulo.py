import contextlib
import io
import tempfile
import unittest

import numpy as np

from modulo_vki import ModuloVKI, __version__
from modulo_vki.core.temporal_structures import temporal_basis_mPOD
from modulo_vki.utils.read_db import ReadData


def synthetic_dataset(seed=42, n_space=36, n_time=128, fs=128.0):
    """Create a deterministic, low-rank signal with two frequency bands."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_time) / fs
    spatial = np.linalg.qr(rng.normal(size=(n_space, 4)))[0]
    temporal = np.column_stack(
        [
            np.sin(2 * np.pi * 5 * time),
            np.cos(2 * np.pi * 5 * time),
            np.sin(2 * np.pi * 22 * time),
            np.cos(2 * np.pi * 22 * time),
        ]
    )
    data = spatial @ np.diag([4.0, 3.0, 2.0, 1.0]) @ temporal.T
    data += 1e-6 * rng.normal(size=data.shape)
    return data


class TestModulo(unittest.TestCase):
    def setUp(self):
        self.fs = 128.0
        self.data = synthetic_dataset(fs=self.fs)
        self.n_modes = 4
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def make_modulo(self, **kwargs):
        params = {
            "data": self.data,
            "n_Modes": self.n_modes,
            "dtype": "float64",
            "FOLDER_OUT": self.tmp.name + "/",
        }
        params.update(kwargs)
        return ModuloVKI(**params)

    def test_version_and_initialization(self):
        self.assertEqual(__version__, "3.0.0")
        model = self.make_modulo(weights=np.ones(self.data.shape[0]))
        np.testing.assert_allclose(model.D, self.data)
        np.testing.assert_allclose(model.Dstar, self.data)
        self.assertEqual(model.N_S, self.data.shape[0])
        self.assertEqual(model.N_T, self.data.shape[1])

        with self.assertRaises(TypeError):
            ModuloVKI(data=None)
        with self.assertRaises(ValueError):
            ModuloVKI(data=self.data, N_PARTITIONS=2)
        with self.assertRaises(NotImplementedError):
            self.make_modulo(eig_solver="unknown")

    def test_pod_backends_agree(self):
        model = self.make_modulo()
        with contextlib.redirect_stdout(io.StringIO()):
            phi_k, psi_k, sigma_k = model.POD(mode="K", verbose=False)
            phi_s, psi_s, sigma_s = model.POD(mode="svd", verbose=False)

        self.assertEqual(phi_k.shape, (self.data.shape[0], self.n_modes))
        self.assertEqual(psi_k.shape, (self.data.shape[1], self.n_modes))
        self.assertEqual(sigma_k.shape, (self.n_modes,))
        np.testing.assert_allclose(sigma_k, sigma_s, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            phi_k @ phi_k.T, phi_s @ phi_s.T, rtol=1e-5, atol=1e-7
        )

    def test_public_decomposition_interfaces(self):
        model = self.make_modulo()
        with contextlib.redirect_stdout(io.StringIO()):
            phi_dft, freq_dft, sigma_dft = model.DFT(self.fs)
            phi_dmd, eig_dmd, freq_dmd, amp_dmd = model.DMD(
                SAVE_T_DMD=False, F_S=self.fs, verbose=False
            )
            phi_t, sigma_t, freq_t = model.SPOD(
                mode="towne",
                F_S=self.fs,
                n_Modes=2,
                SAVE_SPOD=False,
                L_B=32,
                O_B=16,
            )
            phi_k, psi_k, sigma_k = model.kPOD(
                M_DIST=[0, 16], n_Modes=self.n_modes
            )

        self.assertEqual(phi_dft.shape, self.data.shape)
        self.assertEqual(freq_dft.shape, (self.data.shape[1],))
        self.assertEqual(sigma_dft.shape, (self.data.shape[1],))
        self.assertEqual(phi_dmd.shape, (self.data.shape[0], self.n_modes))
        self.assertEqual(eig_dmd.shape, (self.n_modes,))
        self.assertEqual(freq_dmd.shape, (self.n_modes,))
        self.assertEqual(amp_dmd.shape, (self.n_modes,))
        self.assertEqual(phi_t.shape[0:2], (self.data.shape[0], 2))
        self.assertEqual(sigma_t.shape[0], 2)
        self.assertEqual(phi_t.shape[2], freq_t.size)
        self.assertEqual(phi_k.shape, (self.data.shape[0], self.n_modes))
        self.assertEqual(psi_k.shape, (self.data.shape[1], self.n_modes))
        self.assertEqual(sigma_k.shape, (self.n_modes,))

    def test_sieber_spod_memory_saving_matches_in_memory(self):
        memory_dir = self.tmp.name + "/memory"
        ReadData._data_processing(
            D=self.data.copy(),
            N_PARTITIONS=4,
            MR=False,
            FOLDER_OUT=memory_dir,
        )
        in_memory = self.make_modulo()
        partitioned = ModuloVKI(
            data=None,
            N_T=self.data.shape[1],
            N_S=self.data.shape[0],
            N_PARTITIONS=4,
            n_Modes=self.n_modes,
            FOLDER_OUT=memory_dir + "/",
        )

        kwargs = dict(
            mode="sieber",
            F_S=self.fs,
            n_Modes=self.n_modes,
            SAVE_SPOD=False,
            N_O=8,
            f_c=0.2,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            phi_full, psi_full, sigma_full = in_memory.SPOD(**kwargs)
            phi_part, psi_part, sigma_part = partitioned.SPOD(**kwargs)

        np.testing.assert_allclose(sigma_part, sigma_full, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(
            np.abs(psi_part), np.abs(psi_full), rtol=2e-4, atol=2e-5
        )
        np.testing.assert_allclose(
            np.abs(phi_part), np.abs(phi_full), rtol=2e-4, atol=2e-5
        )
        np.testing.assert_allclose(np.linalg.norm(phi_part, axis=0), 1.0)

    def test_classical_mpod_and_boundary_validation(self):
        model = self.make_modulo()
        kwargs = dict(
            Nf=np.array([21, 21]),
            Ex=21,
            F_V=np.array([16.0]),
            Keep=np.array([1, 1]),
            SAT=8,
            boundaries="reflect",
            MODE="reduced",
            dt=1 / self.fs,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            phi, psi, sigma = model.mPOD(**kwargs, verbose=False)
        self.assertEqual(phi.shape, (self.data.shape[0], self.n_modes))
        self.assertEqual(psi.shape, (self.data.shape[1], self.n_modes))
        self.assertEqual(sigma.shape, (self.n_modes,))

        low_level = dict(kwargs)
        low_level.pop("Ex")
        with self.assertRaises(RuntimeError):
            temporal_basis_mPOD(self.data.T @ self.data, Ex=20, **low_level)


if __name__ == "__main__":
    unittest.main(verbosity=2)
